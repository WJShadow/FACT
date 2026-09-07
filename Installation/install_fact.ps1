[CmdletBinding()]
param(
    [ValidateSet('Auto', 'Online', 'Offline')]
    [string]$Mode = 'Auto',
    [switch]$RequireGpu
)

$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentName = 'fact_py311'
$EnvironmentFile = Join-Path $ScriptRoot 'environment_fact_py311.yml'
$ReferenceManifest = Join-Path $ScriptRoot 'reference\fact_py311_reference.json'
$PayloadRoot = Join-Path $ScriptRoot 'payloads'
$OnlinePayloadRoot = Join-Path $PayloadRoot 'online'
$OfflinePayloadRoot = Join-Path $PayloadRoot 'offline'

$FactRoot = Join-Path $env:LOCALAPPDATA 'FACT'
$PrivateCondaRoot = Join-Path $FactRoot 'Miniconda3-24.5.0'
$PrivateCondaExe = Join-Path $PrivateCondaRoot 'Scripts\conda.exe'
$PrivateCondarc = Join-Path $FactRoot 'condarc.fact_py311.yml'
$PrivateCondaPackageCache = Join-Path $FactRoot 'cache\conda-pkgs'
$PrivatePipCache = Join-Path $FactRoot 'cache\pip'
$EnvironmentPath = Join-Path $env:USERPROFILE ".conda\envs\$EnvironmentName"
$EnvironmentParent = Split-Path -Parent $EnvironmentPath
$StateRoot = Join-Path $FactRoot 'state'
$InstallMarker = Join-Path $StateRoot 'fact_py311.installing.json'
$LogRoot = Join-Path $FactRoot 'logs'
$LogPath = Join-Path $LogRoot ("install_fact_py311_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

$MinicondaFileName = 'Miniconda3-py311_24.5.0-0-Windows-x86_64.exe'
$MinicondaUrl = "https://repo.anaconda.com/miniconda/$MinicondaFileName"
$MinicondaSha256 = 'b0cc3e41f42e6cf491fd63ea9e87dccd7160e5328d542602873c69217179bc3b'
$Hdf5FileName = 'hdf5-1.14.3-nompi_h2b43c12_105.conda'
$Hdf5Sha256 = '56c803607a64b5117a8b4bcfdde722e4fa40970ddc4c61224b0981cbb70fb005'

$TranscriptStarted = $false
$OldCondarc = $null
$HadCondarc = $false
$OldCondaOffline = $null
$HadCondaOffline = $false
$OldPipNoIndex = $null
$HadPipNoIndex = $false
$OldFactCondaExe = $null
$HadFactCondaExe = $false
$OldCondaPkgsDirs = $null
$HadCondaPkgsDirs = $false
$OldPipCacheDir = $null
$HadPipCacheDir = $false
$ExitCode = 1
$PathAddedByThisRun = $false

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host $Message
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}

function Assert-SafeEnvironmentPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $expectedRoot = [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.conda\envs')).TrimEnd('\')
    $expected = [IO.Path]::GetFullPath($EnvironmentPath).TrimEnd('\')
    if (-not [String]::Equals($full, $expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify an unexpected environment path: $full"
    }
    if (-not $full.StartsWith($expectedRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the user Conda environment directory: $full"
    }
}

function Remove-EnvironmentSafely {
    if (-not (Test-Path -LiteralPath $EnvironmentPath)) { return }
    Assert-SafeEnvironmentPath -Path $EnvironmentPath
    Write-Log "Removing incomplete environment: $EnvironmentPath"
    [IO.Directory]::Delete([IO.Path]::GetFullPath($EnvironmentPath), $true)
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][AllowEmptyCollection()][string[]]$ArgumentList = @(),
        [string]$Step = $FilePath
    )
    Write-Log "Running: $Step"
    & $FilePath @ArgumentList
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Step failed with exit code $code" }
}

function Ensure-VerifiedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [string]$Url,
        [switch]$AllowDownload
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (-not (Test-Path -LiteralPath $Path)) {
        if (-not $AllowDownload -or -not $Url) { throw "Required local payload is missing: $Path" }
        $partial = "$Path.download"
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                if (Test-Path -LiteralPath $partial) { [IO.File]::Delete($partial) }
                Write-Log "Downloading payload (attempt $attempt/3): $Url"
                Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
                Move-Item -LiteralPath $partial -Destination $Path -Force
                break
            }
            catch {
                if (Test-Path -LiteralPath $partial) { [IO.File]::Delete($partial) }
                if ($attempt -eq 3) { throw }
                Start-Sleep -Seconds (2 * $attempt)
            }
        }
    }
    $actual = Get-Sha256 -Path $Path
    if (-not [String]::Equals($actual, $ExpectedSha256, [StringComparison]::OrdinalIgnoreCase)) {
        throw "SHA-256 mismatch for $Path. Expected $ExpectedSha256; actual $actual"
    }
    Write-Log "Verified SHA-256: $Path"
    return $Path
}

function Verify-PayloadManifest {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$PayloadRoot
    )
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Payload manifest is missing: $ManifestPath"
    }
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ([int]$manifest.schema -ne 1) { throw "Unsupported payload manifest schema in $ManifestPath" }
    if ([bool]$manifest.containsCodeOrData) {
        throw "Refusing a payload that declares code or data content: $ManifestPath"
    }
    if ([string]$manifest.environmentName -ne $EnvironmentName) {
        throw "Payload manifest is for an unexpected environment: $ManifestPath"
    }
    $scriptRootFull = [IO.Path]::GetFullPath($ScriptRoot).TrimEnd('\') + '\'
    $entries = @($manifest.files)
    if ($entries.Count -eq 0) { throw "Payload manifest contains no files: $ManifestPath" }
    foreach ($entry in $entries) {
        $relative = [string]$entry.path
        if ([String]::IsNullOrWhiteSpace($relative) -or [String]::IsNullOrWhiteSpace([string]$entry.sha256)) {
            throw "Payload manifest contains an incomplete entry: $ManifestPath"
        }
        $candidate = [IO.Path]::GetFullPath((Join-Path $PayloadRoot $relative))
        if (-not $candidate.StartsWith($scriptRootFull, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Payload manifest points outside the installer directory: $relative"
        }
        Ensure-VerifiedFile -Path $candidate -ExpectedSha256 ([string]$entry.sha256) | Out-Null
        if ($entry.size -ne $null) {
            $actualSize = ([IO.FileInfo]$candidate).Length
            if ([int64]$entry.size -ne $actualSize) {
                throw "Size mismatch for payload $relative. Expected $($entry.size); actual $actualSize"
            }
        }
    }
    Write-Log "Verified payload manifest: $ManifestPath"
    return $manifest
}

function Set-PrivateCondaEnvironment {
    New-Item -ItemType Directory -Path $FactRoot -Force | Out-Null
    $config = @'
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
channel_priority: flexible
auto_activate_base: false
changeps1: false
remote_max_retries: 3
remote_connect_timeout_secs: 30
remote_read_timeout_secs: 120
'@
    Set-Content -LiteralPath $PrivateCondarc -Value $config -Encoding UTF8
    $script:HadCondarc = $null -ne $env:CONDARC
    $script:OldCondarc = $env:CONDARC
    $env:CONDARC = $PrivateCondarc
    $script:HadCondaPkgsDirs = $null -ne $env:CONDA_PKGS_DIRS
    $script:OldCondaPkgsDirs = $env:CONDA_PKGS_DIRS
    $env:CONDA_PKGS_DIRS = $PrivateCondaPackageCache
    $script:HadPipCacheDir = $null -ne $env:PIP_CACHE_DIR
    $script:OldPipCacheDir = $env:PIP_CACHE_DIR
    $env:PIP_CACHE_DIR = $PrivatePipCache
    New-Item -ItemType Directory -Path $PrivateCondaPackageCache,$PrivatePipCache -Force | Out-Null
}

function Ensure-PrivateConda {
    param([Parameter(Mandatory = $true)][ValidateSet('Online', 'Offline')][string]$InstallMode)
    if (Test-Path -LiteralPath $PrivateCondaExe) {
        $version = (& $PrivateCondaExe --version 2>&1 | Out-String).Trim()
        if ($version -notmatch '^conda 24\.5\.0$') { throw "Private Conda has unexpected version: $version" }
        Write-Log "Using verified private Conda: $PrivateCondaExe"
        return
    }
    $localPayload = if ($InstallMode -eq 'Offline') { Join-Path $OfflinePayloadRoot $MinicondaFileName } else { Join-Path $OnlinePayloadRoot $MinicondaFileName }
    $cachePath = Join-Path (Join-Path $FactRoot 'cache') $MinicondaFileName
    if (Test-Path -LiteralPath $localPayload) {
        $installer = Ensure-VerifiedFile -Path $localPayload -ExpectedSha256 $MinicondaSha256
    } else {
        $installer = Ensure-VerifiedFile -Path $cachePath -ExpectedSha256 $MinicondaSha256 -Url $MinicondaUrl -AllowDownload:($InstallMode -eq 'Online')
    }
    Write-Log "Installing private Conda runtime: $PrivateCondaRoot"
    New-Item -ItemType Directory -Path $FactRoot -Force | Out-Null
    $arguments = @('/S', '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', "/D=$PrivateCondaRoot")
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Private Conda installer failed with exit code $($process.ExitCode)" }
    if (-not (Test-Path -LiteralPath $PrivateCondaExe)) { throw "Private Conda conda.exe was not found: $PrivateCondaExe" }
    $version = (& $PrivateCondaExe --version 2>&1 | Out-String).Trim()
    if ($version -notmatch '^conda 24\.5\.0$') { throw "Installed private Conda has unexpected version: $version" }
    Write-Log "Private Conda installation verified: $version"
}

function Add-PrivateCondaToUserPathIfNeeded {
    param([Parameter(Mandatory = $true)][bool]$HadCondaBeforeInstall)
    if ($HadCondaBeforeInstall) {
        Write-Log 'An existing Conda was detected; user PATH was left unchanged.'
        return $false
    }
    $condabin = Join-Path $PrivateCondaRoot 'condabin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = if ($userPath) { @($userPath -split ';' | Where-Object { $_ }) } else { @() }
    $alreadyPresent = @($entries | Where-Object { [String]::Equals($_.TrimEnd('\'), $condabin.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
    if (-not $alreadyPresent) {
        [Environment]::SetEnvironmentVariable('Path', (($entries + $condabin) -join ';'), 'User')
        Write-Log "Added private Conda to the user PATH: $condabin"
        return $true
    }
    return $false
}

function Remove-PrivateCondaFromUserPath {
    $condabin = Join-Path $PrivateCondaRoot 'condabin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { return }
    $entries = @($userPath -split ';' | Where-Object {
        $_ -and -not [String]::Equals($_.TrimEnd('\'), $condabin.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
    })
    [Environment]::SetEnvironmentVariable('Path', ($entries -join ';'), 'User')
    Write-Log "Removed the private Conda PATH entry after the failed installation: $condabin"
}

function Register-EnvironmentPath {
    $condaHome = Join-Path $env:USERPROFILE '.conda'
    $registry = Join-Path $condaHome 'environments.txt'
    New-Item -ItemType Directory -Path $condaHome -Force | Out-Null
    $full = [IO.Path]::GetFullPath($EnvironmentPath)
    $existing = if (Test-Path -LiteralPath $registry) { @(Get-Content -LiteralPath $registry | ForEach-Object { [string]$_ }) } else { @() }
    $found = @($existing | Where-Object { [String]::Equals($_.TrimEnd('\'), $full.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
    if (-not $found) {
        Add-Content -LiteralPath $registry -Value $full -Encoding UTF8
        Write-Log "Registered environment path: $full"
    }
}

function Register-JupyterKernel {
    $python = Join-Path $EnvironmentPath 'python.exe'
    Write-Log 'Registering (or updating) the fact_py311 Jupyter kernel.'
    Invoke-Checked -FilePath $python -ArgumentList @('-m', 'ipykernel', 'install', '--user', '--name', $EnvironmentName, '--display-name', 'Python (fact_py311)') -Step 'Registering the fact_py311 Jupyter kernel'
}

function Remove-StalePipCondaMetadata {
    $records = @(
        (Join-Path $EnvironmentPath 'conda-meta\mpmath-1.4.1-pyhd8ed1ab_0.json'),
        (Join-Path $EnvironmentPath 'conda-meta\sympy-1.14.0-pyh04b8f61_6.json')
    )
    foreach ($record in $records) {
        if (-not (Test-Path -LiteralPath $record)) { throw "Expected Conda metadata was not found: $record" }
    }
    foreach ($record in $records) { Remove-Item -LiteralPath $record -Force }
    foreach ($record in $records) {
        if (Test-Path -LiteralPath $record) { throw "Could not remove stale Conda metadata: $record" }
    }
    Write-Log 'Removed the two stale Conda records replaced by the pinned pip packages.'
}

function Install-OnlineEnvironment {
    Ensure-PrivateConda -InstallMode Online
    Set-PrivateCondaEnvironment
    if (-not (Test-Path -LiteralPath $EnvironmentFile)) { throw "Environment file not found: $EnvironmentFile" }
    Verify-PayloadManifest -ManifestPath (Join-Path $OnlinePayloadRoot 'payload-manifest.json') -PayloadRoot $OnlinePayloadRoot | Out-Null
    $hdf5Path = Join-Path $OnlinePayloadRoot $Hdf5FileName
    Ensure-VerifiedFile -Path $hdf5Path -ExpectedSha256 $Hdf5Sha256
    New-Item -ItemType Directory -Path $EnvironmentParent -Force | Out-Null
    Invoke-Checked -FilePath $PrivateCondaExe -ArgumentList @('env', 'create', '--yes', '--file', $EnvironmentFile, '--prefix', $EnvironmentPath) -Step 'Creating the Conda and pip environment from the locked YML'
    Invoke-Checked -FilePath $PrivateCondaExe -ArgumentList @('install', '--yes', '--no-update-deps', '--prefix', $EnvironmentPath, $hdf5Path) -Step 'Applying the source environment HDF5 build'
    Remove-StalePipCondaMetadata
}

function Get-ArchiveRoot {
    param([Parameter(Mandatory = $true)][string]$StagingPath)
    if (Test-Path -LiteralPath (Join-Path $StagingPath 'python.exe')) { return $StagingPath }
    $children = @(Get-ChildItem -LiteralPath $StagingPath -Directory)
    $candidate = @($children | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'python.exe') })
    if ($candidate.Count -eq 1) { return $candidate[0].FullName }
    throw 'The offline archive does not contain one identifiable Conda environment root.'
}

function Install-OfflineEnvironment {
    Ensure-PrivateConda -InstallMode Offline
    Set-PrivateCondaEnvironment
    $archive = Join-Path $OfflinePayloadRoot 'fact_py311-win64.tar.gz'
    Verify-PayloadManifest -ManifestPath (Join-Path $OfflinePayloadRoot 'payload-manifest.json') -PayloadRoot $OfflinePayloadRoot | Out-Null
    if (-not (Test-Path -LiteralPath $archive)) { throw "Offline environment archive is missing: $archive" }
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $tar) { throw 'Windows tar.exe is required for the offline archive.' }
    New-Item -ItemType Directory -Path $EnvironmentParent -Force | Out-Null
    $staging = "$EnvironmentPath.__staging_$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    try {
        Invoke-Checked -FilePath $tar.Source -ArgumentList @('-xzf', $archive, '-C', $staging) -Step 'Extracting the offline fact_py311 archive'
        $archiveRoot = Get-ArchiveRoot -StagingPath $staging
        if (Test-Path -LiteralPath $EnvironmentPath) { throw "The target environment appeared during installation: $EnvironmentPath" }
        Move-Item -LiteralPath $archiveRoot -Destination $EnvironmentPath
        if ($archiveRoot -ne $staging -and (Test-Path -LiteralPath $staging)) { [IO.Directory]::Delete([IO.Path]::GetFullPath($staging), $true) }
        $unpackExe = Join-Path $EnvironmentPath 'Scripts\conda-unpack.exe'
        $unpackBat = Join-Path $EnvironmentPath 'Scripts\conda-unpack.bat'
        if (Test-Path -LiteralPath $unpackExe) {
            Invoke-Checked -FilePath $unpackExe -ArgumentList @() -Step 'Relocating the offline environment prefix'
        } elseif (Test-Path -LiteralPath $unpackBat) {
            Invoke-Checked -FilePath 'cmd.exe' -ArgumentList @('/d', '/c', 'call', $unpackBat) -Step 'Relocating the offline environment prefix'
        } else {
            throw 'conda-pack unpack helper was not found in the offline environment.'
        }
    }
    finally {
        if (Test-Path -LiteralPath $staging) { [IO.Directory]::Delete([IO.Path]::GetFullPath($staging), $true) }
    }
}

function Invoke-EnvironmentVerification {
    $verifyScript = Join-Path $ScriptRoot 'Verify_FACT_Environment.ps1'
    if (-not (Test-Path -LiteralPath $verifyScript)) { throw "Verification script not found: $verifyScript" }
    if (-not (Test-Path -LiteralPath $ReferenceManifest)) { throw "Reference manifest not found: $ReferenceManifest" }
    $script:HadFactCondaExe = $null -ne $env:FACT_CONDA_EXE
    $script:OldFactCondaExe = $env:FACT_CONDA_EXE
    $env:FACT_CONDA_EXE = $PrivateCondaExe
    $args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $verifyScript, '-Prefix', $EnvironmentPath, '-ReferencePath', $ReferenceManifest, '-SourcePrefix', $EnvironmentPath)
    if ($RequireGpu) { $args += '-RequireGpu' }
    Invoke-Checked -FilePath 'powershell.exe' -ArgumentList $args -Step 'Verifying package, binary, runtime and pip-check equivalence'
}

function Resolve-InstallMode {
    if ($Mode -ne 'Auto') { return $Mode }
    if (Test-Path -LiteralPath (Join-Path $OfflinePayloadRoot 'fact_py311-win64.tar.gz')) { return 'Offline' }
    return 'Online'
}

try {
    New-Item -ItemType Directory -Path $LogRoot,$StateRoot -Force | Out-Null
    Start-Transcript -LiteralPath $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    $resolvedMode = Resolve-InstallMode
    Write-Log "FACT fact_py311 installer mode: $resolvedMode"
    Write-Log "Target environment: $EnvironmentPath"
    $hadCondaBeforeInstall = $null -ne (Get-Command conda.exe -ErrorAction SilentlyContinue) -or $null -ne (Get-Command conda -ErrorAction SilentlyContinue)

    if (Test-Path -LiteralPath $InstallMarker) {
        $marker = Get-Content -LiteralPath $InstallMarker -Raw | ConvertFrom-Json
        if (-not [String]::Equals([string]$marker.environmentPath, [IO.Path]::GetFullPath($EnvironmentPath), [StringComparison]::OrdinalIgnoreCase)) { throw "An installation marker belongs to another path: $InstallMarker" }
        Write-Log 'A previous interrupted FACT installation was found; cleaning its incomplete target.'
        Remove-EnvironmentSafely
        Remove-Item -LiteralPath $InstallMarker -Force
    }
    if (Test-Path -LiteralPath $EnvironmentPath) { throw "The target path already exists. Refusing to overwrite: $EnvironmentPath" }

    $markerObject = [ordered]@{
        environmentName = $EnvironmentName
        environmentPath = [IO.Path]::GetFullPath($EnvironmentPath)
        mode = (Resolve-InstallMode)
        started = (Get-Date).ToUniversalTime().ToString('o')
    }
    $markerObject | ConvertTo-Json | Set-Content -LiteralPath $InstallMarker -Encoding UTF8

    if ($resolvedMode -eq 'Online') {
        Install-OnlineEnvironment
    } else {
        $script:HadCondaOffline = $null -ne $env:CONDA_OFFLINE
        $script:OldCondaOffline = $env:CONDA_OFFLINE
        $env:CONDA_OFFLINE = 'true'
        $script:HadPipNoIndex = $null -ne $env:PIP_NO_INDEX
        $script:OldPipNoIndex = $env:PIP_NO_INDEX
        $env:PIP_NO_INDEX = '1'
        Install-OfflineEnvironment
    }

    Invoke-EnvironmentVerification
    $PathAddedByThisRun = Add-PrivateCondaToUserPathIfNeeded -HadCondaBeforeInstall $hadCondaBeforeInstall
    Register-EnvironmentPath
    Register-JupyterKernel
    Remove-Item -LiteralPath $InstallMarker -Force
    Write-Log 'FACT fact_py311 installation completed successfully.'
    $ExitCode = 0
}
catch {
    Write-Log ("ERROR: " + $_.Exception.Message)
    Write-Log 'The installation did not complete.'
    try {
        if ($PathAddedByThisRun) {
            Remove-PrivateCondaFromUserPath
            $PathAddedByThisRun = $false
        }
        if (Test-Path -LiteralPath $InstallMarker) {
            Remove-EnvironmentSafely
            Remove-Item -LiteralPath $InstallMarker -Force
            Write-Log 'Incomplete target cleanup completed.'
        }
    }
    catch {
        Write-Log ("WARNING: Could not clean the incomplete target automatically: " + $_.Exception.Message)
        Write-Log "Remove this exact path before retrying: $EnvironmentPath"
    }
    $ExitCode = 1
}
finally {
    if ($HadCondarc) { $env:CONDARC = $OldCondarc } else { Remove-Item Env:CONDARC -ErrorAction SilentlyContinue }
    if ($HadCondaOffline) { $env:CONDA_OFFLINE = $OldCondaOffline } else { Remove-Item Env:CONDA_OFFLINE -ErrorAction SilentlyContinue }
    if ($HadPipNoIndex) { $env:PIP_NO_INDEX = $OldPipNoIndex } else { Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue }
    if ($HadFactCondaExe) { $env:FACT_CONDA_EXE = $OldFactCondaExe } else { Remove-Item Env:FACT_CONDA_EXE -ErrorAction SilentlyContinue }
    if ($HadCondaPkgsDirs) { $env:CONDA_PKGS_DIRS = $OldCondaPkgsDirs } else { Remove-Item Env:CONDA_PKGS_DIRS -ErrorAction SilentlyContinue }
    if ($HadPipCacheDir) { $env:PIP_CACHE_DIR = $OldPipCacheDir } else { Remove-Item Env:PIP_CACHE_DIR -ErrorAction SilentlyContinue }
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}

exit $ExitCode
