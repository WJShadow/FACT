[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$DemoFolderUrl = 'https://drive.google.com/drive/folders/1wFA6jO82x-OEt2BMVXm9FisyT03diZO8'
$UvVersion = '0.11.14'
$UvArchiveName = 'uv-x86_64-pc-windows-msvc.zip'
$UvBaseUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvArchiveName"
$DemoRoot = Join-Path $PSScriptRoot 'Demo'
$RequiredDemoFiles = @(
    'Mouse hippocampus\Data_CA1.tif',
    'Mouse hippocampus\Data_CA3.tif',
    'Meso\Data_meso_strip.tif',
    'MouseBrainCortex\MouseBrainCortex_quat.tiff',
    'Rhesus\Rhesus_quat.tif'
)

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Read-YesNo([string]$Prompt, [bool]$DefaultYes = $false) {
    $hint = if ($DefaultYes) { '[Y/n]' } else { '[y/N]' }
    $answer = Read-Host "$Prompt $hint"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $DefaultYes }
    return $answer.Trim().ToLowerInvariant() -in @('y', 'yes')
}

function Get-DriveFreeBytes([string]$Path) {
    $root = [IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Path).Path)
    $driveName = $root.TrimEnd('\').TrimEnd(':')
    return (Get-PSDrive -Name $driveName).Free
}

function Assert-FreeSpace([string]$Path, [double]$RequiredGiB) {
    $requiredBytes = [int64]($RequiredGiB * 1GB)
    $freeBytes = Get-DriveFreeBytes $Path
    if ($freeBytes -lt $requiredBytes) {
        throw ("Insufficient free space on the destination drive. At least {0:N1} GiB is required; {1:N1} GiB is available." -f $RequiredGiB, ($freeBytes / 1GB))
    }
}

function New-StageDirectory {
    $stage = Join-Path $PSScriptRoot ('.FACT-download-staging-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -LiteralPath $stage | Out-Null
    return $stage
}

function Get-UvExecutable([string]$Stage) {
    $archivePath = Join-Path $Stage $UvArchiveName
    $checksumPath = "$archivePath.sha256"
    $extractPath = Join-Path $Stage 'uv'

    Write-Info 'Preparing the temporary downloader runtime...'
    Invoke-WebRequest -Uri $UvBaseUrl -OutFile $archivePath -UseBasicParsing
    Invoke-WebRequest -Uri "$UvBaseUrl.sha256" -OutFile $checksumPath -UseBasicParsing

    $checksumText = (Get-Content -LiteralPath $checksumPath -Raw).Trim()
    $expectedHash = ($checksumText -split '\s+')[0].ToLowerInvariant()
    if ($expectedHash -notmatch '^[a-f0-9]{64}$') {
        throw 'The published uv checksum could not be read.'
    }
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw 'The downloaded uv runtime did not match its published SHA-256 checksum.'
    }

    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractPath -Force
    $uv = Get-ChildItem -LiteralPath $extractPath -Filter 'uv.exe' -File -Recurse | Select-Object -First 1
    if ($null -eq $uv) { throw 'The verified uv archive did not contain uv.exe.' }

    $env:UV_CACHE_DIR = Join-Path $Stage 'uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Stage 'uv-python'
    return $uv.FullName
}

function Invoke-Gdown([string]$UvExe, [string[]]$GdownArguments) {
    & $UvExe 'tool' 'run' '--isolated' '--python' '3.12' '--from' 'gdown==6.0.0' 'gdown' @GdownArguments
    if ($LASTEXITCODE -ne 0) { throw "gdown failed with exit code $LASTEXITCODE." }
}

function Get-DriveFolderEntries([string]$UvExe, [string]$FolderUrl) {
    $json = & $UvExe 'tool' 'run' '--isolated' '--python' '3.12' '--from' 'gdown==6.0.0' 'gdown' $FolderUrl '--folder' '--json'
    if ($LASTEXITCODE -ne 0) { throw "Unable to list the Google Drive folder (exit code $LASTEXITCODE)." }
    try {
        $entries = @($json | Out-String | ConvertFrom-Json)
    } catch {
        throw 'Google Drive returned an invalid folder listing. The folder may be unavailable or quota-limited.'
    }
    $files = @($entries | Where-Object { $null -ne $_.url -and $null -ne $_.path })
    if ($files.Count -eq 0) { throw 'The Google Drive folder listing contained no downloadable files.' }
    return $files
}

function Normalize-DrivePath([string]$Path) {
    return $Path.Replace('/', '\').Trim('\')
}

function Find-DriveEntry([object[]]$Entries, [string]$ExpectedPath) {
    $expected = Normalize-DrivePath $ExpectedPath
    $matches = @($Entries | Where-Object {
        $candidate = Normalize-DrivePath $_.path
        $candidate -eq $expected -or $candidate.EndsWith("\$expected", [StringComparison]::OrdinalIgnoreCase)
    })
    if ($matches.Count -ne 1) {
        throw "Expected exactly one Google Drive file for '$ExpectedPath'; found $($matches.Count)."
    }
    return $matches[0]
}

function Test-NotHtmlFile([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $buffer = New-Object byte[] 512
        $read = $stream.Read($buffer, 0, $buffer.Length)
    } finally {
        $stream.Dispose()
    }
    $head = [Text.Encoding]::ASCII.GetString($buffer, 0, $read).TrimStart().ToLowerInvariant()
    if ($head.StartsWith('<!doctype') -or $head.StartsWith('<html') -or $head.StartsWith('<body')) {
        throw "Google Drive returned an HTML page instead of '$Path'."
    }
}

function Get-DemoStatus([string[]]$Targets) {
    $present = @($Targets | Where-Object { Test-Path -LiteralPath (Join-Path $DemoRoot $_) -PathType Leaf })
    return [pscustomobject]@{
        Present = $present
        Missing = @($Targets | Where-Object { $_ -notin $present })
        AllPresent = $present.Count -eq $Targets.Count
    }
}

function Get-Confirmation([pscustomobject]$Status) {
    if ($Status.AllPresent) {
        return Read-YesNo 'All required Demo files already exist. Re-download and replace them?'
    }
    if ($Status.Present.Count -gt 0) {
        Write-Host "Found $($Status.Present.Count) required file(s); $($Status.Missing.Count) are missing."
        return Read-YesNo 'Download the complete package to staging and repair only the missing files?'
    }
    return Read-YesNo 'Download the five required Demo TIFF files (about 6.1 GiB installed)?'
}

function Download-DemoFiles([string]$UvExe, [string]$Stage, [string[]]$Targets) {
    $entries = Get-DriveFolderEntries $UvExe $DemoFolderUrl
    $payloadRoot = Join-Path $Stage 'demo-payload'
    foreach ($target in $Targets) {
        $entry = Find-DriveEntry $entries $target
        $outputPath = Join-Path $payloadRoot $target
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
        Write-Info "Downloading $target"
        Invoke-Gdown $UvExe @($entry.url, '--fuzzy', '-O', $outputPath)
        if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf) -or (Get-Item -LiteralPath $outputPath).Length -eq 0) {
            throw "The download for '$target' is missing or empty."
        }
        Test-NotHtmlFile $outputPath
    }
    return $payloadRoot
}

function Install-DemoFiles([string]$PayloadRoot, [string[]]$Targets, [pscustomobject]$Status, [bool]$ReplaceAll, [string]$Stage) {
    $toInstall = @(if ($ReplaceAll) { $Targets } else { $Status.Missing })
    if ($toInstall.Count -eq 0) { return }

    $backupRoot = Join-Path $Stage 'demo-backup'
    $installed = New-Object Collections.Generic.List[string]
    try {
        if ($ReplaceAll) {
            foreach ($target in $toInstall) {
                $destination = Join-Path $DemoRoot $target
                if (Test-Path -LiteralPath $destination -PathType Leaf) {
                    $backup = Join-Path $backupRoot $target
                    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
                    Move-Item -LiteralPath $destination -Destination $backup -Force
                }
            }
        }
        foreach ($target in $toInstall) {
            $source = Join-Path $PayloadRoot $target
            $destination = Join-Path $DemoRoot $target
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
            Move-Item -LiteralPath $source -Destination $destination -Force
            $installed.Add($target)
        }
    } catch {
        foreach ($target in $installed) {
            $destination = Join-Path $DemoRoot $target
            $source = Join-Path $PayloadRoot $target
            if (Test-Path -LiteralPath $destination -PathType Leaf) {
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $source) | Out-Null
                Move-Item -LiteralPath $destination -Destination $source -Force
            }
        }
        if (Test-Path -LiteralPath $backupRoot) {
            Get-ChildItem -LiteralPath $backupRoot -File -Recurse | ForEach-Object {
                $relative = $_.FullName.Substring($backupRoot.Length).TrimStart('\')
                $restore = Join-Path $DemoRoot $relative
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $restore) | Out-Null
                Move-Item -LiteralPath $_.FullName -Destination $restore -Force
            }
        }
        throw
    }
}

$stage = $null
try {
    Write-Host 'FACT Demo data downloader' -ForegroundColor Green
    Write-Host "Destination: $DemoRoot"
    Write-Host 'Installed size: about 6.1 GiB. Allow at least 6.8 GiB free while downloading.'
    $status = Get-DemoStatus $RequiredDemoFiles
    if (-not (Get-Confirmation $status)) {
        Write-Host 'Cancelled. No files were changed.'
        exit 0
    }

    Assert-FreeSpace $PSScriptRoot 6.8
    $stage = New-StageDirectory
    $uvExe = Get-UvExecutable $stage
    $payload = Download-DemoFiles $uvExe $stage $RequiredDemoFiles
    Install-DemoFiles $payload $RequiredDemoFiles $status $status.AllPresent $stage
    Write-Host 'Demo data download completed successfully.' -ForegroundColor Green
} catch {
    Write-Host "Download failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($null -ne $stage -and (Test-Path -LiteralPath $stage)) {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}
