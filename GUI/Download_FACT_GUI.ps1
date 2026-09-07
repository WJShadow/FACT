[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$GuiFolderUrl = 'https://drive.google.com/drive/folders/1MINB9WKUeGR3EO59gEudEFW1hXIPkLtx'
$DemoFolderUrl = 'https://drive.google.com/drive/folders/1wFA6jO82x-OEt2BMVXm9FisyT03diZO8'
$UvVersion = '0.11.14'
$UvArchiveName = 'uv-x86_64-pc-windows-msvc.zip'
$UvBaseUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvArchiveName"
$GuiRoot = $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $GuiRoot
$DemoRoot = Join-Path (Join-Path $RepositoryRoot 'data') 'Demo'
$AllDemoFiles = @(
    'Mouse hippocampus\Data_CA1.tif',
    'Mouse hippocampus\Data_CA3.tif',
    'Meso\Data_meso_strip.tif',
    'MouseBrainCortex\MouseBrainCortex_quat.tiff',
    'Rhesus\Rhesus_quat.tif'
)
$BasicDemoFiles = @('MouseBrainCortex\MouseBrainCortex_quat.tiff')

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Read-YesNo([string]$Prompt, [bool]$DefaultYes = $false) {
    $hint = if ($DefaultYes) { '[Y/n]' } else { '[y/N]' }
    $answer = Read-Host "$Prompt $hint"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $DefaultYes }
    return $answer.Trim().ToLowerInvariant() -in @('y', 'yes')
}

function Read-Selection {
    while ($true) {
        Write-Host ''
        Write-Host '1. FACT-GUI only - about 4.7 GiB installed; allow about 5.2 GiB free.'
        Write-Host '2. FACT-GUI plus basic demo data - about 6.4 GiB installed; allow about 7.1 GiB free.'
        Write-Host '3. FACT-GUI plus all demo data - about 10.7 GiB installed; allow about 11.8 GiB free.'
        Write-Host '4. Cancel.'
        $choice = (Read-Host 'Choose an option').Trim()
        switch ($choice) {
            '1' { return 'GuiOnly' }
            '2' { return 'BasicDemo' }
            '3' { return 'AllDemo' }
            '4' { return 'Cancel' }
            default { Write-Host 'Enter 1, 2, 3, or 4.' -ForegroundColor Yellow }
        }
    }
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
    $stage = Join-Path $GuiRoot ('.FACT-download-staging-' + [guid]::NewGuid().ToString('N'))
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

function Get-GuiRelativePath([string]$Path) {
    $relative = Normalize-DrivePath $Path
    if ($relative.StartsWith('GUI\', [StringComparison]::OrdinalIgnoreCase)) {
        return $relative.Substring(4)
    }
    return $relative
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

function Get-GuiStatus([object[]]$Entries) {
    $missing = New-Object Collections.Generic.List[object]
    foreach ($entry in $Entries) {
        $relative = Get-GuiRelativePath $entry.path
        if (-not (Test-Path -LiteralPath (Join-Path $GuiRoot $relative) -PathType Leaf)) {
            $missing.Add($entry)
        }
    }
    $allPresent = $missing.Count -eq 0 -and
        (Test-Path -LiteralPath (Join-Path $GuiRoot 'FACT-Pipeline.exe') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $GuiRoot 'FACT-Pipeline.runtime') -PathType Container)
    return [pscustomobject]@{
        Total = $Entries.Count
        Missing = $missing.ToArray()
        PresentCount = $Entries.Count - $missing.Count
        AllPresent = $allPresent
    }
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

function Find-StagedGuiRoot([string]$PayloadRoot) {
    $executables = @(Get-ChildItem -LiteralPath $PayloadRoot -Filter 'FACT-Pipeline.exe' -File -Recurse)
    if ($executables.Count -ne 1) { throw 'The GUI download did not contain exactly one FACT-Pipeline.exe.' }
    $sourceRoot = $executables[0].Directory.FullName
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'FACT-Pipeline.runtime') -PathType Container)) {
        throw 'The GUI download did not contain FACT-Pipeline.runtime beside FACT-Pipeline.exe.'
    }
    return $sourceRoot
}

function Download-Gui([string]$UvExe, [string]$Stage, [object[]]$Entries) {
    $payloadRoot = Join-Path $Stage 'gui-payload'
    Write-Info 'Downloading FACT-GUI. This may take some time.'
    Invoke-Gdown $UvExe @($GuiFolderUrl, '--folder', '--fuzzy', '-O', $payloadRoot)
    $sourceRoot = Find-StagedGuiRoot $payloadRoot
    foreach ($entry in $Entries) {
        $relative = Get-GuiRelativePath $entry.path
        if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot $relative) -PathType Leaf)) {
            throw "The staged GUI download is missing '$relative'."
        }
    }
    return $sourceRoot
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

function Install-Gui([string]$SourceRoot, [object[]]$Entries, [pscustomobject]$Status, [bool]$ReplaceAll, [string]$Stage) {
    $backupRoot = Join-Path $Stage 'gui-backup'
    $placed = New-Object Collections.Generic.List[string]
    try {
        if ($ReplaceAll) {
            foreach ($name in @('FACT-Pipeline.exe', 'FACT-Pipeline.runtime')) {
                $destination = Join-Path $GuiRoot $name
                if (Test-Path -LiteralPath $destination) {
                    New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
                    Move-Item -LiteralPath $destination -Destination (Join-Path $backupRoot $name) -Force
                }
            }
            foreach ($name in @('FACT-Pipeline.exe', 'FACT-Pipeline.runtime')) {
                Move-Item -LiteralPath (Join-Path $SourceRoot $name) -Destination (Join-Path $GuiRoot $name) -Force
                $placed.Add($name)
            }
            return
        }

        if ($Status.PresentCount -eq 0) {
            foreach ($name in @('FACT-Pipeline.exe', 'FACT-Pipeline.runtime')) {
                Move-Item -LiteralPath (Join-Path $SourceRoot $name) -Destination (Join-Path $GuiRoot $name) -Force
                $placed.Add($name)
            }
            return
        }

        foreach ($entry in $Status.Missing) {
            $relative = Get-GuiRelativePath $entry.path
            $source = Join-Path $SourceRoot $relative
            $destination = Join-Path $GuiRoot $relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
            Copy-Item -LiteralPath $source -Destination $destination -Force
            $placed.Add($relative)
        }
    } catch {
        if ($ReplaceAll) {
            foreach ($name in $placed) {
                $destination = Join-Path $GuiRoot $name
                if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
            }
            foreach ($name in @('FACT-Pipeline.exe', 'FACT-Pipeline.runtime')) {
                $backup = Join-Path $backupRoot $name
                if (Test-Path -LiteralPath $backup) {
                    Move-Item -LiteralPath $backup -Destination (Join-Path $GuiRoot $name) -Force
                }
            }
        } else {
            foreach ($relative in $placed) {
                $destination = Join-Path $GuiRoot $relative
                if (Test-Path -LiteralPath $destination) { Remove-Item -LiteralPath $destination -Recurse -Force }
            }
        }
        throw
    }
}

$stage = $null
try {
    Write-Host 'FACT-GUI downloader' -ForegroundColor Green
    $selection = Read-Selection
    if ($selection -eq 'Cancel') {
        Write-Host 'Cancelled. No files were changed.'
        exit 0
    }

    $demoTargets = @(switch ($selection) {
        'BasicDemo' { $BasicDemoFiles }
        'AllDemo' { $AllDemoFiles }
        default { @() }
    })
    $requiredGiB = switch ($selection) {
        'GuiOnly' { 5.2 }
        'BasicDemo' { 7.1 }
        'AllDemo' { 11.8 }
    }

    Assert-FreeSpace $GuiRoot $requiredGiB
    $stage = New-StageDirectory
    $uvExe = Get-UvExecutable $stage
    $guiEntries = Get-DriveFolderEntries $uvExe $GuiFolderUrl
    $guiStatus = Get-GuiStatus $guiEntries
    $demoStatus = if ($demoTargets.Count -gt 0) { Get-DemoStatus $demoTargets } else { $null }

    Write-Host "GUI files present: $($guiStatus.PresentCount) of $($guiStatus.Total)."
    if ($null -ne $demoStatus) {
        Write-Host "Selected Demo files present: $($demoStatus.Present.Count) of $($demoTargets.Count)."
    }

    $allPresent = $guiStatus.AllPresent -and ($null -eq $demoStatus -or $demoStatus.AllPresent)
    $somePresent = $guiStatus.PresentCount -gt 0 -or ($null -ne $demoStatus -and $demoStatus.Present.Count -gt 0)
    if ($allPresent) {
        $confirmed = Read-YesNo 'All selected files already exist. Re-download and replace them?'
    } elseif ($somePresent) {
        $confirmed = Read-YesNo 'Some selected files already exist. Download to staging and repair only missing files?'
    } else {
        $confirmed = Read-YesNo 'Download the selected FACT-GUI package now?'
    }
    if (-not $confirmed) {
        Write-Host 'Cancelled. No payload files were changed.'
        exit 0
    }

    $replaceGui = $allPresent -and $guiStatus.AllPresent
    $replaceDemo = $allPresent -and $null -ne $demoStatus -and $demoStatus.AllPresent
    if (-not $guiStatus.AllPresent -or $replaceGui) {
        $guiPayload = Download-Gui $uvExe $stage $guiEntries
        Install-Gui $guiPayload $guiEntries $guiStatus $replaceGui $stage
    }
    if ($null -ne $demoStatus -and (-not $demoStatus.AllPresent -or $replaceDemo)) {
        $demoPayload = Download-DemoFiles $uvExe $stage $demoTargets
        Install-DemoFiles $demoPayload $demoTargets $demoStatus $replaceDemo $stage
    }
    Write-Host 'FACT-GUI download completed successfully.' -ForegroundColor Green
} catch {
    Write-Host "Download failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($null -ne $stage -and (Test-Path -LiteralPath $stage)) {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}
