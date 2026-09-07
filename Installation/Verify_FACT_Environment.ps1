[CmdletBinding()]
param(
    [string]$Prefix,

    [string]$ReferencePath,

    [Parameter(Mandatory = $true)]
    [string]$SourcePrefix,

    [switch]$CreateReference,

    [switch]$RequireGpu
)

$ErrorActionPreference = 'Stop'

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

function Get-CondaExecutable {
    if ($env:FACT_CONDA_EXE -and (Test-Path -LiteralPath $env:FACT_CONDA_EXE)) {
        return $env:FACT_CONDA_EXE
    }
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        $command = Get-Command conda -ErrorAction SilentlyContinue
    }
    if (-not $command) {
        throw 'conda.exe was not found. Set FACT_CONDA_EXE or add Conda to PATH.'
    }
    return $command.Source
}

function Get-PackageRecords {
    param([Parameter(Mandatory = $true)][string]$EnvironmentPrefix)

    $conda = Get-CondaExecutable
    $raw = & $conda list --json --prefix $EnvironmentPrefix 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "conda list failed for $EnvironmentPrefix`n$($raw -join "`n")"
    }
    $jsonText = @($raw | ForEach-Object { [string]$_ }) -join "`n"
    $records = ConvertFrom-Json -InputObject $jsonText
    return @($records | ForEach-Object {
        [PSCustomObject]@{
            name = [string]$_.name
            version = [string]$_.version
            build_string = [string]$_.build_string
            channel = [string]$_.channel
        }
    } | Sort-Object name, version, build_string, channel)
}

function Get-CondaMetaNames {
    param([Parameter(Mandatory = $true)][string]$EnvironmentPrefix)

    $meta = Join-Path $EnvironmentPrefix 'conda-meta'
    if (-not (Test-Path -LiteralPath $meta)) {
        throw "Conda metadata directory not found: $meta"
    }
    return @(
        Get-ChildItem -LiteralPath $meta -File -Filter '*.json' |
            Select-Object -ExpandProperty Name |
            Sort-Object
    )
}

function Get-RuntimeSnapshot {
    param([Parameter(Mandatory = $true)][string]$EnvironmentPrefix)

    $python = Join-Path $EnvironmentPrefix 'python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Python executable not found: $python"
    }

    $code = @'
import json
import sys

import numpy
import torch
import torchvision
import h5py
import scipy
import skimage
import sklearn
import monai
import SimpleITK
import transformers
import ScanImageTiffReader

result = {
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_version": torch.version.cuda,
    "h5py": h5py.__version__,
    "hdf5": h5py.version.hdf5_version,
    "scipy": scipy.__version__,
    "skimage": skimage.__version__,
    "sklearn": sklearn.__version__,
    "monai": monai.__version__,
    "SimpleITK": SimpleITK.Version_VersionString(),
    "transformers": transformers.__version__,
    "scanimage_import": True,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_sum": float(torch.ones(4, device="cuda").sum().item()) if torch.cuda.is_available() else None,
}
print(json.dumps(result, sort_keys=True))
'@

    $probePath = Join-Path ([IO.Path]::GetTempPath()) ("fact_runtime_probe_{0}.py" -f [Guid]::NewGuid().ToString('N'))
    Set-Content -LiteralPath $probePath -Value $code -Encoding UTF8
    try {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $raw = @(& $python $probePath 2>&1)
        $pythonExitCode = $LASTEXITCODE
        $ErrorActionPreference = $oldErrorAction
        if ($pythonExitCode -ne 0) {
            throw "Runtime import probe failed for $EnvironmentPrefix`n$($raw -join "`n")"
        }
        $jsonLine = @($raw | Where-Object { $_ -match '^{' } | Select-Object -Last 1)
        if ($jsonLine.Count -ne 1) {
            throw "Runtime probe did not return one JSON record for $EnvironmentPrefix"
        }
        return ($jsonLine[0] | ConvertFrom-Json)
    }
    finally {
        if (Test-Path -LiteralPath $probePath) {
            Remove-Item -LiteralPath $probePath -Force
        }
    }
}

function Get-PipCheckLines {
    param([Parameter(Mandatory = $true)][string]$EnvironmentPrefix)

    $python = Join-Path $EnvironmentPrefix 'python.exe'
    $raw = & $python -m pip check 2>&1
    return [PSCustomObject]@{
        ExitCode = [int]$LASTEXITCODE
        Lines = @($raw | ForEach-Object { [string]$_ })
    }
}

function Get-CriticalFileSnapshots {
    param([Parameter(Mandatory = $true)][string]$EnvironmentPrefix)

    $relativePaths = @(
        'Library\bin\hdf5.dll',
        'Lib\site-packages\h5py\_errors.cp311-win_amd64.pyd',
        'Lib\site-packages\scipy\_lib\_ccallback_c.cp311-win_amd64.pyd',
        'Lib\site-packages\skimage\_shared\geometry.cp311-win_amd64.pyd',
        'Lib\site-packages\torch\_C.cp311-win_amd64.pyd',
        'Lib\site-packages\torchvision\_C.pyd',
        'Lib\site-packages\ScanImageTiffReader\external\ScanImageTiffReader-1.4.1-win64\lib\ScanImageTiffReaderAPI.dll'
    )

    return @($relativePaths | ForEach-Object {
        $relative = $_
        $path = Join-Path $EnvironmentPrefix $relative
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Critical file not found: $path"
        }
        [PSCustomObject]@{
            path = $relative
            sha256 = Get-Sha256 -Path $path
        }
    })
}

function New-ReferenceManifest {
    param(
        [Parameter(Mandatory = $true)][string]$SourceEnvironmentPrefix,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    $pip = Get-PipCheckLines -EnvironmentPrefix $SourceEnvironmentPrefix
    if ($pip.Lines.Count -ne 1 -or $pip.Lines[0] -notmatch '^imagecodecs .+ has requirement numpy>=2\.0, but you have numpy 1\.26\.4\.$') {
        throw "The source pip check state is not the expected single inherited warning: $($pip.Lines -join '; ')"
    }

    $runtime = Get-RuntimeSnapshot -EnvironmentPrefix $SourceEnvironmentPrefix
    $manifest = [ordered]@{
        schema = 1
        environmentName = 'fact_py311'
        sourceEnvironmentName = 'fact_py311'
        visiblePackages = @(Get-PackageRecords -EnvironmentPrefix $SourceEnvironmentPrefix)
        condaMetaFiles = @(Get-CondaMetaNames -EnvironmentPrefix $SourceEnvironmentPrefix)
        criticalFiles = @(Get-CriticalFileSnapshots -EnvironmentPrefix $SourceEnvironmentPrefix)
        runtime = [ordered]@{
            python = $runtime.python
            numpy = $runtime.numpy
            torch = $runtime.torch
            torchvision = $runtime.torchvision
            cuda_version = $runtime.cuda_version
            h5py = $runtime.h5py
            hdf5 = $runtime.hdf5
            scipy = $runtime.scipy
            skimage = $runtime.skimage
            sklearn = $runtime.sklearn
            monai = $runtime.monai
            SimpleITK = $runtime.SimpleITK
            transformers = $runtime.transformers
            scanimage_import = [bool]$runtime.scanimage_import
        }
        pipCheckExitCode = $pip.ExitCode
        pipCheckLines = @($pip.Lines)
    }

    $parent = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
    Write-Host "Reference manifest written: $OutputPath"
}

function Compare-StringArrays {
    param(
        [Parameter(Mandatory = $true)][object[]]$Expected,
        [Parameter(Mandatory = $true)][object[]]$Actual
    )

    $expectedStrings = @($Expected | ForEach-Object { [string]$_ } | Sort-Object)
    $actualStrings = @($Actual | ForEach-Object { [string]$_ } | Sort-Object)
    return @(
        Compare-Object -ReferenceObject $expectedStrings -DifferenceObject $actualStrings
    )
}

function Compare-PackageRecords {
    param(
        [Parameter(Mandatory = $true)][object[]]$Expected,
        [Parameter(Mandatory = $true)][object[]]$Actual
    )

    $expectedKeys = @($Expected | ForEach-Object { '{0}|{1}|{2}|{3}' -f $_.name, $_.version, $_.build_string, $_.channel } | Sort-Object)
    $actualKeys = @($Actual | ForEach-Object { '{0}|{1}|{2}|{3}' -f $_.name, $_.version, $_.build_string, $_.channel } | Sort-Object)
    return @(
        Compare-Object -ReferenceObject $expectedKeys -DifferenceObject $actualKeys
    )
}

if ($CreateReference) {
    if (-not $ReferencePath) {
        throw '-ReferencePath is required with -CreateReference.'
    }
    New-ReferenceManifest -SourceEnvironmentPrefix $SourcePrefix -OutputPath $ReferencePath
    exit 0
}

if (-not $ReferencePath) {
    throw '-ReferencePath is required when verifying an environment.'
}
if (-not $Prefix) {
    throw '-Prefix is required when verifying an environment.'
}
if (-not (Test-Path -LiteralPath $ReferencePath)) {
    throw "Reference manifest not found: $ReferencePath"
}

$reference = Get-Content -LiteralPath $ReferencePath -Raw | ConvertFrom-Json
$actualPackages = Get-PackageRecords -EnvironmentPrefix $Prefix
$actualMeta = Get-CondaMetaNames -EnvironmentPrefix $Prefix
$actualCritical = Get-CriticalFileSnapshots -EnvironmentPrefix $Prefix
$actualRuntime = Get-RuntimeSnapshot -EnvironmentPrefix $Prefix
$actualPip = Get-PipCheckLines -EnvironmentPrefix $Prefix

$packageDiff = Compare-PackageRecords -Expected @($reference.visiblePackages) -Actual $actualPackages
$metaDiff = Compare-StringArrays -Expected @($reference.condaMetaFiles) -Actual $actualMeta

$criticalDiff = @()
foreach ($expectedFile in @($reference.criticalFiles)) {
    $actualFile = @($actualCritical | Where-Object { $_.path -eq $expectedFile.path }) | Select-Object -First 1
    if (-not $actualFile -or $actualFile.sha256 -ne $expectedFile.sha256) {
        $criticalDiff += [PSCustomObject]@{
            path = $expectedFile.path
            expected = $expectedFile.sha256
            actual = if ($actualFile) { $actualFile.sha256 } else { '<missing>' }
        }
    }
}

$runtimeKeys = @(
    'python', 'numpy', 'torch', 'torchvision', 'cuda_version', 'h5py', 'hdf5',
    'scipy', 'skimage', 'sklearn', 'monai', 'SimpleITK', 'transformers', 'scanimage_import'
)
$runtimeDiff = @()
foreach ($key in $runtimeKeys) {
    $expectedValue = [string]$reference.runtime.$key
    $actualValue = [string]$actualRuntime.$key
    if ($expectedValue -ne $actualValue) {
        $runtimeDiff += [PSCustomObject]@{
            field = $key
            expected = $expectedValue
            actual = $actualValue
        }
    }
}

$expectedPipLines = @($reference.pipCheckLines | ForEach-Object { [string]$_ })
$actualPipLines = @($actualPip.Lines | ForEach-Object { [string]$_ })
$pipDiff = Compare-StringArrays -Expected $expectedPipLines -Actual $actualPipLines

Write-Host ("VISIBLE_RECORDS expected={0} actual={1} differences={2}" -f @($reference.visiblePackages).Count, $actualPackages.Count, $packageDiff.Count)
Write-Host ("CONDA_META expected={0} actual={1} differences={2}" -f @($reference.condaMetaFiles).Count, $actualMeta.Count, $metaDiff.Count)
Write-Host ("CRITICAL_FILE_DIFFERENCES={0}" -f $criticalDiff.Count)
Write-Host ("RUNTIME_DIFFERENCES={0}" -f $runtimeDiff.Count)
Write-Host ("PIP_CHECK expected_exit={0} actual_exit={1} differences={2}" -f $reference.pipCheckExitCode, $actualPip.ExitCode, $pipDiff.Count)

if ($packageDiff.Count) { $packageDiff | Format-Table -AutoSize }
if ($metaDiff.Count) { $metaDiff | Format-Table -AutoSize }
if ($criticalDiff.Count) { $criticalDiff | Format-Table -AutoSize }
if ($runtimeDiff.Count) { $runtimeDiff | Format-Table -AutoSize }
if ($pipDiff.Count) { $pipDiff | Format-Table -AutoSize }

if ($RequireGpu) {
    if (-not [bool]$actualRuntime.cuda_available -or [double]$actualRuntime.gpu_sum -ne 4.0) {
        throw 'GPU smoke test failed or CUDA is unavailable.'
    }
    Write-Host ("GPU={0}; CUDA={1}; tensor_sum={2}" -f $actualRuntime.gpu_name, $actualRuntime.cuda_version, $actualRuntime.gpu_sum)
}

if ($packageDiff.Count -or $metaDiff.Count -or $criticalDiff.Count -or $runtimeDiff.Count -or $pipDiff.Count) {
    exit 2
}

Write-Host 'FACT fact_py311 environment verification passed.'
exit 0
