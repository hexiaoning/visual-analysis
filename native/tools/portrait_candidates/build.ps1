[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$OnnxRuntimeDirectory,
    [Parameter(Mandatory)][string]$OpenCvDirectory,
    [Parameter(Mandatory)][string]$JsonIncludeDirectory,
    [Parameter(Mandatory)][string]$MsvcDirectory,
    [Parameter(Mandatory)][string]$WindowsSdkDirectory,
    [string]$WindowsSdkVersion = "10.0.26100.0",
    [string]$DirectMlDirectory,
    [string]$CrtDirectory
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Compiler = Join-Path $MsvcDirectory "bin\Hostx64\x64\cl.exe"
$SdkHeaders = Join-Path $WindowsSdkDirectory "Include\$WindowsSdkVersion"
$SdkLib = Join-Path $WindowsSdkDirectory "Lib\$WindowsSdkVersion"
if (-not (Test-Path -LiteralPath $SdkLib -PathType Container)) {
    $SdkLib = $WindowsSdkDirectory
}
$OrtNative = Join-Path $OnnxRuntimeDirectory "runtimes\win-x64\native"
$OpenCvNative = Join-Path $OpenCvDirectory "x64\vc16"
$World = @(Get-ChildItem -LiteralPath (Join-Path $OpenCvNative "lib") -Filter "opencv_world*.lib" -File |
    Where-Object BaseName -NotLike "*d")
if ($World.Count -ne 1) {
    throw "Expected one x64 OpenCV release world library."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$Output = (Resolve-Path -LiteralPath $OutputDirectory).ProviderPath
$Objects = Join-Path $Output "obj"
New-Item -ItemType Directory -Path $Objects -Force | Out-Null
$Arguments = @(
    "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", "/fp:precise", "/W4", "/utf-8",
    "/DUNICODE", "/D_UNICODE", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
    "/external:W0", "/external:I$OnnxRuntimeDirectory\build\native\include",
    "/external:I$OpenCvDirectory\include", "/external:I$JsonIncludeDirectory",
    "/I$MsvcDirectory\include", "/I$SdkHeaders\ucrt", "/I$SdkHeaders\um", "/I$SdkHeaders\shared",
    "/Fo$Objects\", "/Fe$Output\PortraitCandidates.exe",
    (Join-Path $PSScriptRoot "main.cpp"), (Join-Path $PSScriptRoot "portrait_selection.cpp"),
    (Join-Path $PSScriptRoot "duplicate_consensus.cpp"), (Join-Path $PSScriptRoot "texture_guard.cpp"),
    (Join-Path $PSScriptRoot "exposure_normalizer.cpp"), (Join-Path $PSScriptRoot "geometry_calibration.cpp"),
    "/link", "/subsystem:console", "/incremental:no", "/libpath:$MsvcDirectory\lib\x64",
    "/libpath:$SdkLib\ucrt\x64", "/libpath:$SdkLib\um\x64",
    (Join-Path $OrtNative "onnxruntime.lib"), $World[0].FullName
)
& $Compiler @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "PortraitCandidates build failed with exit code $LASTEXITCODE."
}

$Libraries = @(
    (Join-Path $OrtNative "onnxruntime.dll"),
    (Join-Path $OpenCvNative ("bin\" + $World[0].BaseName + ".dll"))
)
$SharedProviders = Join-Path $OrtNative "onnxruntime_providers_shared.dll"
if (Test-Path -LiteralPath $SharedProviders -PathType Leaf) {
    $Libraries += $SharedProviders
}
if ($DirectMlDirectory) {
    $Libraries += Join-Path $DirectMlDirectory "bin\x64-win\DirectML.dll"
}
if ($CrtDirectory) {
    foreach ($Name in "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll") {
        $Libraries += Join-Path $CrtDirectory $Name
    }
}
Copy-Item -LiteralPath $Libraries -Destination $Output -Force
