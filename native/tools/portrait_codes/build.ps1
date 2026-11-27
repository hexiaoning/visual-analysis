[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$OnnxRuntimeDirectory,
    [Parameter(Mandatory)][string]$DirectMlDirectory,
    [Parameter(Mandatory)][string]$OpenCvDirectory,
    [Parameter(Mandatory)][string]$MsvcDirectory,
    [Parameter(Mandatory)][string]$WindowsSdkDirectory,
    [string]$WindowsSdkVersion = "10.0.26100.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
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
$Arguments = @(
    "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", "/fp:precise", "/W4", "/utf-8",
    "/DUNICODE", "/D_UNICODE", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
    "/external:W0", "/external:I$OnnxRuntimeDirectory\build\native\include", "/external:I$OpenCvDirectory\include",
    "/I$MsvcDirectory\include", "/I$SdkHeaders\ucrt", "/I$SdkHeaders\um", "/I$SdkHeaders\shared",
    "/Fo$Output\PortraitCodes.obj", "/Fe$Output\PortraitCodes.exe", (Join-Path $PSScriptRoot "main.cpp"),
    "/link", "/subsystem:console", "/incremental:no", "/libpath:$MsvcDirectory\lib\x64",
    "/libpath:$SdkLib\ucrt\x64", "/libpath:$SdkLib\um\x64",
    (Join-Path $OrtNative "onnxruntime.lib"), $World[0].FullName
)
& (Join-Path $MsvcDirectory "bin\Hostx64\x64\cl.exe") @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "PortraitCodes build failed with exit code $LASTEXITCODE."
}

$Libraries = @(
    (Join-Path $OrtNative "onnxruntime.dll"),
    (Join-Path $DirectMlDirectory "bin\x64-win\DirectML.dll"),
    (Join-Path $OpenCvNative ("bin\" + $World[0].BaseName + ".dll"))
)
$SharedProviders = Join-Path $OrtNative "onnxruntime_providers_shared.dll"
if (Test-Path -LiteralPath $SharedProviders -PathType Leaf) {
    $Libraries += $SharedProviders
}
Copy-Item -LiteralPath $Libraries -Destination $Output -Force
