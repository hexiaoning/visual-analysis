[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$OpenCvDirectory,
    [Parameter(Mandatory)][string]$JsonIncludeDirectory,
    [Parameter(Mandatory)][string]$MsvcDirectory,
    [Parameter(Mandatory)][string]$WindowsSdkDirectory,
    [string]$WindowsSdkVersion = "10.0.26100.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$compiler = Join-Path $MsvcDirectory "bin/Hostx64/x64/cl.exe"
$headers = Join-Path $WindowsSdkDirectory "Include/$WindowsSdkVersion"
$sdkLibraries = Join-Path $WindowsSdkDirectory "Lib/$WindowsSdkVersion"
if (-not (Test-Path -LiteralPath $sdkLibraries -PathType Container)) {
    $sdkLibraries = $WindowsSdkDirectory
}
$opencvNative = Join-Path $OpenCvDirectory "x64/vc16"
$world = @(Get-ChildItem -LiteralPath (Join-Path $opencvNative "lib") -Filter "opencv_world*.lib" -File |
    Where-Object BaseName -NotLike "*d")
if ($world.Count -ne 1) {
    throw "Expected one x64 OpenCV release world library."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$destination = (Resolve-Path -LiteralPath $OutputDirectory).ProviderPath
$obj = Join-Path $destination "obj"
New-Item -ItemType Directory -Path $obj -Force | Out-Null
$flags = @(
    "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", "/W4", "/fp:precise", "/utf-8",
    "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN", "/external:W0",
    "/external:I$OpenCvDirectory/include", "/external:I$JsonIncludeDirectory",
    "/I$MsvcDirectory/include", "/I$headers/ucrt", "/I$headers/um", "/I$headers/shared",
    "/Fo$obj\", "/Fe$destination/PortraitScreen.exe",
    (Join-Path $PSScriptRoot "main.cpp"), (Join-Path $PSScriptRoot "screen_restore.cpp"),
    (Join-Path $PSScriptRoot "screen_replay.cpp"), (Join-Path $PSScriptRoot "capture_tone_guard.cpp"),
    (Join-Path $PSScriptRoot "grid_fusion.cpp"), (Join-Path $PSScriptRoot "align_model_views.cpp"),
    "/link", "/subsystem:console", "/incremental:no",
    "/libpath:$MsvcDirectory/lib/x64", "/libpath:$sdkLibraries/ucrt/x64", "/libpath:$sdkLibraries/um/x64",
    $world[0].FullName
)
& $compiler @flags
if ($LASTEXITCODE -ne 0) {
    throw "PortraitScreen build failed with exit code $LASTEXITCODE."
}
$dll = Join-Path $opencvNative ("bin/" + $world[0].BaseName + ".dll")
Copy-Item -LiteralPath $dll -Destination $destination -Force
