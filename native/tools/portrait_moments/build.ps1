[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$OpenVinoDirectory,
    [Parameter(Mandatory)][string]$OpenCvDirectory,
    [Parameter(Mandatory)][string]$MsvcDirectory,
    [Parameter(Mandatory)][string]$WindowsSdkDirectory,
    [string]$WindowsSdkVersion = "10.0.26100.0"
)

$ErrorActionPreference = "Stop"
$Headers = Join-Path $WindowsSdkDirectory "Include\$WindowsSdkVersion"
$Libraries = Join-Path $WindowsSdkDirectory "Lib\$WindowsSdkVersion"
if (!(Test-Path -LiteralPath $Libraries -PathType Container)) {
    $Libraries = $WindowsSdkDirectory
}
$OpenCvLibraries = Join-Path $OpenCvDirectory "x64\vc16\lib"
$World = @(Get-ChildItem -LiteralPath $OpenCvLibraries -Filter "opencv_world*.lib" -File |
    Where-Object BaseName -NotLike "*d")
if ($World.Count -ne 1) {
    throw "Expected one OpenCV x64 release world library."
}

$Folder = New-Item -ItemType Directory -Path $OutputDirectory -Force
$Exe = Join-Path $Folder.FullName "PortraitMoments.exe"
$Object = Join-Path $Folder.FullName "PortraitMoments.obj"
$Arguments = @(
    "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", "/W4", "/utf-8", "/DNOMINMAX",
    "/external:W0", "/external:I$OpenVinoDirectory\include", "/external:I$OpenCvDirectory\include",
    "/I$MsvcDirectory\include", "/I$Headers\ucrt", "/I$Headers\um", "/I$Headers\shared",
    "/Fe$Exe", "/Fo$Object", (Join-Path $PSScriptRoot "main.cpp"),
    "/link", "/subsystem:console", "/incremental:no", "/libpath:$MsvcDirectory\lib\x64",
    "/libpath:$Libraries\ucrt\x64", "/libpath:$Libraries\um\x64", $World[0].FullName,
    (Join-Path $OpenVinoDirectory "libs\openvino.lib"),
    (Join-Path $OpenVinoDirectory "libs\openvino_onnx_frontend.lib")
)
& (Join-Path $MsvcDirectory "bin\Hostx64\x64\cl.exe") @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "PortraitMoments build failed with exit code $LASTEXITCODE."
}
