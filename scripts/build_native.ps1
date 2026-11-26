[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$OnnxRuntimeDirectory,
    [Parameter(Mandatory)][string]$DirectMlDirectory,
    [Parameter(Mandatory)][string]$OpenVinoDirectory,
    [Parameter(Mandatory)][string]$OpenCvDirectory,
    [Parameter(Mandatory)][string]$FFmpegDirectory,
    [Parameter(Mandatory)][string]$JsonIncludeDirectory,
    [Parameter(Mandatory)][string]$MsvcDirectory,
    [Parameter(Mandatory)][string]$WindowsSdkDirectory,
    [string]$ModelDirectory,
    [string]$WindowsSdkVersion = "10.0.26100.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).ProviderPath
$Sources = Join-Path $Project "native\src"
$Cl = Join-Path $MsvcDirectory "bin\Hostx64\x64\cl.exe"
$OrtHeaders = Join-Path $OnnxRuntimeDirectory "build\native\include"
$OrtNative = Join-Path $OnnxRuntimeDirectory "runtimes\win-x64\native"
$JsonHeader = Join-Path $JsonIncludeDirectory "nlohmann\json.hpp"
if ((Get-FileHash -LiteralPath $JsonHeader -Algorithm SHA256).Hash -ine "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63") {
    throw "Use the verified nlohmann/json 3.12.0 single header."
}
$OpenCvHeaders = Join-Path $OpenCvDirectory "include"
$OpenCvNative = Join-Path $OpenCvDirectory "x64\vc16"
$MediaLibraries = @("avformat", "avcodec", "avutil", "swscale")
$SdkHeaders = Join-Path $WindowsSdkDirectory "Include\$WindowsSdkVersion"
$SdkLink = Join-Path $WindowsSdkDirectory "Lib\$WindowsSdkVersion"
if (-not (Test-Path -LiteralPath $SdkLink -PathType Container)) {
    $SdkLink = $WindowsSdkDirectory
}
$OpenCvLib = @(Get-ChildItem -LiteralPath (Join-Path $OpenCvNative "lib") -Filter "opencv_world*.lib" -File |
    Where-Object BaseName -NotLike "*d")
if ($OpenCvLib.Count -ne 1) {
    throw "Expected one OpenCV x64 release world library."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$Output = (Resolve-Path -LiteralPath $OutputDirectory).ProviderPath
$Objects = Join-Path $Output "obj"
New-Item -ItemType Directory -Path $Objects -Force | Out-Null
$Executable = Join-Path $Output "VisualAnalysis.exe"
$OrtLibrary = Join-Path $OrtNative "onnxruntime.lib"
$Arguments = @(
    "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", "/W4", "/utf-8", "/MP2",
    "/DUNICODE", "/D_UNICODE", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
    "/external:W0", "/external:I$OrtHeaders", "/external:I$OpenCvHeaders",
    "/external:I$FFmpegDirectory\include",
    "/external:I$JsonIncludeDirectory",
    "/external:I$OpenVinoDirectory\include",
    "/I$MsvcDirectory\include", "/I$SdkHeaders\ucrt", "/I$SdkHeaders\shared", "/I$SdkHeaders\um",
    "/Fo$Objects\", "/Fe$Executable", (Join-Path $Sources "app.cpp"),
    (Join-Path $Sources "local_images.cpp"), (Join-Path $Sources "portrait.cpp"), (Join-Path $Sources "plate.cpp"),
    (Join-Path $Sources "portrait_video.cpp"),
    (Join-Path $Sources "storage.cpp"), (Join-Path $Sources "vision_models.cpp"),
    (Join-Path $Sources "people.cpp"),
    (Join-Path $Sources "vehicle.cpp"), (Join-Path $Sources "operations.cpp"), (Join-Path $Sources "operations_window.cpp"),
    (Join-Path $Sources "media_operations.cpp"), (Join-Path $Sources "plate_video.cpp"),
    (Join-Path $Sources "preview_native.cpp"), (Join-Path $Sources "workspace_theme.cpp"), (Join-Path $Sources "inference_backend.cpp"),
    (Join-Path $Sources "openvino_model.cpp"),
    "/link", "/subsystem:windows", "/incremental:no", "/libpath:$MsvcDirectory\lib\x64",
    "/libpath:$SdkLink\ucrt\x64", "/libpath:$SdkLink\um\x64",
    $OrtLibrary, $OpenCvLib[0].FullName, (Join-Path $OpenVinoDirectory "libs\openvino.lib"), "/libpath:$FFmpegDirectory\lib",
    "user32.lib", "gdi32.lib", "comdlg32.lib", "shell32.lib", "bcrypt.lib", "winsqlite3.lib", "normaliz.lib",
    "crypt32.lib", "ole32.lib", "uuid.lib", "dwmapi.lib", "dxgi.lib", "d3d12.lib"
)
$Arguments += @($MediaLibraries | ForEach-Object { "$_.lib" })
& $Cl @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "The native Visual Analysis build failed with exit code $LASTEXITCODE."
}
$Common = $Arguments[0..($Arguments.IndexOf("/Fo$Objects\") - 1)]
$WorkerOutput = Join-Path $Output "VehicleCatalogueWorker.exe"
& $Cl @Common "/Fo$Objects\" "/Fe$WorkerOutput" (Join-Path $Sources "catalogue_worker.cpp") /link /subsystem:console /incremental:no `
    "/libpath:$MsvcDirectory\lib\x64" "/libpath:$SdkLink\ucrt\x64" "/libpath:$SdkLink\um\x64" winhttp.lib crypt32.lib
if ($LASTEXITCODE -ne 0) {
    throw "The native vehicle catalogue worker build failed with exit code $LASTEXITCODE."
}
$Libraries = @(
    (Join-Path $OrtNative "onnxruntime.dll"),
    (Join-Path $OrtNative "onnxruntime_providers_shared.dll"),
    (Join-Path $DirectMlDirectory "bin\x64-win\DirectML.dll"),
    (Join-Path $OpenCvNative ("bin\" + $OpenCvLib[0].BaseName + ".dll"))
)
Copy-Item -LiteralPath $Libraries -Destination $Output -Force
$IntelLibraries = @("openvino.dll", "openvino_onnx_frontend.dll", "openvino_intel_gpu_plugin.dll",
    "tbb12.dll", "tbbmalloc.dll", "tbbbind_2_5.dll")
foreach ($File in $IntelLibraries) {
    Copy-Item -LiteralPath (Join-Path $OpenVinoDirectory "libs\$File") -Destination $Output -Force
}
foreach ($Name in $MediaLibraries + "swresample") {
    $Library = @(Get-ChildItem -LiteralPath (Join-Path $FFmpegDirectory "bin") -Filter "$Name-*.dll" -File)
    if ($Library.Count -ne 1) {
        throw "Expected a single x64 FFmpeg shared library for $Name."
    }
    Copy-Item -LiteralPath $Library[0].FullName -Destination $Output -Force
}
if ($ModelDirectory) {
    $Models = @{
        "scrfd_10g_bnkps.onnx" = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
        "pcflow-bfr-fp32.onnx" = "74ea9dcf3ded27a4dbd4a3c71dfa994fe7ee03365c275f78a9023a6c2a68e91a"
        "gfpgan-v1.4-fp32.onnx" = "a05c0040e64b383861810701f183b7455718313fb3a7465446cdc67b1fa7807d"
        "glintr100.onnx" = "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"
        "rtdetr_v2_r50vd.onnx" = "d6d5ce9f633189539bc892c70b6a60120a32a9799798811daeb8456f9b9e2c15"
        "chinese_clip_vision.onnx" = "980020bf226e528a202de26c6a6186df125b74c612b09cfcd797a082e046b628"
        "chinese_clip_text.onnx" = "5ddc2d8971b09acda8063048003d18cca9b587f089a1c0a4df7d35624bd0fcea"
        "chinese_clip_vocab.txt" = "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c"
    }
    $ModelOutput = Join-Path $Output "models"
    New-Item -ItemType Directory -Path $ModelOutput -Force | Out-Null
    foreach ($Name in $Models.Keys) {
        $Source = Join-Path $ModelDirectory $Name
        if ((Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash -ine $Models[$Name]) {
            throw "Native model SHA-256 verification failed for $Name."
        }
        Copy-Item -LiteralPath $Source -Destination $ModelOutput -Force
    }
}
Write-Output $Executable
