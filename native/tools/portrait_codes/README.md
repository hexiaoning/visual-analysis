# 人像候选的本地码本解码

`PortraitCodes.exe` 在 Windows x64 上运行。它允许同一张对齐糊图的两个曝光端点分别提供码本线索，同时由原图提供解码跳接。推理和码本选择均在本机完成。端点由 [`PortraitMoments.exe`](../portrait_moments/README.md) 从同一张糊图生成。配对结果及剩余重影见[评测记录](../../../docs/portrait-native-paired-evaluation.md)。

## 构建

使用 Windows SDK、MSVC x64、ONNX Runtime DirectML NuGet 包、DirectML NuGet 包和 OpenCV x64。构建脚本复制对应的运行库。

```powershell
.\build.ps1 -OutputDirectory D:\Data\Build\portrait-codes `
    -OnnxRuntimeDirectory $env:ONNXRUNTIME_DIR -DirectMlDirectory $env:DIRECTML_DIR `
    -OpenCvDirectory $env:OPENCV_BUILD -MsvcDirectory $env:MSVC_TOOLS `
    -WindowsSdkDirectory $env:WINDOWS_SDK
```

## 运行

`run.ps1` 从一张糊图生成端点、解码候选并筛选。模型目录需包含 `jin-240.onnx`、`signed-512.onnx`、`encode.onnx`、`decode.onnx`、`hr-nosa-wflw-geometry.onnx` 及 ONNX 引用的外部权重。按 `PortraitMoments.exe` 的说明配置 OpenVINO 运行库。脚本全部完成后发布输出目录，并打印选择清单路径。

```powershell
.\run.ps1 -InputPng .\blurred.png -OutputDirectory .\results -ModelDirectory .\models `
    -MomentsExecutable D:\Data\Build\portrait-moments\PortraitMoments.exe `
    -CodesExecutable D:\Data\Build\portrait-codes\PortraitCodes.exe `
    -CandidatesExecutable D:\Data\Build\portrait-candidates\PortraitCandidates.exe
```

用 `-Device CPU` 可指定 CPU。也可以单独复核编码、混分和解码三个步骤。

添加 `-RefineClarity` 可在初次选择后试验同组清晰候选替换。原展示的追加项保持各自位置。此时入口返回最终选择回执，`refinement-inputs.log` 按输入顺序记录回执中的候选索引所对应的文件。

添加 `-CloseSeparation 0.025` 可在通常追加结束后，对剩余名额尝试保留带通距离较近、双向五官检查通过的备选。参数取值大于 0 且小于通常使用的 0.055 半径。初次选择目录的回执记录实际使用的半径和检测结果。

添加 `-TransferGuidedDetail` 可从同一张糊图再生成至多四张备选。脚本把直接解码的人像分别移到两个曝光端点的轮廓，并各保留一个局部对比度对照供筛选。需一并使用支持 `--transfer-guided-detail` 的 `PortraitCandidates.exe`。两个端点的回执路径记录在 `guided-detail-paths.log`，筛选仍按回放、清晰度和五官差异决定展示哪些图。

编码输入为 512×512 图像，经程序转换成 `[1,3,512,512]` 的 RGB FP32，值域为 `[-1,1]`。编码模型输出 `[1,256,1024]` 的未归一化码本评分，`encode` 依次写出评分的 FP32 原始字节。`blend` 从原图和端点评分逐格选择编码，写出 `[1,16,16]` 的小端 int64，取值范围为 0–1023。`decode` 接受原糊图和所选编码，输出一张 PNG。原图和端点均采用同一裁切坐标。编码器和解码器的 ONNX 与它们引用的外部权重文件需已在本机。

```powershell
$exe = 'D:\Data\Build\portrait-codes\PortraitCodes.exe'
& $exe encode dml .\encode.onnx .\blurred.png .\raw.f32 .\first-moment.png .\first.f32 .\last-moment.png .\last.f32
& $exe blend .\raw.f32 .\first.f32 .\first.bin .\raw.f32 .\last.f32 .\last.bin
& $exe decode dml .\decode.onnx .\blurred.png .\first.bin .\candidate-first.png .\blurred.png .\last.bin .\candidate-last.png
```

运行命令前准备独立输出路径。每张输出写完后单独发布，已有文件受到保护。将 `dml` 改为 `cpu` 可用 CPU 复验。两个推理命令输出 ONNX Runtime 的执行分析文件路径。后续可用 [`PortraitCandidates.exe`](../portrait_candidates/README.md#在原结果后追加有观测支持的候选) 比较回放、清晰度和五官差异。

本轮验证模型位于 `D:/Data/Build/visual-analysis-dev/portrait-exposure-latent-01`。模型与测试照片保存在试验目录。
