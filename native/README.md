# Visual Analysis 原生 Windows 迁移

当前原生入口是 Windows 11 x64 的 `VisualAnalysis.exe`。Win32 提供窗口，OpenCV C++ 负责图像和几何处理，ONNX Runtime C++ 直接加载本地模型。视频人像还原通过 FFmpeg 的 C API 解码、导出 FFV1/MKV 并复制源音轨。

## 现有原生功能

- 图片人像还原使用 SCRFD 关键点、PCFlow 和 GFPGAN v1.4。分数综合原图一致性与有原图支撑的边缘细节，降低碎纹较多的候选排名。同一张照片的近似输出只保留一个。第二个候选须同时通过原图一致性、纹理和眼嘴局部差异筛选。
- 勾选「偏召回」会多试去屏纹输入。常规检测未找到人脸时，也会补做一次低阈值检测。导出前使用同样的候选筛选规则。
- 图片车牌还原需要先框选车牌区域，按多个方向和长度去拖影，保留视觉差异足够的候选。
- 人像视频使用 GFPGAN v1.4 和对齐后的残差平滑，导出时保留帧时间和音轨。

「偏召回」适用于图像还原。人脸检索、以图搜索、语义搜索、人车检测、车型核验、人员库、标注库和视频车牌仍需完成原生迁移。模型使用 CPU 推理，Windows GPU 后端待验收。

现有 GLINT 人脸特征、RT-DETRv2 检测和 Chinese-CLIP 图文编码也已有 ONNX 权重。迁移时要复用现有的张量预处理、分数规则、中文词表和 SQLite 数据格式，并用相同输入对比输出。车型联网核验仍需独立的联网进程和授权数据源。

## 构建与案例检查

用 `scripts/build_native.ps1`，指定 MSVC x64 工具链目录、Windows SDK、三个本机 SDK 以及产物目录。此次构建使用 ONNX Runtime 1.24.4 的 Windows x64 DLL 与 C++ 头文件、OpenCV 4.13.0 Windows x64 版以及 FFmpeg 8.1 LGPL shared SDK。[ONNX Runtime C++](https://onnxruntime.ai/docs/get-started/with-cpp.html) · [OpenCV Windows 发布包](https://github.com/opencv/opencv/releases/tag/4.13.0) · [FFmpeg Windows 构建](https://ffmpeg.org/download.html)

GUI 默认在 EXE 同级的 `models/` 读取权重。向构建脚本传入 `-ModelDirectory`，可在校验 SHA-256 后复制 SCRFD、PCFlow 及 GFPGAN 权重。PCFlow 与 GFPGAN 使用现有已校验的产物，无需安装原项目 Python SDK。[PCFlow](https://github.com/aiimaginglab/PCFlow) · [GFPGAN](https://github.com/TencentARC/GFPGAN)

```powershell
$exe = 'D:\DevTools\visual-analysis-dev\native\build\VisualAnalysis.exe'
$argsRun = @('--check-restoration-cases', 'D:\visual-analysis\cases\inputs',
    '--case-output', 'D:\visual-analysis\cases\outputs',
    '--report', 'D:\visual-analysis\cases\outputs\native-report.json',
    '--models', 'D:\DevTools\visual-analysis-dev\models', '--prefer-recall')
$run = Start-Process -FilePath $exe -ArgumentList $argsRun -WindowStyle Hidden -Wait -PassThru
if ($run.ExitCode) { throw "Native case check failed." }
```

省略 `--prefer-recall` 可测试默认模式。报告保存每个输入的 SHA-256、真实 PNG 路径、候选分数和置信等级。本机案例及结果见 [案例库说明](../cases/README.md)。
