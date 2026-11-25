# Visual Analysis 原生 Windows 版

在 Windows 11 x64 上打开 `VisualAnalysis.exe`。顶部导航使用一级和二级标签。设置入口在右上角。Win32 提供窗口，OpenCV C++ 负责图像处理，ONNX Runtime C++ 直接加载本机模型，FFmpeg C API 读取并导出视频。SQLite 在本机保存人员库、标注和语义索引。

## 使用方法

- 「人像检索」支持照片和视频混选查库，同页可打开本机摄像头，处理中可继续追加素材。可用人员库、性别、地区码和年龄过滤。
- 「目标查找」包括目标检测、以图搜索和语义搜索。目标检测可保存带检测框的 MP4、目标截图及穿线或入区时的原始整帧。
- 「车辆分析 → 车牌识别」在本机定位图片或视频中的车牌。可指定起始帧后手动框选。结果显示识读状态、候选号码、原始裁图及适用的去拖影、透视校正或多帧融合图像。
- 「人像工具」提供双照比对。人像还原页面显示「开发中」，内部还原案例可继续用命令行验收。
- 「警情筛选」结合 110 导出的报警子类、细类、类型、类别和事件详情查找关注警情。否定、咨询、重复告警或冲突分类会标为「待复核」。可保存默认规则 JSON 并编辑分类词、叙事关键词、组合词及排除词。导出的 Excel 保留选中记录的原始字段、长文本编号和时间格式，以及筛选证据与状态。
- 规则的可选项 `review_all_of` 将证据不足的词组匹配送入复核。`context_targets` 指定用于识别局部否定或不确定表述的词。`outcome_exclude` 和全局 `outcome_review_keywords` 只检查反馈和处理结果。旧规则省略这些项时，对应的检查关闭。
- 「资料库」下设人员库和标注库。此前录入的记录继续使用 `%LOCALAPPDATA%\FaceMatching\data\faces.db`。
- 「车型识别」默认实时查询 Wikimedia 和 NHTSA vPIC。支持本地参考图库，车型查询通过独立进程进行。其他分析功能使用本机数据。[车型资料说明](../docs/vehicle-catalogues.md)

神经网络优先使用本机 DirectX 12 显卡。Intel 显卡上的 RT-DETR 人车检测和中文图文编码优先使用 OpenVINO，人像与人脸模型按推理后端配置运行。车牌定位和识读默认通过 ONNX Runtime DirectML 使用本机显卡，GPU 不可用时回退到 CPU。可用 `VISUAL_ANALYSIS_BACKEND=cpu` 运行兼容模式。人脸库的旧照片和旧特征保存在本机。既有特征的模型档与当前模型不同时，可在「人员库」按原始照片重建特征。

## 构建与案例检查

用 `scripts/build_native.ps1`，指定 MSVC x64、Windows SDK、ONNX Runtime DirectML 1.24.4、DirectML 1.15.4、OpenVINO 2026.4.1、OpenCV 4.13.0、FFmpeg 8.1 LGPL shared 版和 nlohmann/json 3.12.0 单头文件目录。依赖可预先放在构建机，脚本校验模型权重与 JSON 头文件。[ONNX Runtime C++](https://onnxruntime.ai/docs/get-started/with-cpp.html) · [OpenCV Windows 发布包](https://github.com/opencv/opencv/releases/tag/4.13.0) · [FFmpeg Windows 构建](https://ffmpeg.org/download.html)

GUI 默认在 EXE 同级的 `models/` 读取权重。向构建脚本传入 `-ModelDirectory`，可在校验 SHA-256 后复制 SCRFD、PCFlow 及 GFPGAN 权重。PCFlow 与 GFPGAN 使用现有已校验的产物，无需安装原项目 Python SDK。向构建脚本传入 `-PlateModelDirectory`，可固定校验并复制 HyperLPR3 的定位和识读权重。[PCFlow](https://github.com/aiimaginglab/PCFlow) · [GFPGAN](https://github.com/TencentARC/GFPGAN)

```powershell
$exe = 'D:\DevTools\visual-analysis-dev\native\build\VisualAnalysis.exe'
$argsRun = @('--check-restoration-cases', 'D:\visual-analysis\cases\portrait-restoration',
    '--case-output', 'D:\visual-analysis\cases\outputs',
    '--report', 'D:\visual-analysis\cases\outputs\native-report.json',
    '--models', 'D:\DevTools\visual-analysis-dev\models', '--prefer-recall')
$run = Start-Process -FilePath $exe -ArgumentList $argsRun -WindowStyle Hidden -Wait -PassThru
if ($run.ExitCode) { throw "Native case check failed." }
```

省略 `--prefer-recall` 可测试默认模式。报告保存每个输入的 SHA-256、真实 PNG 路径、候选分数和置信等级。本机案例及结果见 [案例库说明](../cases/README.md)。
