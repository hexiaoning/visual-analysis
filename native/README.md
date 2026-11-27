# Visual Analysis 原生 Windows 版

在 Windows 11 x64 上打开 `VisualAnalysis.exe`。通过顶部导航在同一窗口切换分析功能。Win32 提供窗口，OpenCV C++ 负责图像处理，ONNX Runtime C++ 直接加载本机模型，FFmpeg C API 读取并导出视频。SQLite 在本机保存人员库、标注和语义索引。

## 使用方法

- 图片人像还原使用 SCRFD 关键点、PCFlow 和 GFPGAN v1.4。分数综合原图一致性与有原图支撑的边缘细节，降低碎纹较多的候选排名。同一张照片的近似输出只保留一个。第二个候选须同时通过原图一致性、纹理和眼嘴局部差异筛选。
- 勾选「偏召回」会多试去屏纹输入。常规检测未找到人脸时，也会补做一次低阈值检测。导出前使用同样的候选筛选规则。
- 图片车牌还原需要先框选车牌区域，按多个方向和长度去拖影，保留视觉差异足够的候选。
- 人像视频使用 GFPGAN v1.4 和对齐后的残差平滑，导出时保留帧时间和音轨。
- 「视频车牌」可以预览起始帧并框选车牌。程序跟踪它在接下来几帧中的位置，按多帧一致性融合，并保存所用原始选区和结果。
- 「人脸检索」包含单张照片查库、两张单人照片比对、视频和本机摄像头检索。同一次可混选照片与视频，打开后自动检索，处理中可继续追加素材。在「目标人脸库」按库名勾选一个或多个库，也可以全选。查库时还可按性别、地区码和年龄筛选。此前录入的记录继续使用 `%LOCALAPPDATA%\FaceMatching\data\faces.db`。
- 「人车检测」读取照片或视频，可导出带检测框的 MP4 或目标截图。选择源视频和保存目录后，可在首帧画线或框选矩形区域，自动保存目标穿线或入区时的原始整帧。「以图搜索」支持画面选区、外观相似和人脸相似。「语义搜索」支持人物衣着、车辆颜色、磁盘索引和批量导出目标截图。
- 「车型识别」默认实时查询 Wikimedia 和 NHTSA vPIC。支持本地参考图库，车型查询通过独立进程进行。其他分析功能使用本机数据。[车型资料说明](../docs/vehicle-catalogues.md)

神经网络优先使用本机 DirectX 12 显卡。Intel 显卡上的 RT-DETR 人车检测和中文图文编码优先使用 OpenVINO，其余模型使用 DirectML。可用 `VISUAL_ANALYSIS_BACKEND=cpu` 运行兼容模式。人脸库的旧照片和旧特征保存在本机。既有特征的模型档与当前模型不同时，可在「人员库」按原始照片重建特征。

## 构建与案例检查

用 `scripts/build_native.ps1`，指定 MSVC x64、Windows SDK、ONNX Runtime DirectML 1.24.4、DirectML 1.15.4、OpenVINO 2026.4.1、OpenCV 4.13.0、FFmpeg 8.1 LGPL shared 版和 nlohmann/json 3.12.0 单头文件目录。依赖可预先放在构建机，脚本校验模型权重与 JSON 头文件。[ONNX Runtime C++](https://onnxruntime.ai/docs/get-started/with-cpp.html) · [OpenCV Windows 发布包](https://github.com/opencv/opencv/releases/tag/4.13.0) · [FFmpeg Windows 构建](https://ffmpeg.org/download.html)

GUI 默认在 EXE 同级的 `models/` 读取权重。向构建脚本传入 `-ModelDirectory`，可在校验 SHA-256 后复制 SCRFD、PCFlow 及 GFPGAN 权重。PCFlow 与 GFPGAN 使用现有已校验的产物，无需安装原项目 Python SDK。[PCFlow](https://github.com/aiimaginglab/PCFlow) · [GFPGAN](https://github.com/TencentARC/GFPGAN)

```powershell
$exe = 'D:\Data\Build\visual-analysis-dev\native\build\VisualAnalysis.exe'
$argsRun = @('--check-restoration-cases', 'D:\visual-analysis\cases\inputs',
    '--case-output', 'D:\visual-analysis\cases\outputs',
    '--report', 'D:\visual-analysis\cases\outputs\native-report.json',
    '--models', 'D:\Data\Build\visual-analysis-dev\models', '--prefer-recall')
$run = Start-Process -FilePath $exe -ArgumentList $argsRun -WindowStyle Hidden -Wait -PassThru
if ($run.ExitCode) { throw "Native case check failed." }
```

省略 `--prefer-recall` 可测试默认模式。报告保存每个输入的 SHA-256、真实 PNG 路径、候选分数和置信等级。本机案例及结果见 [案例库说明](../cases/README.md)。
