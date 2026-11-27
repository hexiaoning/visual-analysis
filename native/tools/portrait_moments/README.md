# 单图曝光时刻候选

Windows x64 研究工具从一张对齐的 512×512 人像生成多个候选。运行时使用本地 OpenVINO 模型和 OpenCV。控制值范围为 0 到 1，默认生成 0、0.5、1 三个候选。控制值描述该模型学到的外观顺序，无法当作照片的实际拍摄时刻。

## 构建

从该工具目录运行以下命令。参数使用 Visual Analysis 同一套 Windows SDK、MSVC、OpenCV 和 OpenVINO 目录。

```powershell
.\build.ps1 -OutputDirectory D:\Data\Build\portrait-moments `
    -OpenVinoDirectory $env:OPENVINO_DIR -OpenCvDirectory $env:OPENCV_BUILD `
    -MsvcDirectory $env:MSVC_TOOLS -WindowsSdkDirectory $env:WINDOWS_SDK
```

## 运行

模型接受 `[1,4,256,256]` 浮点数，前三个平面为 0 到 1 的 RGB，最后一个平面为控制值。输出为 `[1,3,256,256]` 浮点 RGB。测试用的本地导出在 `D:/Data/Build/visual-analysis-dev/portrait-exposure-inversion-01/cfmd-native.onnx`。模型来源和数值对照见[评测记录](../../../docs/portrait-native-paired-evaluation.md)。

将 OpenVINO `libs` 及 OpenCV `x64\vc16\bin` 目录加入 `PATH`。输入是一张已对齐的人像，输出目录使用新路径。完成全部候选后才发布目录，PNG 按控制值的参数顺序编号。

```powershell
& D:\Data\Build\portrait-moments\PortraitMoments.exe CPU `
    .\cfmd-native.onnx .\portrait.png .\candidates 0 0.5 1
```

## 从单张糊图生成四张联合训练的候选

`--refine-sequence` 以同一张 512×512 对齐糊图运行 Jin 七帧 ONNX，再用局部修复模型处理第 1、3、5、7 个位置。模型读取糊图、当前帧以及七帧的均值和变化范围，把修正量接回原始帧的脸区。输出目录有四张 PNG，顺序对应七帧模型的位置。

```powershell
& D:\Data\Build\portrait-moments\PortraitMoments.exe --refine-sequence GPU `
    .\jin-240.onnx .\set-192.onnx .\portrait.png .\candidates
```

七帧图接受 `[1,3,240,240]`，输出 `[1,21,240,240]`。修复图接受 `[1,13,H,H]`，输出 `[1,3,H,H]`，支持 `H=192` 和 `H=512`。输入、输出均为本地 FP32。192 像素修复图使用下采样观察并把修正放回 512 像素。512 像素修复图直接使用原观察，七帧以立方插值和 8 位量化对齐到同一分辨率。已验证的七帧图及 192 像素修复图分别位于 `D:/Data/Build/visual-analysis-dev/portrait-hypercut-01/jin-240.onnx` 和 `D:/Data/Build/visual-analysis-dev/portrait-unordered-refinement-01/set-192.onnx`。[训练与配对评测](../../../docs/portrait-native-paired-evaluation.md)

可在同一命令末尾重复提供 `portrait.png output`，复用已编译的两张模型图及推理请求。每张输入的四张图完整写出后单独发布，使用互不相同的新输出路径。

```powershell
& D:\Data\Build\portrait-moments\PortraitMoments.exe --refine-sequence GPU `
    .\jin-240.onnx .\signed-512.onnx .\portrait-1.png .\candidates-1 .\portrait-2.png .\candidates-2
```

可用 `--positions` 从七帧中指定输出位置。默认 `1,3,5,7`。自定义列表以逗号分隔，取值为 1 至 7，每个位置只写一次。按传入顺序编号 PNG，所有输入图共用该列表。位置描述七帧模型的输出索引。

```powershell
& D:\Data\Build\portrait-moments\PortraitMoments.exe --refine-sequence GPU `
    .\jin-240.onnx .\signed-512.onnx --positions '2,4,6' .\portrait.png .\intermediate-candidates
```


`--additional-refinement` 可重复指定其他修复模型，放在首个修复模型之后、`--positions` 之前。每张输入只运行一次七帧模型。输出按模型顺序分组，再按指定位置顺序编号。日志用 `refinement` 记录模型序号。以默认四个位置和两个模型为例，`candidate-1.png` 至 `candidate-4.png` 属于首个模型，`candidate-5.png` 至 `candidate-8.png` 属于第二个模型。同一张输入的所有结果齐全后才发布目录。

```powershell
& D:\Data\Build\portrait-moments\PortraitMoments.exe --refine-sequence GPU `
    .\jin-240.onnx .\decoder-a.onnx --additional-refinement .\decoder-b.onnx `
    .\portrait.png .\eight-candidates
```
