# 单图曝光时刻候选

Windows x64 研究工具从一张对齐的 512×512 人像生成多个候选。运行时使用本地 OpenVINO 模型和 OpenCV。控制值范围为 0 到 1，默认生成 0、0.5、1 三个候选。控制值描述该模型学到的外观顺序，无法当作照片的实际拍摄时刻。

## 构建

从该工具目录运行以下命令。参数使用 Visual Analysis 同一套 Windows SDK、MSVC、OpenCV 和 OpenVINO 目录。

```powershell
.\build.ps1 -OutputDirectory D:\DevTools\build-portrait-moments `
    -OpenVinoDirectory $env:OPENVINO_DIR -OpenCvDirectory $env:OPENCV_BUILD `
    -MsvcDirectory $env:MSVC_TOOLS -WindowsSdkDirectory $env:WINDOWS_SDK
```

## 运行

模型接受 `[1,4,256,256]` 浮点数，前三个平面为 0 到 1 的 RGB，最后一个平面为控制值。输出为 `[1,3,256,256]` 浮点 RGB。测试用的本地导出在 `D:/DevTools/visual-analysis-dev/portrait-exposure-inversion-01/cfmd-native.onnx`。模型来源和数值对照见[评测记录](../../../docs/portrait-native-paired-evaluation.md)。

将 OpenVINO `libs` 及 OpenCV `x64\vc16\bin` 目录加入 `PATH`。输入是一张已对齐的人像，输出目录使用新路径。完成全部候选后才发布目录，PNG 按控制值的参数顺序编号。

```powershell
& D:\DevTools\build-portrait-moments\PortraitMoments.exe CPU `
    .\cfmd-native.onnx .\portrait.png .\candidates 0 0.5 1
```

## 从单张糊图生成四张联合训练的候选

`--refine-sequence` 以同一张 512×512 对齐糊图运行 Jin 七帧 ONNX，再用局部修复模型处理第 1、3、5、7 个位置。模型读取糊图、当前帧以及七帧的均值和变化范围，把修正量接回原始帧的脸区。输出目录有四张 PNG，顺序对应七帧模型的位置。

```powershell
& D:\DevTools\build-portrait-moments\PortraitMoments.exe --refine-sequence GPU `
    .\jin-240.onnx .\set-192.onnx .\portrait.png .\candidates
```

七帧图接受 `[1,3,240,240]`，输出 `[1,21,240,240]`。修复图接受 `[1,13,192,192]`，输出 `[1,3,192,192]`，两者均为本地 FP32 模型。已验证的七帧图及修复图分别位于 `D:/DevTools/visual-analysis-dev/portrait-hypercut-01/jin-240.onnx` 和 `D:/DevTools/visual-analysis-dev/portrait-unordered-refinement-01/set-192.onnx`。构建和推理均可在 Windows 上使用本地依赖。[训练与配对评测](../../../docs/portrait-native-paired-evaluation.md)
