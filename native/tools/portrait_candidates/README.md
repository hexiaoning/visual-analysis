# 离线人脸候选筛选

本研究命令输入同一人脸的三张 512×512 对齐裁图，依次为原始观察、已生成的 CodeFormer 和 AMD PSFR-GAN 候选。它用 HR-noSA 的 98 个关键点测量形状变化，并将候选模糊、校正曝光后与原始观察比较。关键点推理及图像计算均使用 CPU 单线程。运行时只读取本地文件。

## 构建

使用 MSVC x64、Windows SDK、OpenCV 4.13.0 Windows world 包、ONNX Runtime 1.24.4 x64 CPU NuGet 包及 nlohmann/json 3.12.0 单头文件。构建参数与仓库主构建脚本采用相同目录结构。下例中的环境变量指向已展开的依赖根目录，`JSON_HEADERS` 内应有 `nlohmann/json.hpp`。

```powershell
$build = @{
    OutputDirectory = "$env:TEMP\portrait-candidates"
    MsvcDirectory = $env:MSVC_TOOLS
    WindowsSdkDirectory = $env:WINDOWS_SDK
    OpenCvDirectory = $env:OPENCV_BUILD
    OnnxRuntimeDirectory = $env:ORT_NUGET
    JsonIncludeDirectory = $env:JSON_HEADERS
}
.\build.ps1 @build
```

默认 SDK 版本为 10.0.26100.0，可用 `-WindowsSdkVersion` 指定。脚本将程序及 OpenCV、ORT DLL 放入输出目录。目标机可安装 x64 Visual C++ 运行库，也可用 `-CrtDirectory` 将现有 x64 运行库复制到程序旁。如果复用主应用的 DirectML 版 ORT 包，另传 `-DirectMlDirectory` 复制依赖 DLL，命令仍使用 CPU。

## 运行

准备 WFLW 版 HR-noSA ONNX 关键点模型。已验证导出的 SHA-256 为 `7c155d39f979ff29d8af4801f8bfe24e94c3b4c154628f5d9aa9a53346efda66`，输入为 `pixels`，输出为 `landmarks`，形状为 1×98×2。

```powershell
$exe = Join-Path $build.OutputDirectory 'PortraitCandidates.exe'
& $exe .\hr-nosa-wflw-geometry.onnx .\source.png .\codeformer.png .\amd.png .\outputs
& $exe --fit-duplicate .\hr-nosa-wflw-geometry.onnx .\source.png .\codeformer.png .\amd.png .\outputs
```

标准输出每行是一个新结果目录的 UTF-8 绝对路径。批量处理可在同一命令后重复追加四个参数 `source codeformer amd output-directory`。同一批复用关键点模型会话。每个成功样本独立发布，重试会生成单独的结果目录。

## 选择规则和输出

- 默认保留 CodeFormer。只有 AMD 的回放误差不超过 CodeFormer 加 0.007，清晰度至少为 CodeFormer 的 96%，且扣除整体相似变换后至少一个五官区域的平均位移大于 3 像素，才同时展示 AMD。
- 满足回放条件、五个区域平均位移均不超过 3 像素时，程序在共同的模糊和曝光模型下拟合 AMD 的混合系数。`--fit-duplicate` 开启后，系数大于 0.5 时展示 AMD，否则展示 CodeFormer。输出始终是原候选的像素。这个系数衡量重投影吻合度，身份保真度仍需用独立清晰图评价。
- `selection.json` 中 `selected` 是实际展示顺序，`previous_rule_selected` 是默认规则，`fitted_duplicate_rule_selected` 是启用近重复拟合时的顺序，`fit_duplicate_enabled` 记录本次开关。两种选择都在一次运行中计算。
- `clarity` 是脸部带通响应的均方根，`clarity_vs_source` 是相对于输入的比值。`replay_error` 是在 128×128 亮度图上、脸部特征区域内测得的截断残差。曝光或特征区有效像素不足时误差为 `null`，对应回放图也为空。
- `candidate-1.png` 和可能的 `candidate-2.png` 对应实际展示的候选。`observed-luminance.png` 与 `candidate-*-replay.png` 供直接核对观测和评分用回放。

## 公开样本回归

使用此前已生成的 DAEFR 测试样本三元组。在同一 `fixtures` 目录下建立 `1848`、`2762`、`739` 三个子目录，各放入 `source.png`、`codeformer.png` 和 `amd.png`。表中结果只对应当时的固定候选。验证脚本通过 Windows PowerShell 和 .NET System.Drawing 解码 PNG，并核对固定样本的 BGR 像素 SHA-256。

| 样本 | 默认展示 | 启用拟合 |
| --- | --- | --- |
| 1848 | CodeFormer | AMD |
| 2762 | CodeFormer、AMD | CodeFormer、AMD |
| 739 | CodeFormer | CodeFormer |

```powershell
.\verify_public.ps1 -Executable $exe -Model .\hr-nosa-wflw-geometry.onnx -FixtureDirectory .\fixtures
```

验证脚本同时检查两套选择、近重复开关、观测与回放的灰度像素以及每张展示图与对应原候选的逐像素一致性。
