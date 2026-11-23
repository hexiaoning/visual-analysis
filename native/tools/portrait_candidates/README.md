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

准备 WFLW 版 HR-noSA ONNX 关键点模型。已验证导出的 SHA-256 为 `7c155d39f979ff29d8af4801f8bfe24e94c3b4c154628f5d9aa9a53346efda66`，输入为 `pixels`，输出为 `landmarks`，形状为 1×98×2。 此导出使用 ONNX opset 17，接收 1×3×256×256 的 BGR float32 张量（归一化到 [-1, 1]）。对应 WFLW 权重的 SHA-256 为 `066d150329f57bdba780ae8915a7ec7c158436c8affaba6dfcd98603cab84b1e`。[作者模型实现](https://github.com/ca-joe-yang/regression-without-softarg/tree/b391d21aae976fdeecbc22285aae8702faecd00e) · [作者 WFLW 权重](https://huggingface.co/Yeh-Purdue/regression-without-softarg/tree/5f47d639d868cf54afcc6226091a5178165e68be/WFLW)

```powershell
$exe = Join-Path $build.OutputDirectory 'PortraitCandidates.exe'
& $exe .\hr-nosa-wflw-geometry.onnx .\source.png .\codeformer.png .\amd.png .\outputs
& $exe --fit-duplicate .\hr-nosa-wflw-geometry.onnx .\source.png .\codeformer.png .\amd.png .\outputs
& $exe --texture-guard .\texture-guard-v1.json .\hr-nosa-wflw-geometry.onnx .\source.png .\codeformer.png .\amd.png .\outputs
```

标准输出每行是一个新结果目录的 UTF-8 绝对路径。批量处理可在同一命令后重复追加四个参数 `source codeformer amd output-directory`。同一批复用关键点模型会话。每个成功样本独立发布，重试会生成单独的结果目录。

## 单张模糊图的曝光校正

`--normalize-exposure` 在生成候选之前运行，输入为一张已对齐的 512×512 BGR 图像。参数文件由公开配对数据的离线训练生成。此模式使用 C++ 和 OpenCV 估计亮度及色偏，需要提供研究参数文件，无需加载关键点模型。示例将得到一张新路径下的 PNG，可送入原生人像生成器。

```powershell
& $exe --normalize-exposure .\exposure-model.json .\source.png .\corrected.png
```

标准输出为一行 JSON，包含 `applied`、BGR 三通道 `gain_bgr`、`black_level`、是否触发校正的打分 `evidence`，以及 `recovered_grid`。采样网格未检出时为 0，低于校正阈值时输出像素与输入相同。

参数文件包含 99 项的 `center`、`scale` 和每行 99 项的 `centers`。设特征长度为 99、基函数个数为 K，`weights` 为 (100+K)×4 数组，`classifier` 长度为 100+K，`gate_limit` 为标量。设计向量依次为常数 1、中心化后的特征、RBF 响应。研究参数由 CelebA 配对拟合，保存于本机评测目录。公开样本及由其拟合的本轮参数用于研究评测。[评测与限制](../../../docs/portrait-native-paired-evaluation.md) · [CelebA 数据使用协议](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html#agreement)

## 选择规则和输出

- 默认保留 CodeFormer。只有 AMD 的回放误差不超过 CodeFormer 加 0.007，清晰度至少为 CodeFormer 的 96%，且扣除整体相似变换后至少一个五官区域的平均位移大于 3 像素，才同时展示 AMD。
- 满足回放条件、五个区域平均位移均不超过 3 像素时，程序在共同的模糊和曝光模型下拟合 AMD 的混合系数。`--fit-duplicate` 开启后，系数大于 0.5 时展示 AMD，否则展示 CodeFormer。输出始终是原候选的像素。这个系数衡量重投影吻合度，身份保真度仍需用独立清晰图评价。
- `selection.json` 中 `selected` 是实际展示顺序，`previous_rule_selected` 是默认规则，`fitted_duplicate_rule_selected` 是启用近重复拟合时的顺序，`fit_duplicate_enabled` 记录本次开关。
- `clarity` 是脸部带通响应的均方根，`clarity_vs_source` 是相对于输入的比值。`replay_error` 是在 128×128 亮度图上、脸部特征区域内测得的截断残差。曝光或特征区有效像素不足时误差为 `null`，对应回放图也为空。
- `candidate-1.png` 和可能的 `candidate-2.png` 对应实际展示的候选。`observed-luminance.png` 与 `candidate-*-replay.png` 供直接核对观测和评分用回放。

## 纹理保护实验

`--texture-guard <model.json>` 包含近重复拟合。仅当拟合会把主候选替换成 AMD 时，程序计算 AMD 的纹理分。分数小于或等于科研模型的固定阈值 **0.5641734981312992** 时接受替换，超过阈值则保留原先选择。模型使用绝对分数，计算过程中只读取 AMD 的候选图。近重复以外的双张展示按原规则处理。

`selection.json` 的 `mode` 记录实际使用的模式，`texture_guard_selected` 记录保护后的选择；以上两条已有规则的结果也分别保留。加载保护模型后记录 `texture_cutoff`。只有需要评估替换时才记录 `replacement_texture_score`，其余情况为 `null`。

科研用的 `texture-guard-v1.json` 包含 96 个线性系数、截距和阈值，可直接由 C++ 读取。训练程序 `train_face_artifact_protection.py` 的 `fit()` 使用 DAEFR 的清晰照片和对应的模拟光学模糊、屏拍退化。160 张照片用于训练，另外 40 张用于校准；每张提供清晰和模糊两个负例、叠加屏纹和模拟屏拍两个正例。特征是在 256×256 图上对亮度和两个色差通道分四个频带，在人脸区域取 16×16 块的原始与局部对比度归一化能量的四个分位数。特征标准差下限为 0.05，岭回归惩罚为训练例数的 0.1 倍，阈值取校准负例的第 95 百分位。模型文件内记录了原始参数包、训练脚本和冻结协议的 SHA-256。

在另行冻结的 20 张照片上，默认选择和单独拟合各通过 10 张，纹理保护通过 11 张。16 张模拟屏拍中，保护后通过 11 张；另 4 张暗光运动模糊在三种模式下均未通过。此前的 60 张属于开发对照。

## 公开样本回归

使用此前已生成的 DAEFR 测试样本三元组。在同一 `fixtures` 目录下建立 `1848`、`2762`、`739`、`1003` 和 `1432` 五个子目录，各放入 `source.png`、`codeformer.png` 和 `amd.png`。表中结果只对应当时的固定候选。验证脚本通过 Windows PowerShell 和 .NET System.Drawing 解码 PNG，并核对固定样本的 BGR 像素 SHA-256。

| 样本 | 默认展示 | 单独拟合 | 纹理保护 |
| --- | --- | --- | --- |
| 1848 | CodeFormer | AMD | CodeFormer |
| 2762 | CodeFormer、AMD | CodeFormer、AMD | CodeFormer、AMD |
| 739 | CodeFormer | CodeFormer | CodeFormer |
| 1003 | CodeFormer | AMD | CodeFormer |
| 1432 | CodeFormer | AMD | AMD |

```powershell
.\verify_public.ps1 -Executable $exe -Model .\hr-nosa-wflw-geometry.onnx -FixtureDirectory .\fixtures
```

验证脚本同时检查三种模式的选择、保护分数及阈值、观测与回放的灰度像素以及每张展示图与对应原候选的逐像素一致性。
