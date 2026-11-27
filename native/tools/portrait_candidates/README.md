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

## 生成不同嘴形的候选

`--mouth-variants` 对一张 512×512 对齐候选生成若干局部嘴形。开合量是内唇沿嘴角连线法向的跨度除以嘴角距离，参数范围为 `(0, 1]`。同值只处理一次。变形限制在口部，位移向嘴角和外缘平滑衰减。模型再对输出测量一次开合量。`selection.json` 记录请求值、重测值和 PNG 文件名。

```powershell
& $exe --mouth-variants .\hr-nosa-wflw-geometry.onnx .\candidate.png .\outputs 0.08 0.20
```

## 对多张候选做严格回放筛选

`--select-by-replay` 对同一张糊图的首选和一个或多个原始备选进行比较。输入都在同一 512×512 对齐坐标下。近似备选的拟合权重须大于 0.50，回放误差须不高于首选，测得的脸部带通强度须至少为糊图的 1.20 倍、至多为首选的 1.40 倍，才可替换为单张。上限用于减少屏摄细纹重新进入单张结果的情况。没有合格替换时，沿用已有的五官差异和回放条件挑选第二张，同时要求它达到糊图的 1.20 倍带通强度。同一类备选中优先选回放误差最低的，误差相同时沿用参数顺序。没有合格备选时输出首选。

```powershell
& $exe --select-by-replay .\hr-nosa-wflw-geometry.onnx .\source.png .\primary.png .\outputs .\alternate-a.png .\alternate-b.png
```

`selection.json` 的 `compared_alternate_index` 为报告所用备选在参数列表里的位置，从 1 开始。`selected` 沿用 `codeformer` 和 `amd` 两个字段名，分别代表所给首选和该备选。输出图使用所选输入候选的像素。[独立配对评测](../../../docs/portrait-native-paired-evaluation.md)

## 合并三路形状候选的细节

`--combine-detail-consensus` 接收同一首图回贴出来的三张 512×512 对齐候选，逐通道取三者中值。只需本地输入，在指定的新路径写一张 PNG。

```powershell
& $exe --combine-detail-consensus .\fixed-points.png .\shape-a.png .\shape-b.png .\combined.png
```

### 重组同一曝光的两个候选

`--recombine-exposure` 接收由同一张糊图生成的两个 512×512 对齐候选。用本地 HR-noSA 定位双眼和嘴，分别双向交换对应区域。两张输出使用相同的羽化遮罩，在线性 RGB 中一边增加的量恰好从另一边扣除。因此在等权曝光模型下，两张图的逐像素平均在量化前保持一致。保存在 8 位 PNG 后存在舍入误差。该命令可为一对输入各生成一对眼部图和一对口部图。

```powershell
& $exe --recombine-exposure .\hr-nosa-wflw-geometry.onnx .\phase-1.png .\phase-7.png .\exchanges
```

鼻部点位偏移过大、羽化边缘周围的颜色差过大或区域尺寸异常时会跳过相应结果。没有有效像素变化时也只写回执。`selection.json` 记录每个区域的状态、有效输出、修改的像素数以及量化前后平均值的误差。可在命令末尾追加 `first.png second.png output-directory`，复用关键点模型。[配对验证](../../../docs/portrait-native-paired-evaluation.md)

中值合并减少单一路线独有的改动。用于评测的展示规则及独立配对结果另有记录。[配对评测](../../../docs/portrait-native-paired-evaluation.md)

## 按五官组合多张候选

`--compose-local-donors` 接收同一张糊图衍生的对齐底图和三张供体，尺寸均为 512×512。参数文件的四行依次对应画面左眼、右眼、鼻和嘴，每行按所给供体顺序列出三项权重。剩余权重分配给底图。脸部软窗外保持底图像素，重叠处利用归一化的软窗混合。结果写入一个新 PNG 路径。

```json
{
  "schema": "portrait-local-donors-v1",
  "weights": [[1, 0, 0], [0, 0.5, 0.25], [0, 0, 0], [0, 1, 0]]
}
```

```powershell
& $exe --compose-local-donors .\weights.json .\baseline.png .\first.png .\second.png .\third.png .\mixed.png
```

权重由调用方提供。该命令只完成本地像素组合。[配对评测](../../../docs/portrait-native-paired-evaluation.md)

## 限制候选的细带强度

`--cap-fine-band` 接收一张 512×512 对齐候选和以归一化亮度表示的细带 RMS 上限。程序只在脸部软窗内调整亮度，保持每个像素的 BGR 通道差。上限由调用方提供。结果写入一个新 PNG 路径，标准输出同时报告 `target_rms`、`output_rms` 以及在 2% 容差内是否达到目标的 `reached`。窗外细节、每像素限幅和整数化都可能使设定上限无法达到。

```powershell
& $exe --cap-fine-band .\candidate.png 0.025 .\limited.png
```

## 单张模糊图的曝光校正

`--normalize-exposure` 在生成候选之前运行，输入为一张已对齐的 512×512 BGR 图像。参数文件由公开配对数据的离线训练生成。此模式使用 C++ 和 OpenCV 估计亮度及色偏，需要提供研究参数文件，无需加载关键点模型。示例将得到一张新路径下的 PNG，可送入原生人像生成器。

```powershell
& $exe --normalize-exposure .\exposure-model.json .\source.png .\corrected.png
```

标准输出为一行 JSON，包含 `applied`、BGR 三通道 `gain_bgr`、`black_level`、是否触发校正的打分 `evidence`，以及 `recovered_grid`。采样网格未检出时为 0，低于校正阈值时输出像素与输入相同。

参数文件包含 99 项的 `center`、`scale` 和每行 99 项的 `centers`。设特征长度为 99、基函数个数为 K，`weights` 为 (100+K)×4 数组，`classifier` 长度为 100+K，`gate_limit` 为标量。设计向量依次为常数 1、中心化后的特征、RBF 响应。研究参数由 CelebA 配对拟合，保存于本机评测目录。公开样本及由其拟合的本轮参数用于研究评测。[评测与限制](../../../docs/portrait-native-paired-evaluation.md) · [CelebA 数据使用协议](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html#agreement)

## 已生成候选的亮度匹配

`--match-candidate-tone` 接收同一糊图生成的两张 512×512 对齐彩色图。先给需要调整的候选，再给作为亮度参照的候选。命令在对数线性颜色空间计算两张图的低频差异，从脸部中央向边缘渐变应用到第一张图。像素位置和局部细节来自第一张图。命令只读本地图片，结果写入一个新 PNG 路径。

```powershell
& $exe --match-candidate-tone .\candidate.png .\guide.png .\tone-matched.png
```

`--condition-candidate-tone` 也使用两张 512×512 对齐彩色图。它将参照和候选的 BGR 差值按给定像素尺度平滑，再在脸部软窗内向候选加入有限的低频校正。尺度取 4 到 24。窗外像素逐个保持，输出使用新路径。对屏拍的试验使用从同一输入测得周期后得到的去纹图作为参照。

```powershell
& $exe --condition-candidate-tone 6 .\candidate.png .\depatterned-observation.png .\conditioned.png
```

此研究模式使用固定校正强度。评测中不同候选的效果有差异，可将输出送入同一糊图的回放筛选，核对是否适合作为单张结果。[配对评测](../../../docs/portrait-native-paired-evaluation.md)

## 已生成候选的屏拍颜色校准

`--calibrate-capture-color` 在同一 512×512 BGR 对齐坐标下读取原始拍摄图和已生成的候选。用两张图的亮度分位数及相应色差统计，估算线性 B、R 通道相对于 G 的校正值。校正只在脸部软窗内改变色度，保留候选的逐像素 8 位灰度。任一图有效像素不足时输出原候选。

```powershell
& $exe --calibrate-capture-color .\capture-color.json .\source.png .\candidate.png .\corrected.png
```

指定一个新输出文件。命令打印 `log_gain_blue_red` 两个对数增益。模型 JSON 的 `schema` 为 `portrait-capture-color-quantiles-v1`，`center`、`scale` 各有 28 项，`bias` 有两项，`weights` 为 28×2。该模式用于屏拍配对校色研究，运行时仅读取本地 JSON 和图像。[配对验证](../../../docs/portrait-native-paired-evaluation.md)

## 已生成候选的局部几何校正

`--calibrate-capture-geometry` 读取原始观察、由同一观察生成的两路去纹图和一张既有候选。四张图都使用同一 512×512 BGR 对齐坐标。程序从两档亮度频带测量局部位移，再用配对标定的模型预测每个窗口的有限修正量。像素经连续位移场从原候选重采样，脸部软窗外保持原样。

```powershell
& $exe --calibrate-capture-geometry .\geometry.json .\source.png .\guide-a.png .\guide-b.png .\candidate.png .\corrected.png
```

模型 JSON 的 `schema` 为 `portrait-capture-geometry-linear-v1`，`center` 和 `scale` 各有 31 项，`bias` 有两项，`weights` 为 31×2。研究命令仅使用本地输入，结果存到新的 PNG 路径。[配对验证与选择限制](../../../docs/portrait-native-paired-evaluation.md)

## 选择规则和输出

- 默认保留 CodeFormer。同时展示 AMD 要求其回放误差不超过 CodeFormer 加 0.007，清晰度至少为 CodeFormer 的 96%，并有一个区域同时通过位移和像素检验。区域扣除整体相似变换后的平均位移须大于 3 像素。以主候选关键点覆盖区域外扩 8 像素计算亮度带通差异，原幅度差的均方根须达到 1.5 个灰阶，按局部带通强度归一化后的均方根须达到 0.25。`visible_shape_difference` 记录这两项检验的联合结果。
- 满足回放条件且没有通过联合形状检验的区域时，程序在共同的模糊和曝光模型下拟合 AMD 的混合系数。`--fit-duplicate` 开启后，系数大于 0.5 时展示 AMD，否则展示 CodeFormer。输出始终是原候选的像素。这个系数衡量重投影吻合度，身份保真度仍需用独立清晰图评价。
- `selection.json` 中 `selected` 是实际展示顺序，`previous_rule_selected` 是默认规则，`fitted_duplicate_rule_selected` 是启用近重复拟合时的顺序，`fit_duplicate_enabled` 记录本次开关。
- `clarity` 是脸部带通响应的均方根，`clarity_vs_source` 是相对于输入的比值。`replay_error` 是在 128×128 亮度图上、脸部特征区域内测得的截断残差。曝光或特征区有效像素不足时误差为 `null`，对应回放图也为空。
- `candidate-1.png` 和可能的 `candidate-2.png` 对应实际展示的候选。`observed-luminance.png` 与 `candidate-*-replay.png` 供直接核对观测和评分用回放。

## 为候选库保留外观代表

`--cover-shapes` 接受已对齐的 512×512 BGR 候选。先保留首张，再从其余图中补入离已保留图最远的一张，直到每张输入和至少一张代表的实测五官距离都在指定半径内。半径可取 0 到 2。距离取左右眼、鼻和口中最远的区域，以亮度带通响应计算。如果只有一张图在该区域有明显结构，该区域距离计为 1。

```powershell
& $exe --cover-shapes 0.10 .\cover-output .\primary.png .\alternative-1.png .\alternative-2.png
```

输出的候选 PNG 与对应输入像素一致。`selection.json` 中的 `outputs` 按保留顺序列出原输入序号及文件名。`nearest_input_index` 和 `measured_distance` 为每张输入列出最近代表的原输入序号及距离。所有序号从 0 开始。该半径只约束候选库中已生成的图和上述距离，外观评审仍需与可能的清晰原像比较。

## 在原结果后追加有观测支持的候选

`--extend-shape-cover` 接收原始糊图和同一 512×512 对齐坐标的一组候选。开头指定数量的候选按原顺序输出，默认最多追加一张。程序允许小幅平移和模糊回放，在分开的图块拟合及检验亮度。追加图要求脸部细频不低于糊图，回放误差不超过原结果最小误差加给定容差，而且与每张原结果的五官带通距离均大于半径。若有多张满足，优先保留能覆盖较多相近且有观测支持的备选的候选，降低近重复簇的权重。

```powershell
& $exe --extend-shape-cover 0.055 0.028 1 --landmarks .\hr-nosa-wflw-geometry.onnx .\source.png .\outputs .\primary.png .\alternative-1.png .\alternative-2.png
```

可选 `--landmarks` 使用本地 HR-noSA 对追加图和每张原结果再做一次关键点及局部像素差异检验，全部通过才追加。`selection.json` 记录 `landmark_checks`、各输入的 `replay_errors`、到原结果集合的最短距离 `distance_to_protected` 及实际输出。输出直接使用所选候选像素。回放容差采用 0 到 1 的亮度标度。[配对验证](../../../docs/portrait-native-paired-evaluation.md)

```powershell
& $exe --extend-shape-cover 0.055 0.028 1 --max-additions 3 --landmarks .\hr-nosa-wflw-geometry.onnx .\source.png .\outputs .\primary.png .\alternative-1.png .\alternative-2.png
```

显式传入 `--max-additions` 时，每接受一张就更新覆盖距离。新图必须与全部已输出图满足距离门槛。启用 `--landmarks` 时，还需要逐对通过关键点及局部像素检验。被拒绝后继续尝试其他候选，能通过多少张就输出多少张。`0` 只输出指定的开头候选。该模式用 `pair_checks` 记录逐对结果，用 `max_additions` 记录追加上限。

## 固定数量时复核已有候选

```powershell
& $exe --refine-representatives 2 .\hr-nosa-wflw-geometry.onnx .\source.png .\outputs .\face-1.png .\face-2.png .\alternative.png
```

保持展示数量和位置，最多替换其中一张。备选的五官带通距离需接近拟替换的图，脸部细频需接近糊图及旧候选。将脸区分成四组交错图块，每次用其中三组为旧候选拟合同一个模糊、平移及亮度变换，锁定参数后分别回放旧图和备选，用剩下一组比较误差。四组都达到至少 3% 且至少 0.00015 的改善后，再用 HR-noSA 验证两张是否落在同一外观组、备选是否与其他已展示图不同。`selection.json` 的 `replacement` 包含被替换与新候选索引、五官距离及四组回放改善。没有达标替换时为 `null`。回放分数衡量与输入的吻合程度。[配对验证](../../../docs/portrait-native-paired-evaluation.md)

可用 `--refine-clear-representatives` 检查同一外观组中更清晰的候选。先给要复核的原代表数量和应保护的追加数量，再按原代表、已追加候选、其余候选的顺序列出文件。输出保持原显示数量，每次至多替换一张原代表。

```powershell
& $exe --refine-clear-representatives 1 1 .\hr-nosa-wflw-geometry.onnx .\source.png .\outputs `
    .\primary.png .\previous-addition.png .\alternative.png
```

在双眼、鼻和口测量与粗边缘同向的细梯度，要求候选在至少两个部位有清晰度提升。四折回放确认它与原图的吻合度。关键点及局部像素检查保留已有的五官差异。替换后若会与已追加的外观重复，就保留之前的组合。回执 `replacement` 记录各部位清晰度变化、四折误差变化及容差和最大位移变化。

## 检查整组候选的展示约束

```powershell
& $exe --inspect-appearance-bank 0.055 .\hr-nosa-wflw-geometry.onnx .\source.png .\outputs .\primary.png .\alternative-1.png .\alternative-2.png
```

原糊图与候选使用同一 512×512 对齐坐标。该命令在新建的运行目录写入 `selection.json`。`source_detail` 和 `detail` 是脸区细频强度，`replay_errors` 是配准后的亮度回放误差，`distance` 为两两五官带通距离。`compatible` 要求距离高于给定半径，同时以较早输入的候选为参照通过关键点及局部像素差异检查。两张矩阵均按候选输入顺序排列，对角线分别为 0 和 `false`。这些数据用于复核选图约束。

用 `--distance-only` 替代模型路径时只比较带通距离，`landmarks_enabled` 记录是否检测关键点。

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
