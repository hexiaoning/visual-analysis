# 从一张模糊照片生成人像候选

在仓库根目录运行以下命令。Python 环境需要 NumPy、OpenCV 和 ONNX Runtime。模型目录需含 `scrfd_10g_bnkps.onnx`、`gfpgan-v1.4-fp32.onnx` 和 `pcflow-bfr-fp32.onnx`。输出目录使用新目录名。

```powershell
python scripts/restore_portrait_set.py photo.jpg candidates --models D:/models
```

程序选取检测分数最高且五点对齐通过的人脸，输出 512 × 512 的人脸 PNG。已按 FFHQ 模板对齐的方形单人裁图可加 `--aligned`。模型对原图和镜像分别推理，在暗光输入上也尝试提亮后还原。两种模型对五官位置的判断有差异时，沿它们之间的光流生成更多候选。所有像素不同的结果均写入目录。`manifest.json` 记录输入哈希、对齐方式、候选哈希和生成分支。

要增加三种细节编码结果，在安装了 PyTorch、einops、timm 的环境中加以下参数。第一项目录含 DiffBIR 兼容的 `swinir.py` 和 `face-swinir-v1.ckpt`，第二项是 DicFace 源码根目录，第三项是作者 BFR-100k 权重。使用可信的本机源码。推理会核对两份权重的 SHA-256。

```powershell
python scripts/restore_portrait_set.py photo.jpg candidates-extended --models D:/models --sparse-prior D:/models/ntire2025-swinir D:/models/DicFace D:/models/dicface-bfr-100k.pth
```

## 覆盖判定

目标要求每一种可能的清晰原像都有至少一张候选看起来与它相同。当前实现可以对两条估计的光流路径给出一个较窄的保证。在每条路径的设计区间内，每个连续变形与最近采样变形的回采坐标，在插值之前逐像素相差至多 1 个像素。这个界限描述几何采样密度。真实人像还可能具有这两条路径之外的五官结构，清单将全体原像覆盖记为 `unverified`。

有多份已知清晰参考时，可使用 `scripts/evaluate_portrait_coverage.py`。其文件头给出了清单格式。先用 `--review-sheet` 生成绑定参考及候选哈希的评审表。两幅脸按相同大小显示，逐对检查脸型、双眼、鼻形、嘴形及相对位置。亮度或色调的差异可以单独记录。观察到明显结构变化就选 `different`，结构在整脸和局部均一致才选 `same`，无法判断保留 `unreviewed`。同一张候选需同时满足这些条件。

填好的表放在原清单所在目录，交给同一脚本再次运行。任一已知原像的所有候选均为 `different`，即报告该原像漏覆盖。对受控运动模糊测试可加 `--require-replay`，使用记录的拍摄参数从每一份清晰参考重新生成 JPEG 并逐字节核对。原照和拍摄参数仅用于评估。

对方块平均测试，同样使用 `--require-replay`。清单的 `observation` 填写原生低分辨率 PNG，`replay` 填写 `{"kind": "block-mean-rgb8-v1", "scale": 16}`，其中 `scale` 为方块边长。脚本按默认增益 1、偏置 0 回放每一张参考及候选，要求解码后的观测逐通道一致。评审表同时绑定成像规则和图像文件。回放通过后，外观覆盖仍由评审表逐对记录。

## 已知方块平均成像时的严格覆盖

若输入像素由清晰图的不重叠方块取均值得到，可以给出全部相容 RGB8 图像的隐式表示，并从任意相容人像开始逐张枚举。以下例子的原生输入为对齐的 64 × 64 人脸，方块边长为 8，清晰图尺寸为 512 × 512。普通照片可省略 `--aligned`，程序会把人脸候选映回整幅图后再做一致性检查。

```powershell
python scripts/restore_portrait_set.py native.png output --models D:/models --aligned --sensor-scale 8
python scripts/explore_portrait_preimage.py output/sensor-preimage/set.json check clear.png
python scripts/explore_portrait_preimage.py output/sensor-preimage/set.json enumerate page --count 4 --start output/sensor-preimage/gfpgan.png --discard-bits 1
python scripts/explore_portrait_preimage.py output/sensor-preimage/set.json bound --radius 1
```

只需建立完整相容集合时，可直接运行 `python scripts/explore_portrait_preimage.py native.png define camera-set --scale 8`。这条命令不需要人脸检测或模型权重，极度模糊而无法定位五官的观测也可使用。随后对 `camera-set/set.json` 运行同样的检查和枚举命令。

该成像模型在 RGB8 编码值上对每个方块、每个通道独立求均值，乘以有理数增益 `--sensor-gain`，加整数偏置 `--sensor-offset`，按 `floor(x + 1/2)` 取整，再截到 0–255。默认增益为 1，偏置为 0。程序从观测反推出每个方块每个通道允许的整数总和区间。满足所有区间就属于这个相机的逆像。清单绑定解码后的原生观测，若给定相机无法产生该观测，清单标记为空。

打开 `sensor-preimage` 下的人像 PNG 可查看优先生成的可行解。除模型输出外，程序还对眼距、嘴宽取不同组合，再将每幅图投影到可行集合。投影逐块选择满足总和区间、且相对提案的像素平方改动最小的结果，并重新模拟相机核对每个观测像素。

枚举器的默认设置会遍历每个相容光栅恰好一次。`--discard-bits b` 将每个通道每个像素划入宽度为 `2^b` 的区间，每个有解的组合取一张相容代表。穷尽后，对于任一相容原图，至少有一幅代表在所有通道、所有像素上的误差都不超过 `2^b − 1`。页清单中的 `all_feasible_cells_written` 说明这次是否真的走完。取一个前缀只保证已输出图像通过回放。`bound` 给出按指定逐通道最大误差覆盖全体相容光栅所需数量的数学下界。

真实拍摄可能包含运动模糊、重采样偏移和 JPEG 压缩。这里的逐像素保证只对声明的成像模型成立。数学集合也容许带纹理噪声的图像，人像外观仍需单独评估。

## 从极低分辨率的观测生成可查看的人脸

对于已裁成方形、正面单人脸大体居中的输入，可用 StyleGAN2 FFHQ 512 生成不同五官、脸型和发型的提案。即使输入只剩几个色块，也可以用这条命令。程序在 CPU 上按种子生成人像，先对整幅提案施加连续的色调校正，再修正整数像素，逐通道核对重新成像的结果。只有通过的图才会写入输出目录。每个种子的 PNG、哈希和相机参数列在 `faces.json`。严重欠曝时，候选仍可能带有明显色偏。

```powershell
python scripts/explore_portrait_preimage.py camera-set/set.json sample-faces faces --stylegan-source D:/models/stylegan3 --weights D:/models/stylegan2-ffhq-512x512.pkl --first-seed 0 --count 16
```

运行需要已安装 CPU 版 PyTorch 及 NVIDIA 源码依赖的 Python 环境，可信的本地 NVlabs/stylegan3 源码，以及 NVIDIA 发布的 StyleGAN2 FFHQ 512×512 权重。已测源码提交为 `c233a919a6faee6e36a316ddd4eddababad1adf9`。程序在加载权重前校验 SHA-256。每张模糊图是唯一的个案输入，模型、相机参数和种子范围属于运行配置。延续取样可使用新的输出目录，并将 `--first-seed` 设为上一批的结束种子加一。[官方源码](https://github.com/NVlabs/stylegan3/tree/c233a919a6faee6e36a316ddd4eddababad1adf9) · [官方权重目录](https://catalog.ngc.nvidia.com/orgs/nvidia/research/models/stylegan2/1/file-browser)

种子决定一开始生成哪一张脸。观测用于约束其重新成像的结果。每张候选都可能通过这个约束，而某一张真实原照的长相仍可能在整批中缺席。增加种子数量能扩充候选，程序尚无判定所有人像外观已覆盖的停止条件。

## 两段曝光的精确短曝光范围

如果每个像素为两幅对齐的 8 位画面按已知整数权重混合再取整，长曝光画面允许在观测的逐通道容差内自由变化，Windows 原生命令可以从一张观测确定所有短曝光光栅的可行范围。短曝光的量化单元可按页遍历。每个输出包含能使观测逐像素复现的长曝光画面。用于指导生成的提案如已相容，投影会原样保留。

设短、长画面权重为 `p` 和 `q`，观测通道允许的整数混合总和为 `[L, U]`，长曝光通道取值在 `[a, b]`。短曝光取值范围是 `[ceil((L−q·b)/p), floor((U−q·a)/p)]` 与 `[0,255]` 的交集。中间量化箱至少包含 `q` 个连续整数，因而相邻长曝光取值产生的可行区间之间没有缺口。对纯黑和纯白，短曝光与长曝光的物理范围已满足外侧边界。所有像素在这个模型里独立，因此两幅边界图定义的逐通道区间乘积就是完整相容集合。

这一保证覆盖上述成像模型的相容光栅。实际照片还可能有空间位移、非线性传感器响应、压缩或屏纹。用真实图验证时仍须检查已知清晰原像是否落在该模型内，并另行判断最终输出的人像外观。[原生命令与续跑方法](../native/tools/portrait_candidates/README.md#枚举两段曝光相容的短曝光画面)

可另外允许最多 `K` 个通道出现不超过 `t` 灰阶的回放误差。每个通道精确区间仍记为 `I`。放宽后的区间 `J` 由观测灰阶从 `y − t` 到 `y + t` 的连续量化箱共同决定。一个短曝光光栅可行，当且仅当每个通道取值都在 `J`，且最多 `K` 个通道的取值在 `I` 外。对于一个分箱，优先在相交的 `I` 内选代表。只有分箱与 `I` 整体不相交时才占用一个通道额度。这个判定保留所有含有可行光栅的分箱，完整遍历后的逐通道误差界限与精确回放模式相同。

投影时先取各通道精确区间内最近的灰阶，再选至多 `K` 个改用放宽区间的通道，按平方误差下降量从大到小分配。该投影最小化对提案的像素平方改动。曝光比例、最大误差、通道数都是成像假设。输入照片本身无法核验这些参数适用于真实曝光。[原生命令参数](../native/tools/portrait_candidates/README.md#枚举两段曝光相容的短曝光画面)

### 短曝光画面存在整数位移时

允许短曝光裁图相对观测有整数位移。给定偏移后，在两张图相交的区域把每个传感器通道的精确区间和放宽区间平移到候选坐标。落在传感器外的候选像素没有采样约束，其通道范围为 `[0,255]`，也不消耗误差额度。重叠之外的传感器像素依然需要短、长曝光回放一致，可给两个曝光都取观测值。每个候选通道最多映到一个传感器通道，因此在选定偏移下，既有区间、通道预算、量化分页和最小平方投影仍然精确成立。

对有限个允许的位移，相容集合是各固定偏移集合的并集。穷尽每个偏移的分页可以覆盖该并集，重复代表允许存在。对每个偏移单独找最佳投影并取全局最小值，可精确保留任何至少能在一个偏移下回放的提案。输出的候选保持原有坐标。此保证适用于声明的整数裁切平移和两段曝光模型。真实采集中的亚像素运动、旋转、滚动快门和变形仍需独立验证，人像外观仍须用清晰参考及目视核对。[Windows 原生命令](../native/tools/portrait_candidates/README.md#枚举两段曝光相容的短曝光画面)
