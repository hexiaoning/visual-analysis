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
