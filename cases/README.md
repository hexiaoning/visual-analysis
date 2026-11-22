# 图像还原案例库

本机的三个翻拍案例保存在 inputs/，原件校验清单为 index.local.json。把运行结果写入 outputs/。三个路径皆为 Git 忽略项。应用发布包只包含软件、运行库、模型和说明文档。

回归覆盖有屏幕网纹、偏暗的画面和低分辨率侧脸。使用同一个离线模型入口处理全部图片。每个案例至少产出一个实际还原结果。分数用于排序，置信等级表示候选受到原图及另一模型支持的程度。

用 `scripts/test_native_operations.ps1` 的 `-CaseDirectory D:\visual-analysis\cases\inputs` 参数，以实际准备发布的 EXE 跑默认和偏召回两轮。也可使用 `VisualAnalysis.exe --check-restoration-cases` 检查指定目录，调用示例见 [原生构建说明](../native/README.md)。案例未通过时验收脚本会报错。
