# 图像还原案例库

本机的三个翻拍案例保存在 inputs/，原件校验清单为 index.local.json。把运行结果写入 outputs/。三个路径皆为 Git 忽略项。应用发布包只包含软件、运行库、模型和说明文档。

回归覆盖有屏幕网纹、偏暗的画面和低分辨率侧脸。使用同一个离线模型入口处理全部图片。每个案例至少产出一个实际还原结果。分数用于排序，置信等级表示候选受到原图及另一模型支持的程度。

给 `scripts/build_offline_bundle.ps1` 传入 `-RestorationCaseDirectory D:\visual-analysis\cases\inputs`，会在生成 ZIP 前用打包后的 EXE 运行案例。报告与还原图会保存在本机 `cases/outputs/`。任一案例无结果时构建会报错。
