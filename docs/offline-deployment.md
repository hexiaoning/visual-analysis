# Visual Analysis 部署

`v2.11.0` 的正式交付物是 PyInstaller `onedir` 便携包。联网构建机负责下载和固定哈希校验；目标机不运行 `pip`、不下载模型，也不要求预装 Python、CUDA Toolkit、cuDNN 或 OpenVINO Toolkit。

## 构建机

要求 Windows 11 x64、64 位 Python 3.12 和互联网连接。在 PowerShell 中运行：

```powershell
.\scripts\build_offline_bundle.ps1 -Clean -AcceptResearchWeights
```

默认同时包含 `lvface-b` 和 `auraface`，因此 GUI 中可离线切换。若没有接受 LVFace 非商业研究权重许可，只构建可商用模型档：

```powershell
.\scripts\build_offline_bundle.ps1 -Clean -Profiles auraface
```

输出位于 `dist\FaceMatching-v2.11.0-windows-x64.zip`，旁边的 `.sha256` 文件校验整个压缩包；包内 `SHA256SUMS.txt` 校验每个文件。包中包含：

- Python 3.12 运行时和应用代码；
- PySide6、OpenCV、PyAV 与其 FFmpeg 编解码组件、NumPy；
- `onnxruntime-gpu` 及其 CUDA 12/cuDNN/cuBLAS 运行 DLL；
- OpenVINO Intel GPU 运行库；
- 构建时选择且经过 SHA-256 校验的 SCRFD/LVFace/AuraFace ONNX 模型；
- Chinese-CLIP ViT-B/16 FP32 图像/文本编码器、匹配的中文BERT词表及MIT许可；
- 车型核验子进程、离线自检脚本和第三方许可说明。

仅交付完整离线 ZIP，不生成增量包或 EXE 安装程序。全部功能打入同一应用，包含 SCRFD、人脸识别模型、RT-DETRv2 及离线地区参考数据。

## 目标机

目标机要求 Windows 11 x64，以及 NVIDIA RTX 或 Intel UHD/Iris/Arc GPU 和对应显卡驱动。

1. 解压完整 ZIP，双击 `启动综合分析.cmd`。
2. 在同一窗口内使用人脸检索、人车检测、车型识别、图像增强、以图搜索、语义搜索、人员库、标注库和设置。各模型按需初始化，初始化失败不会阻止其它页面使用。
3. 运维完整自检运行 `FaceMatching.exe --check-runtime --diagnose --report diagnostics.json`，验证全部人脸、人车及中文图文模型GPU推理；`verify_offline.ps1`也调用完整自检。

主应用仅处理本机媒体。车型识别在用户发起时启动独立的 VehicleCatalogueWorker.exe 查询授权接口。断网仍可做本地初判。见 [车型联网核验](vehicle-catalogues.md)。

人员数据库和录入照片默认写入 `%LOCALAPPDATA%\FaceMatching`，应用升级不会覆盖。需要把数据放到加密盘时，在启动前设置 `FACE_MATCHING_HOME`。

语义搜索索引保存在上述数据目录的 `data/semantic-index.db`，不改变原始素材。索引按源文件路径、大小、修改时间、文件标识、模型和抽帧设置区分；未完成的单个源文件索引会回滚，完整文件索引可复用。保留原始素材才能预览或播放结果。视频时间点来自解码器时间戳，无有效时间戳时用帧号/帧率计算；它是视频内相对时间，不是摄像头绝对时间。当前不自动读取画面中的日期水印。

## 交付检查

- 在一台禁网且从未安装 Python/CUDA Toolkit 的目标机上解压验证。
- `verify_offline.ps1` 发现的所有 profile 均返回 `"ok": true`。
- 使用目标摄像头的正负样本校准阈值，记录 FAR/TAR，而不是把余弦分数当概率。
- 对数据目录启用 BitLocker/EFS、最小权限、审计和保留期策略。

## 发布到百度网盘

默认以百度官方 Windows 命令行工具 `bdpan` 上传。扫码登录后，在构建机设置 `BAIDU_NETDISK_CREDENTIALS_FILE`，指向包含 `baidu_netdisk` 配置项的 JSON 文件。配置项 `executable`、`config_path` 和 `release_directory` 分别指向 CLI、CLI 管理的本机授权文件和网盘发布目录。账号授权由 `bdpan` 保存和续期。

打包会把校验通过的 ZIP 和 `.zip.sha256` 上传到「我的应用数据/bdpan/Visual-Analysis」下以版本和 ZIP 哈希前缀命名的子目录。核对网盘上的两个文件后，打包脚本删除上一版，再清理本次本地产物。上传或核对失败时会保留本地文件和上一版。明确只需本地包时，打包加 `-SkipBaiduUpload`。

如需从已经生成的 ZIP 重新上传，可运行：

```powershell
.\scripts\upload_bdpan_release.ps1 -Archive .\dist\FaceMatching-v2.11.0-windows-x64.zip -CredentialsFile $env:BAIDU_NETDISK_CREDENTIALS_FILE -PrunePrevious
```

手动重传命令会保留本地文件。开放平台接入方也可单独使用 `upload_baidu_release.py`，向可写的 `/apps/<应用名>` 目录上传。此方式依赖 `BAIDU_NETDISK_RELEASE_DIR` 和有效的 `BAIDU_NETDISK_ACCESS_TOKEN`。

[百度网盘官方 CLI](https://github.com/baidu-netdisk/bdpan-storage) · [百度网盘官方上传接口示例](https://github.com/baidu-netdisk/baidu-drive-sdk-go#上传与下载)
