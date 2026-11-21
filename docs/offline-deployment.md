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

构建完成并生成校验文件后，可以由构建机直接调用百度网盘上传接口。先在百度网盘开放平台完成应用接入和账号 OAuth 授权，再在构建进程中配置：

- `BAIDU_NETDISK_RELEASE_DIR`，授权应用下已有的网盘目录，例如 `/apps/实际应用名/Visual-Analysis`。网盘客户端中对应「我的应用数据/实际应用名/Visual-Analysis」。
- `BAIDU_NETDISK_ACCESS_TOKEN`，上述账号的有效访问令牌。通过本机或发布环境的凭据存储注入，令牌过期后更新。项目不保管令牌或刷新令牌。

配置目录后，常规打包命令自动上传 ZIP 和 `.zip.sha256`。上传流程重新校验 ZIP，以 4 MiB 分片传输，单文件上限 4 GiB。上传失败会保留本地压缩包和校验文件，命令返回失败。若要上传已生成的包，可单独运行：

```powershell
python .\scripts\upload_baidu_release.py .\dist\FaceMatching-v2.11.0-windows-x64.zip
```

如百度返回同名冲突，检查是否已有完整的 ZIP 和校验文件，或选择另一个授权目录重试。上传脚本使用不重命名策略，避免自动覆盖。目标机无需配置百度网盘。

[百度网盘官方上传接口示例](https://github.com/baidu-netdisk/baidu-drive-sdk-go#上传与下载) · [百度网盘开放平台上传限制](https://help.baidu.com/question?class=800&id=1002121&prod_id=257)
