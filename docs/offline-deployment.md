# Windows 11 离线部署

1. 将 ZIP 和同目录的 `.sha256` 下载到本机。
2. 使用 `Get-FileHash -Algorithm SHA256` 校验 ZIP。
3. 完整解压到本机磁盘，运行 `VisualAnalysis.exe`。

图像还原、人脸检索、人车检测、以图搜索、语义搜索、人员库和标注库使用随包模型与 DLL。车型识别按需启动独立联网进程，找不到可用线上来源时显示「本地初判，未联网核验」。

## 构建机

`scripts/build_native.ps1` 使用 MSVC x64、Windows SDK、ONNX Runtime、OpenCV、FFmpeg C SDK 和 nlohmann/json 头文件。向 `-ModelDirectory` 传入离线权重和词表目录。构建脚本校验模型 SHA-256。

`scripts/test_native_operations.ps1` 以实际 EXE 跑人员录入、照片比对、标注、视频、检测、搜索、车型公开资料和三例图片还原。用于验收的原图、结果与索引放在构建机，发布包只包含程序、运行库、模型、说明和不含案例内容的验收摘要。

`scripts/publish_native.ps1` 读取验收摘要，校验 EXE、工作进程、DLL 与权重，生成并逐文件验证 ZIP。上传到百度网盘后会核对远端文件、生成分享链接，再按项目约定清理。
