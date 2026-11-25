# 工作目录约定

- 在 D:\visual-analysis\ 主工作目录开发、验证和打包。临时文件、安装包、依赖和缓存尽量放在 OneDrive 目录之外。

# 提交约定

- 每次改动通过必要检查后，立即提交并推送到 origin/main，无需等到打包。用户指定其他分支时按用户要求执行。

# GPU 加速约定

- 对模型推理、视频和图像处理等计算密集功能，用代表性输入比较 GPU 与 CPU 的实际耗时和结果。GPU 有明确性能收益时，实现 GPU 路径，并在兼容设备上默认优先使用。
- 验收时用运行记录确认关键算子确实在 GPU 上执行，并核对输出。CPU 路径用于设备不支持、GPU 执行失败或显式指定 CPU 的情况。

# 发布约定

- 每次打包生成完整的离线 ZIP 和 SHA-256 校验文件。ZIP 包含应用、运行库和模型。本机产物暂放 D:\visual-analysis\dist\。
- 如无特殊说明，包生成并通过 SHA-256 校验后，自动把 ZIP 和校验文件上传到已授权的百度网盘发布目录。
- 构建脚本依次打包、校验、上传、核对网盘、生成分享链接、清理。分享新版目录，让同一个链接包含 ZIP 和校验文件。
- 把分享链接和提取码记录到 %LOCALAPPDATA%\VisualAnalysis\latest-release.json，并把链接交给用户。发布记录放在本机。
- 新版上传且核对通过、分享链接已记录后，删除项目发布目录里的上一版，再删除本机本次生成的 ZIP、校验文件和打包展开目录。上传、核对、分享或记录失败时，保留上一版和本机产物。
- 上传时更新 %LOCALAPPDATA%\VisualAnalysis\release-progress.json，记录阶段、已确认字节数及基于上传进程读盘量的估算。
- 原生版使用 scripts/build_native.ps1 构建，使用 scripts/test_native_operations.ps1 生成验收摘要。通过后用 scripts/publish_native.ps1 完成打包、校验、上传、分享和清理。
