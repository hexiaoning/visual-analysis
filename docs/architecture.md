# Visual Analysis 架构

## 进程与依赖

`VisualAnalysis.exe` 是 Win32 x64 主程序。ONNX Runtime C++ 执行本机模型，OpenCV C++ 做对齐、几何匹配、颜色证据和图像还原处理，FFmpeg C API 读写本机视频。SQLite 使用 Windows 自带的原生 DLL。

`VehicleCatalogueWorker.exe` 是车型核验进程，通过 WinHTTP 实时查询公开目录。请求只含候选车型文字和授权网关设置。车型图片和媒体在主程序中处理，报告包含资料链接和查询时间。

## 本机数据

默认应用目录为 `%LOCALAPPDATA%\FaceMatching`。可以通过 `FACE_MATCHING_HOME` 指定测试或数据目录。

| 路径 | 内容 |
| --- | --- |
| `data/faces.db` | SQLite 人员、分类、模型特征、识别事件和标注 |
| `data/face_images` | 用户录入的本机照片副本 |
| `data/annotations` | 人脸检索结果的本机帧和人脸截图 |
| `data/semantic-index.db` | 语义检索的本机索引 |
| `config.json` | 设置及人脸阈值 |
| `vehicle-providers.json` | 可选的授权网关地址 |

人员库兼容历史表结构和样本特征标识。模型与镜像增强模式必须匹配当前设置。可从已有照片重建特征。

本机媒体路径校验拒绝网络盘、远程路径和重定向链接。视频解码器限定文件协议与本机容器格式。
