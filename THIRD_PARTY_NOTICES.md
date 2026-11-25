# 第三方组件与权重

| 组件 | 上游与用途 |
| --- | --- |
| ONNX Runtime DirectML 1.24.4 | Microsoft MIT，随包提供 DirectML 和 CPU 执行后端。 |
| DirectML 1.15.4 | Microsoft DirectX 12 机器学习运行库，随包附许可证及第三方声明。 |
| OpenVINO 2026.4.1 | Apache-2.0，Intel GPU 的中文图文模型推理。 |
| OpenCV 4.13.0 | Apache-2.0，图像解码、几何对齐、光流和局部特征。 |
| FFmpeg 8.1 LGPL shared | FFmpeg C API，动态库版。构建版本 n8.1.3-39-g660b5aa7db-20261121。[FFmpeg 源码](https://github.com/FFmpeg/FFmpeg/tree/660b5aa7db) · [Windows 构建配方](https://github.com/FFmpeg-Builds/FFmpeg-Builds) |
| nlohmann/json 3.12.0 | MIT，构建时使用单头文件。[上游发行版](https://github.com/nlohmann/json/releases/tag/v3.12.0) |
| RT-DETRv2-R50 | Apache-2.0，人和车辆检测。ONNX 导出固定为 1699ea019e5570bb7283d80ee9a9e42d6245436f。 |
| Chinese-CLIP ViT-B/16 | MIT，中文图文特征。双编码器为 felixdu 固定提交 47080d16c631d8416d2e6b155c59f8fd2c322e98，词表为 Xenova 提交 f26904860903e70e050b8f48255e5f48401816e9。 |
| PCFlow BFR | MIT，使用作者的 BFR/LCPL 检查点和 TAESD3 解码器。[PCFlow](https://github.com/aiimaginglab/PCFlow) · [TAESD3](https://huggingface.co/madebyollin/taesd3) |
| HyperLPR3 | Apache-2.0，使用官方固定提交 58ad4ef736fec5289e8c8cb842e066e70c4672c3 的 640 车牌检测和字符识读 ONNX 权重与词表。[上游资源](https://github.com/szad670401/HyperLPR/tree/58ad4ef736fec5289e8c8cb842e066e70c4672c3/resource/models/onnx) |
| GFPGAN v1.4 | Apache-2.0，使用官方 clean 权重导出的 ONNX。 |
| SCRFD 与 AuraFace R100 | 人脸关键点、特征模型。权重来自固定 AuraFace-v1 发行版本。[模型卡](https://huggingface.co/fal/AuraFace-v1) |

发布包包含依赖及模型的相关许可证文本。随包的 CRT 来自 Windows x64 Visual C++ 运行库。
