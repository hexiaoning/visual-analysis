# 图像还原

## 人像

- 照片采用 PCFlow 官方 BFR 检查点，先对齐人脸，执行五步 Euler 推理，再以平滑蒙版将还原区域贴回原图。GFPGAN v1.4 提供第二个同源结果。两者都在 FP32 ONNX 中执行。包内权重及转换哈希固定。
- 视频使用可分发的 GFPGAN v1.4 基线，对同一轨迹的前帧残差执行光流对齐后做局部平滑。镜头变化、遮挡或匹配置信不足时重置历史。音频沿用原轨，视频按原 PTS 导出。
- DicFace 和 CodeFormer 的官方代码采用 S-Lab 许可，作为独立对比基线。发布包由构建脚本按固定模型清单生成。

## 车牌

- 第二轮两位评审均采用方案 C。中国车牌微调的 PP-OCRv6 加逐帧聚合作为识读基线，再比较 Geo-CTM。Geo-CTM 需要逐字符框和置信度，默认巴西字符规则要替换。识读验收看固定整牌误读率下的正确覆盖率。
- MF-LPR² 利用多帧光流和时空一致性过滤进行像素融合。论文在 RLPR 上报告 86.44% 的识读准确率，最好的多帧基线为 82.55%。本应用使用 ECC 与双向光流检查作为该类方法的轻量实现。
- LP-Diff 是扩散式多帧还原对照。作者在 MDLP 上报告准确率 30.5%，同一张表中 Real-ESRGAN 为 16.1%。这些结果来自各自的数据集。当前公开仓库提供训练和评价代码，正式集成还需取得可分发权重与许可。
- 类似的候选图像可以合并。分数由可测的图像一致性与边缘细节组成。未校准错误率的情况保留候选，供打开原始素材复核。识读模型和文字规则不参与当前像素填补。

[MF-LPR²](https://doi.org/10.1016/j.cviu.2025.104361) · [LP-Diff](https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html) · [Geo-CTM](https://github.com/ufukasia/geo-ctm) · [PP-OCRv6](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)

## 模型转换

使用 Python 3.12、PyTorch 2.5.1 CPU、torchvision 0.20.1、ONNX 1.23.2 和 Safetensors。转换与运行隔离。GFPGAN 需要 BasicSR 1.4.2 的源码可导入。图像端转换脚本指定固定文件名、输入归一化和 FP32 数值对照。

- GFPGAN 源代码固定为提交 7552a7791caad982045a7bbe5634bbf1cd5c8679，预训练源文件为 GFPGANv1.4.pth。运行 scripts/export_gfpgan_onnx.py <源码> <权重> <模型目录>/gfpgan-v1.4-fp32.onnx。
- PCFlow 源码固定为提交 2028210a4252a088891744e35a6e47778c1d7266，使用作者公开的 BFR/lcpl/model_500.pth 和 BFR/lcpl/lq_encoder_500.pth，解码器为 TAESD3 固定提交 d58dcaccd2b36fcb7a6b9e93c1cc507acab5a778。把三个权重分别放为 pcflow-model-500.pth、pcflow-lq-encoder.pth、taesd3.safetensors，然后运行 scripts/export_pcflow_onnx.py <源码> <权重目录> <模型目录>/pcflow-bfr-fp32.onnx。

构建时用 -ModelDirectory 指向含上述两个 ONNX 的模型目录。离线启动自检验证权重哈希并实际运行两张还原图。开发或复现实测时用 scripts/benchmark_face_restoration.py 对同一份对齐素材运行四个模型。保存每个输入的 SHA-256 及视频帧时间点。

[GFPGAN](https://github.com/TencentARC/GFPGAN) · [PCFlow](https://github.com/aiimaginglab/PCFlow) · [DicFace](https://github.com/fudan-generative-vision/DicFace) · [CodeFormer](https://github.com/sczhou/CodeFormer) · [TAESD3](https://huggingface.co/madebyollin/taesd3)
