# 图像还原

## 人像

- 照片采用 PCFlow 官方 BFR 检查点，先对齐人脸，执行五步 Euler 推理，再以平滑蒙版将还原区域贴回原图。GFPGAN v1.4 同时产生待比较结果。两者都由 ONNX Runtime C++ 执行 FP32 模型。只有同时满足原图一致性、局部结构差异和碎纹限制时，才保留第二张。原图找不到可对齐的人脸时，抑制细密翻拍网纹后再试定位，模型区域以外保留原像素。包内权重及转换哈希固定。
- 视频使用可分发的 GFPGAN v1.4 基线，对同一轨迹的前帧残差执行光流对齐后做局部平滑。镜头变化、遮挡或匹配置信不足时重置历史。音频沿用原轨，视频按原 PTS 导出。
- DicFace 和 CodeFormer 的官方代码采用 S-Lab 许可，作为独立对比基线。发布包由构建脚本按固定模型清单生成。

## 车牌

- HyperLPR3 以牌照四点定位后做透视变换，再用牌照专用 CTC 模型识读。高分辨率图像可补做重叠区域搜索。视频按运动关联车牌，来自不同帧的一致识读可用于确认。同一帧的原始和增强图像属于同一份证据。
- 保存原始车牌裁图供复核。透视校正、去拖影和 ECC 与双向光流融合产生候选图像。识读先看原始帧支持，增强图像的号码单独显示作核对参考。低置信、车牌过小或竞争读数转为「待核对」。
- 识读效果通过整牌完全匹配率、误读率与未读出率衡量。视频以轨迹为单位，实际监控画面的识读受车牌像素尺寸、曝光、角度和拖影影响。

[HyperLPR3](https://github.com/szad670401/HyperLPR/tree/58ad4ef736fec5289e8c8cb842e066e70c4672c3) · [CCPD](https://github.com/detectRecog/CCPD) · [MF-LPR²](https://doi.org/10.1016/j.cviu.2025.104361) · [PP-OCRv6](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)

## 模型来源

原生构建使用已有的 FP32 ONNX 权重。SHA-256 固定在 scripts/build_native.ps1。模型制备与运行分开，应用从本机加载校验后的产物。

- GFPGAN 使用官方 v1.4 clean 权重，源码固定为 7552a7791caad982045a7bbe5634bbf1cd5c8679。转换产物为 gfpgan-v1.4-fp32.onnx。
- PCFlow 使用作者的 BFR/LCPL 权重，源码固定为 2028210a4252a088891744e35a6e47778c1d7266。TAESD3 解码器固定为 d58dcaccd2b36fcb7a6b9e93c1cc507acab5a778。转换产物为 pcflow-bfr-fp32.onnx。

发布前使用 scripts/test_native_operations.ps1 以默认和偏召回两轮复测三张本机照片，保存原图哈希、还原 PNG、分数与置信等级。

[GFPGAN](https://github.com/TencentARC/GFPGAN) · [PCFlow](https://github.com/aiimaginglab/PCFlow) · [DicFace](https://github.com/fudan-generative-vision/DicFace) · [CodeFormer](https://github.com/sczhou/CodeFormer) · [TAESD3](https://huggingface.co/madebyollin/taesd3)
