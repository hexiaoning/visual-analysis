# 图像还原

## 人像

- 照片采用 PCFlow 官方 BFR 检查点，先对齐人脸，执行五步 Euler 推理，再以平滑蒙版将还原区域贴回原图。GFPGAN v1.4 同时产生待比较结果。两者都由 ONNX Runtime C++ 执行 FP32 模型。只有同时满足原图一致性、局部结构差异和碎纹限制时，才保留第二张。原图找不到可对齐的人脸时，抑制细密翻拍网纹后再试定位，模型区域以外保留原像素。包内权重及转换哈希固定。
- 视频使用可分发的 GFPGAN v1.4 基线，对同一轨迹的前帧残差执行光流对齐后做局部平滑。镜头变化、遮挡或匹配置信不足时重置历史。音频沿用原轨，视频按原 PTS 导出。
- DicFace 和 CodeFormer 的官方代码采用 S-Lab 许可，作为独立对比基线。发布包由构建脚本按固定模型清单生成。

## 车牌

- 第二轮两位评审均采用方案 C。中国车牌微调的 PP-OCRv6 加逐帧聚合作为识读基线，再比较 Geo-CTM。Geo-CTM 需要逐字符框和置信度，默认巴西字符规则要替换。识读验收看固定整牌误读率下的正确覆盖率。
- MF-LPR² 利用多帧光流和时空一致性过滤进行像素融合。论文在 RLPR 上报告 86.44% 的识读准确率，最好的多帧基线为 82.55%。本应用使用 ECC 与双向光流检查作为该类方法的轻量实现。
- LP-Diff 是扩散式多帧还原对照。作者在 MDLP 上报告准确率 30.5%，同一张表中 Real-ESRGAN 为 16.1%。这些结果来自各自的数据集。当前公开仓库提供训练和评价代码，正式集成还需取得可分发权重与许可。
- 类似的候选图像可以合并。分数由可测的图像一致性与边缘细节组成。未校准错误率的情况保留候选，供打开原始素材复核。识读模型和文字规则不参与当前像素填补。

[MF-LPR²](https://doi.org/10.1016/j.cviu.2025.104361) · [LP-Diff](https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html) · [Geo-CTM](https://github.com/ufukasia/geo-ctm) · [PP-OCRv6](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)

## 模型来源

原生构建使用已有的 FP32 ONNX 权重。SHA-256 固定在 scripts/build_native.ps1。模型制备与运行分开，应用从本机加载校验后的产物。

- GFPGAN 使用官方 v1.4 clean 权重，源码固定为 7552a7791caad982045a7bbe5634bbf1cd5c8679。转换产物为 gfpgan-v1.4-fp32.onnx。
- PCFlow 使用作者的 BFR/LCPL 权重，源码固定为 2028210a4252a088891744e35a6e47778c1d7266。TAESD3 解码器固定为 d58dcaccd2b36fcb7a6b9e93c1cc507acab5a778。转换产物为 pcflow-bfr-fp32.onnx。

发布前使用 scripts/test_native_operations.ps1 以默认和偏召回两轮复测三张本机照片，保存原图哈希、还原 PNG、分数与置信等级。

[GFPGAN](https://github.com/TencentARC/GFPGAN) · [PCFlow](https://github.com/aiimaginglab/PCFlow) · [DicFace](https://github.com/fudan-generative-vision/DicFace) · [CodeFormer](https://github.com/sczhou/CodeFormer) · [TAESD3](https://huggingface.co/madebyollin/taesd3)
