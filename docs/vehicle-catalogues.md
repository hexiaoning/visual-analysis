# 车型联网核验

在 Visual Analysis 的「车型识别」页选择本机图片或视频，再选择本地车型参考库。也可以填写人工候选品牌和车系。当前版本用 RT-DETRv2 提取车辆区域，用离线 Chinese-CLIP 对参考图排序。没有参考库时，可从 18 个内置车系形成初判候选，覆盖和准确性取决于具体素材。图片选择占画面最大的车辆，视频对第一个可跟踪的车辆采样。相似度供复核使用。车型识别支持 GPU 和 CPU，并在结果中显示推理后端。

每次开始核验，应用都让独立的 VehicleCatalogueWorker.exe 并行查询工信部和懂车帝的授权数据源。请求只带车型候选字段。原始媒体、车牌和图像向量留在本机。响应中的参考图由本机编码和比较。结果记录两方的资料编号或链接、查询时间和状态，并可导出 JSON。

获取两方一致资料时界面提示双源一致。一方有可用的相符资料时显示「部分联网核验」。两方均不可用时显示「本地初判，未联网核验」，保留本地候选。

## 本地车型参考库

使用有权使用的车辆图片，把目录下的图片相对路径写进 JSON。目录不得使用网络盘或重定向链接。可在界面选择 JSON，也可放在应用数据目录下，文件名为 vehicle-catalogue.json。

~~~json
{
  "schema_version": 1,
  "vehicles": [
    {
      "brand": "示例品牌",
      "series": "示例车系",
      "model": "示例车型",
      "year": "2025",
      "trim": "示例配置",
      "catalogue_id": "authorised-catalogue-key",
      "images": ["images/front.jpg", "images/rear.jpg"]
    }
  ]
}
~~~

## 授权网关

应用支持给两家数据源各配置一个授权 HTTPS 网关。这里描述的是本应用的网关协议，供应方的实际接口须按各自授权文档适配。当前程序可使用准备好的网关；官方公开查询页面可以人工查看。

在应用数据目录（默认 %LOCALAPPDATA%\FaceMatching）创建 vehicle-providers.json。以下两个地址仅用于展示结构。

~~~json
{
  "miit": {"url": "https://gateway.example.invalid/catalogue/miit"},
  "dongchedi": {"url": "https://gateway.example.invalid/catalogue/dongchedi"}
}
~~~

从启动环境读取 VISUAL_VEHICLE_MIIT_TOKEN 和 VISUAL_VEHICLE_DONGCHEDI_TOKEN。凭据通过 Authorization Bearer 发送到对应地址，通过环境变量单独管理。程序拒绝 HTTPS 重定向。

两个连接器使用 POST JSON。请求结构示例：

~~~json
{
  "schema_version": 1,
  "source": "miit",
  "candidates": [
    {
      "brand": "示例品牌",
      "series": "示例车系",
      "model": "示例车型",
      "year": "",
      "trim": "",
      "catalogue_id": "authorised-catalogue-key"
    }
  ]
}
~~~

网关应为当前任务实时查对应授权来源，并将响应映射到以下结构。请为同一次任务避免复用旧查询结果。records 为空表示查询完成但未找到资料。车系和来源链接或编号必填。参数值使用文本。可在 image_jpeg_base64 中提供有权使用的 JPEG 参考图，最多两张。可选的 catalogue_id 用于两边命名不同但已有授权映射的车型。

~~~json
{
  "records": [
    {
      "brand": "示例品牌",
      "series": "示例车系",
      "model": "示例车型",
      "year": "2025",
      "trim": "示例配置",
      "catalogue_id": "authorised-catalogue-key",
      "record_id": "official-record-number",
      "source_url": "https://source.example.invalid/record",
      "parameters": {"轴距": "2800 mm"},
      "image_jpeg_base64": []
    }
  ]
}
~~~

公开入口：[工信部公告信息查询系统](https://www.miit-eidc.org.cn/art/2025/9/30/art_1657_11737.html) · [懂车帝企业开放平台](https://open.dongchedi.com/)。
