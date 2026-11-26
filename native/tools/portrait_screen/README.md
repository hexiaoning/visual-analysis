# 屏拍人像底图选择

这个 Windows 11 x64 离线研究工具同时输入原生屏拍像素、同一人脸的 512×512 原始裁切及既有首选。它先在原生像素上定位重复屏纹，再沿实测周期取样。它比较两张底图在已测屏纹频带和五官中频的幅度，并核对相对原图的低频偏差。最后保留一张。

## 构建

使用主应用采用的 MSVC x64、Windows SDK、OpenCV 4.13.0 的 Windows world 版及 nlohmann/json 3.12.0 单头文件。构建脚本将 EXE 和 OpenCV 运行库写入指定目录。运行时用 OpenCV 处理本地文件指定的图像。

```powershell
.\build.ps1 -OutputDirectory "$env:TEMP\portrait-screen" `
    -MsvcDirectory $env:MSVC_TOOLS -WindowsSdkDirectory $env:WINDOWS_SDK `
    -OpenCvDirectory $env:OPENCV_BUILD -JsonIncludeDirectory $env:JSON_HEADERS
```

## 执行

每行图像路径以 UTF-8 写在 `rows` 对象内。以下示例中，`native` 是保留拍摄像素的源图，`alignedRaw` 是同一张图按仿射矩阵得到的脸区，`previous` 是同一脸区的既有首选图。`nativeToAligned` 中的坐标为先 x 后 y。矩阵与原始裁切必须来自同一次人脸对齐。

```json
{
  "rows": {
    "example": {
      "native": "D:/cases/native.png",
      "alignedRaw": "D:/cases/raw-face.png",
      "previous": "D:/cases/previous.png",
      "nativeToAligned": [[2, 0, 0], [0, 2, 0]],
      "transport": {"mode": "direct"}
    }
  }
}
```

如原生输入是完整拍摄图的一块，可用 `delta`，在 `transport.matrix` 提供这块裁切到 512 网格的矩阵，去纹改变量在边缘渐弱后贴回原始裁切。已缩小解码的参考组用 `reduced`，在 `transport` 内另提供 `withinSource`、`full`、`reduced`、`resized` 和 `matrix`，让校正量依次沿完整采集坐标、缩小解码尺寸、面积重采样尺寸和最终对齐变换贴回。尺寸按宽、高给出。

```powershell
$exe = "$env:TEMP\portrait-screen\PortraitScreen.exe"
& $exe D:\cases\manifest.json D:\cases\screen-run-01
```

请为每次执行指定新的输出目录。每个样例保存周期去纹图 `phase.png` 和选中的 `selected.png`，成功处理完全部样例后才写入 `results.json`。其中记录选路、频峰、三类判断量及耗时。屏纹频带的下降可能伴随其他可见色带残留，选择结果仍需用有清晰参考的配对验收。

需要比较去屏纹后再去运动模糊时，可在清单的 `rows` 同级设置 `"writeNarrow": true`。每个样例会额外保存 `narrow.png` 并记录 `subtractMs`。该分支在线性 RGB 像素中估计实测频峰附近的本地正弦和余弦幅度，按相位一致性减少重复纹理。若下游模型需保留拍摄像素尺度，可使用 512×512 原生区域，将该区域同时填作 `native`、`alignedRaw` 和 `previous`，并使用单位矩阵及 `direct` 传递。该对照不参与 `selected.png` 的选择。

## 按拍摄像素重排已有候选

清单设置 `"rankAtCaptureResolution": true` 时，每个样例提供原生拍摄图、原生图到对齐脸区的仿射矩阵，以及该脸区的 512×512 候选。用 `previous` 给出目前展示的一个或两个候选名称。

```json
{
  "rankAtCaptureResolution": true,
  "rows": {
    "example": {
      "native": "D:/cases/screen.png",
      "nativeToAligned": [[2, 0, 0], [0, 2, 0]],
      "previous": ["primary"],
      "candidates": {
        "primary": "D:/cases/previous.png",
        "demoire": "D:/cases/demoire.png"
      }
    }
  }
}
```

工具把候选回放到拍摄像素网格，用五官之外的像素拟合亮度和首图的模糊程度，再比较眼、鼻、嘴的残差。原先展示一张时，误差下降超过 2% 且脸区带通幅度相对首图为 0.60–1.40 的备选可参与替换。单张输出写入 `selected.png`，既有双候选按原顺序写入 `selected.png` 和 `selected-2.png`。`results.json` 记录入选名称、模糊尺度及全部候选的误差和细节幅度比。该结果用于屏摄配对评测。

在清单同级设置 `"guardCaptureDetail": true` 可试用按屏纹强度放宽上限的规则。工具在拍摄像素上测量五官之外脸区的 Lab 色度带通均方根，写入 `chromaBandRms`，有效像素不足时该值为 `null`。色度低于 2.0 且未检出方形网格时，误差比原首图下降超过 20% 的候选可超过 1.40 倍细节上限。若另一张的误差降幅也超过 20%、回放误差至多比当前入选图高 10%、细节幅度至少再高 20%，则优先显示它。

`squareCarrier` 表示检测到方形网格并启用原细节上限。检测使用拍摄网格上的重复频峰，两支频率的模长相差至多 8%，夹角余弦的绝对值至多 0.12。这个判断也适用于灰度网格。单向条纹、长方网格和明显透视变形仍需其他检测方法。
