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
