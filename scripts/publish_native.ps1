[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BuildDirectory,
    [Parameter(Mandatory)][string]$AcceptanceReport,
    [Parameter(Mandatory)][string]$CrtDirectory,
    [Parameter(Mandatory)][string]$OnnxRuntimeDirectory,
    [Parameter(Mandatory)][string]$OpenCvDirectory,
    [Parameter(Mandatory)][string]$FFmpegDirectory,
    [string]$Version = "2.13.0",
    [switch]$SkipBaiduUpload
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Invalid release version." }
$Project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).ProviderPath
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).ProviderPath
$Report = Get-Content -LiteralPath $AcceptanceReport -Raw -Encoding UTF8 | ConvertFrom-Json
$RequiredChecks = @("people", "annotations", "face_search", "visual_search", "semantic_search",
    "object_detection", "video_plate", "portrait_video", "vehicle_public", "vehicle_offline", "portrait_standard", "portrait_recall")
if (-not $Report.ok) { throw "Native acceptance is required before packaging." }
foreach ($Check in $RequiredChecks) {
    if (-not $Report.checks.PSObject.Properties[$Check] -or -not $Report.checks.$Check) {
        throw "Native acceptance did not pass $Check."
    }
}
foreach ($Pair in @(@("VisualAnalysis.exe", "exe_sha256"), @("VehicleCatalogueWorker.exe", "worker_sha256"))) {
    $File = Join-Path $BuildDirectory $Pair[0]
    $Field = $Pair[1]
    if (-not $Report.PSObject.Properties[$Field] -or (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -ine $Report.$Field) {
        throw "The build binary differs from the accepted binary. $($Pair[0])"
    }
}
if (@(git -C $Project ls-files "*.py").Count -gt 0) {
    throw "Complete the accepted source cutover before the native release."
}
$Commit = (git -C $Project rev-parse HEAD).Trim()
if ($LASTEXITCODE -or @(git -C $Project status --porcelain).Count -gt 0) {
    throw "Commit the native source cutover before creating the release."
}
$ReleaseRoot = Join-Path $Project "dist"
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
$ReleaseRoot = (Resolve-Path -LiteralPath $ReleaseRoot).ProviderPath
$ReleaseRootItem = Get-Item -LiteralPath $ReleaseRoot
if ($ReleaseRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint -or
    -not $ReleaseRoot.StartsWith($Project + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing a redirected release directory."
}
$Name = "FaceMatching-v$Version-windows-x64"
$Stage = Join-Path $ReleaseRoot $Name
$Archive = Join-Path $ReleaseRoot "$Name.zip"
$Checksum = "$Archive.sha256"
if ((Test-Path -LiteralPath $Stage) -or (Test-Path -LiteralPath $Archive) -or (Test-Path -LiteralPath $Checksum)) {
    throw "A local stage or package already exists for this version."
}

New-Item -ItemType Directory -Path $Stage | Out-Null
$Payload = @("VisualAnalysis.exe", "VehicleCatalogueWorker.exe", "onnxruntime.dll", "opencv_world4130.dll",
    "avcodec-62.dll", "avformat-62.dll", "avutil-60.dll", "swresample-6.dll", "swscale-9.dll")
foreach ($File in $Payload) {
    $RuntimeSource = Join-Path $BuildDirectory $File
    if ($File.EndsWith(".dll")) {
        $RuntimeHash = $Report.native_runtime.PSObject.Properties[$File]
        if (-not $RuntimeHash -or (Get-FileHash -LiteralPath $RuntimeSource -Algorithm SHA256).Hash -ine $RuntimeHash.Value) {
            throw "Native runtime DLL differs from acceptance. $File"
        }
    }
    Copy-Item -LiteralPath $RuntimeSource -Destination $Stage
}
$CrtFiles = @("concrt140.dll", "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")
foreach ($File in $CrtFiles) {
    Copy-Item -LiteralPath (Join-Path $CrtDirectory $File) -Destination $Stage
}
$ModelStage = Join-Path $Stage "models"
New-Item -ItemType Directory -Path $ModelStage | Out-Null
$ModelNames = @("scrfd_10g_bnkps.onnx", "glintr100.onnx", "rtdetr_v2_r50vd.onnx", "chinese_clip_text.onnx",
    "chinese_clip_vision.onnx", "chinese_clip_vocab.txt", "pcflow-bfr-fp32.onnx", "gfpgan-v1.4-fp32.onnx")
foreach ($File in $ModelNames) {
    $ModelSource = Join-Path $BuildDirectory "models\$File"
    $AcceptedHash = $Report.models.PSObject.Properties[$File]
    if (-not $AcceptedHash -or (Get-FileHash -LiteralPath $ModelSource -Algorithm SHA256).Hash -ine $AcceptedHash.Value) {
        throw "The staged weight differs from acceptance. $File"
    }
    Copy-Item -LiteralPath $ModelSource -Destination $ModelStage
}

$Licenses = Join-Path $Stage "licenses"
New-Item -ItemType Directory -Path $Licenses | Out-Null
Copy-Item -Path (Join-Path $Project "packaging\licenses\*") -Destination $Licenses
Copy-Item -LiteralPath (Join-Path $OnnxRuntimeDirectory "LICENSE") -Destination (Join-Path $Licenses "ONNX-Runtime.txt")
Copy-Item -LiteralPath (Join-Path $OnnxRuntimeDirectory "ThirdPartyNotices.txt") -Destination (Join-Path $Licenses "ONNX-Runtime-ThirdParty.txt")
Copy-Item -LiteralPath (Join-Path $OpenCvDirectory "LICENSE") -Destination (Join-Path $Licenses "OpenCV.txt")
Copy-Item -LiteralPath (Join-Path $FFmpegDirectory "LICENSE.txt") -Destination (Join-Path $Licenses "FFmpeg-LGPL.txt")
Copy-Item -LiteralPath (Join-Path $Project "README.md") -Destination (Join-Path $Stage "使用说明.md")
Copy-Item -LiteralPath (Join-Path $Project "THIRD_PARTY_NOTICES.md") -Destination $Stage
$DocStage = Join-Path $Stage "docs"
New-Item -ItemType Directory -Path $DocStage | Out-Null
foreach ($File in @("architecture.md", "offline-deployment.md", "restoration.md", "vehicle-catalogues.md")) {
    Copy-Item -LiteralPath (Join-Path $Project "docs\$File") -Destination $DocStage
}
$NativeDocStage = Join-Path $Stage "native"
New-Item -ItemType Directory -Path $NativeDocStage | Out-Null
Copy-Item -LiteralPath (Join-Path $Project "native\README.md") -Destination $NativeDocStage
$CaseDocStage = Join-Path $Stage "cases"
New-Item -ItemType Directory -Path $CaseDocStage | Out-Null
Copy-Item -LiteralPath (Join-Path $Project "cases\README.md") -Destination $CaseDocStage
Copy-Item -LiteralPath $AcceptanceReport -Destination (Join-Path $Stage "验收摘要.json")

$Manifest = [ordered]@{
    application = "Visual Analysis"; version = $Version; source_commit = $Commit
    runtime = "Win32 x64"; release_created_at = [DateTime]::UtcNow.ToString("o")
}
$Manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Stage "release-info.json") -Encoding UTF8
$Files = @(Get-ChildItem -LiteralPath $Stage -Recurse -File | Sort-Object FullName)
$Inventory = foreach ($File in $Files) {
    [ordered]@{ path = $File.FullName.Substring($Stage.Length + 1).Replace('\', '/')
        size = $File.Length; sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
ConvertTo-Json -InputObject @($Inventory) -Depth 5 |
    Set-Content -LiteralPath (Join-Path $Stage "contents.json") -Encoding UTF8

Write-Host "Creating and verifying native ZIP..." -ForegroundColor Cyan
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($Stage, $Archive, [IO.Compression.CompressionLevel]::Fastest, $false)
$Zip = [IO.Compression.ZipFile]::OpenRead($Archive)
try {
    $Actual = @($Zip.Entries | Where-Object { -not $_.FullName.EndsWith('/') })
    if ($Actual.Count -ne ($Inventory.Count + 1)) { throw "ZIP entry count changed from the staged inventory." }
    foreach ($File in $Inventory) {
        $Entry = $Zip.GetEntry($File.path)
        if (-not $Entry -or $Entry.Length -ne $File.size) { throw "Missing ZIP payload $($File.path)." }
        $Stream = $Entry.Open()
        try {
            if ((Get-FileHash -Algorithm SHA256 -InputStream $Stream).Hash -ine $File.sha256) {
                throw "ZIP payload digest mismatch for $($File.path)."
            }
        }
        finally { $Stream.Dispose() }
    }
}
finally { $Zip.Dispose() }
$Hash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$Hash  $Name.zip" | Set-Content -LiteralPath $Checksum -Encoding ascii
Write-Host "Verified native release ZIP at $Archive"
if ($SkipBaiduUpload) { return }

$CredentialsFile = $env:BAIDU_NETDISK_CREDENTIALS_FILE
if (-not $CredentialsFile) { $CredentialsFile = [Environment]::GetEnvironmentVariable("BAIDU_NETDISK_CREDENTIALS_FILE", "User") }
if (-not $CredentialsFile) { throw "The configured Baidu Netdisk credentials path is missing." }
& (Join-Path $PSScriptRoot "upload_bdpan_release.ps1") -Archive $Archive -CredentialsFile $CredentialsFile -PrunePrevious
$Latest = Get-Content -LiteralPath (Join-Path $env:LOCALAPPDATA "VisualAnalysis\latest-release.json") -Raw |
    ConvertFrom-Json
if ($Latest.version -ne "v$Version" -or $Latest.sha256 -ine $Hash -or -not $Latest.share_url) {
    throw "The Baidu share record does not match the verified native ZIP."
}
foreach ($Target in @($Stage, $Archive, $Checksum)) {
    $Resolved = Get-Item -LiteralPath $Target
    if ($Resolved.Attributes -band [IO.FileAttributes]::ReparsePoint -or
        (Split-Path -Parent $Resolved.FullName) -ne $ReleaseRoot) {
        throw "Refusing redirected or misplaced release output."
    }
}
Remove-Item -LiteralPath $Archive, $Checksum -Force
Remove-Item -LiteralPath $Stage -Recurse -Force
$Latest
