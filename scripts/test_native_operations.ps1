[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string]$FaceImage,
    [Parameter(Mandatory)][string]$VehicleImage,
    [Parameter(Mandatory)][string]$FaceVideo,
    [Parameter(Mandatory)][string]$CaseDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$FFprobe,
    [string]$FFmpeg = (Join-Path (Split-Path -Parent $FFprobe) "ffmpeg.exe"),
    [string]$ExpectedVehicleSeries = "海鸥"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Executable = (Resolve-Path -LiteralPath $Executable).ProviderPath
New-Item -ItemType Directory -Path $OutputDirectory -ErrorAction Stop | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).ProviderPath
$PreviousHome = $env:FACE_MATCHING_HOME
$PreviousProfile = $env:VISUAL_ANALYSIS_GPU_PROFILE
$PreviousBackend = $env:VISUAL_ANALYSIS_BACKEND
$env:VISUAL_ANALYSIS_GPU_PROFILE = Join-Path $OutputDirectory "inference-profiles"
$env:VISUAL_ANALYSIS_BACKEND = $null
New-Item -ItemType Directory -Path $env:VISUAL_ANALYSIS_GPU_PROFILE | Out-Null
$env:FACE_MATCHING_HOME = Join-Path $OutputDirectory "app-home"
New-Item -ItemType Directory -Path $env:FACE_MATCHING_HOME | Out-Null
$Checks = [ordered]@{}

function Quote-Argument([string]$Value) { '"{0}"' -f $Value }

function Invoke-NativeOperation([string]$Name, [hashtable]$Request, [string]$Id = $Name, [string]$Program = $Executable) {
    $InputFile = Join-Path $OutputDirectory "$Id-request.json"
    $Report = Join-Path $OutputDirectory "$Id-report.json"
    $Request | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $InputFile -Encoding UTF8
    $Backend = $env:VISUAL_ANALYSIS_BACKEND
    if ($Id -notin "face.compare", "region-trigger", "semantic-first") { $env:VISUAL_ANALYSIS_BACKEND = "cpu" }
    try {
        $Run = Start-Process -FilePath $Program -WindowStyle Hidden -Wait -PassThru -ArgumentList @(
            "--native-operation", $Name, "--request", (Quote-Argument $InputFile), "--report", (Quote-Argument $Report)
        )
    } finally { $env:VISUAL_ANALYSIS_BACKEND = $Backend }
    if (-not (Test-Path -LiteralPath $Report -PathType Leaf)) { throw "Missing $Name report, exit $($Run.ExitCode)." }
    $Value = Get-Content -LiteralPath $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Run.ExitCode -ne 0 -or -not $Value.ok) { throw "$Name failed. $($Value.error)" }
    $Value.result
}

function Require([bool]$Predicate, [string]$Message) {
    if (-not $Predicate) { throw $Message }
}

try {
    $People = @(Invoke-NativeOperation "person.add" @{
        name = "Native acceptance fixture"; id_card = "fixture-" + [Guid]::NewGuid().ToString("N"); photos = @($FaceImage)
    })
    Require ($People.Count -eq 1 -and $People[0].photo_count -eq 1) "SQLite enrollment output was incomplete."
    $Person = $People[0]
    $Similarity = Invoke-NativeOperation "face.compare" @{ input = $FaceImage; reference = $FaceImage }
    $Gallery = @(Invoke-NativeOperation "face.gallery" @{ input = $FaceImage; library_id = "default" })
    Require ($Similarity.similarity -gt .99 -and $Gallery.Count -eq 1 -and $Gallery[0].person_id -eq $Person.id) "Face similarity or enrolled gallery result failed."
    $Checks.people = $true
    $Scanned = Invoke-NativeOperation "face.scan" @{ source = $FaceImage }
    Require (@($Scanned.hits).Count -gt 0 -and $Scanned.hits[0].accepted) "Gallery identification returned no accepted match."
    $Checks.face_search = $true
    $TrackFixture = Join-Path $OutputDirectory "face-track.mkv"
    & $FFmpeg -v error -loop 1 -framerate 5 -i $FaceImage -frames:v 6 -c:v ffv1 $TrackFixture
    Require ($LASTEXITCODE -eq 0) "Could not generate the face track fixture."
    $Track = Invoke-NativeOperation "face.scan" @{ source = $TrackFixture } "face-track"
    Require (@($Track.hits).Count -eq 1 -and $Track.hits[0].accepted -and
        $Track.hits[0].observations -ge 4 -and $Track.hits[0].candidate.person_id -eq $Person.id) "Fused video track did not upgrade to a confirmed match."
    $RecurringFixture = Join-Path $OutputDirectory "face-track-recurring.mkv"
    & $FFmpeg -v error -loop 1 -framerate 5 -i $FaceImage -vf 'drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable=between(n\,4\,11)' -frames:v 16 -c:v ffv1 $RecurringFixture
    Require ($LASTEXITCODE -eq 0) "Could not generate the recurring face fixture."
    $Recurring = Invoke-NativeOperation "face.scan" @{ source = $RecurringFixture } "face-recurring"
    Require (@($Recurring.failures).Count -eq 0 -and @($Recurring.hits).Count -eq 1 -and
        $Recurring.hits[0].accepted -and $Recurring.hits[0].candidate.person_id -eq $Person.id -and
        @(Get-ChildItem -LiteralPath (Split-Path -Parent $Recurring.hits[0].face_path) -File).Count -eq 2) "A confirmed person returning in the same video left duplicate candidates or snapshots."
    $Checks.face_video_tracking = $true
    $Mixed = Invoke-NativeOperation "face.scan" @{
        sources = @($FaceImage, $TrackFixture); library_id = "default"
    } "face-mixed-sources"
    Require (@($Mixed.failures).Count -eq 0 -and @($Mixed.hits).Count -eq 2 -and
        @($Mixed.hits | Where-Object accepted | Select-Object -ExpandProperty source -Unique).Count -eq 2) "Mixed image and video sources did not both match in one batch."
    $Checks.face_mixed_sources = $true
    $Libraries = @(Invoke-NativeOperation "library.add" @{ name = "Native acceptance review library" })
    $Review = $Libraries | Where-Object name -eq "Native acceptance review library"
    $Members = @(Invoke-NativeOperation "person.add" @{
        name = "Native acceptance review fixture"; id_card = "fixture-" + [Guid]::NewGuid().ToString("N")
        photos = @($FaceImage); library_ids = @($Review.id)
    } "person-add-review")
    $Reviewer = $Members | Where-Object name -eq "Native acceptance review fixture"
    $Combined = @(Invoke-NativeOperation "face.gallery" @{
        input = $FaceImage; library_ids = @("default", $Review.id, $Review.id)
    } "face-gallery-multiple-libraries")
    Require ($Combined.Count -eq 2 -and $Person.id -in $Combined.person_id -and $Reviewer.id -in $Combined.person_id) "Gallery did not search the union of selected libraries."
    $Restricted = Invoke-NativeOperation "face.scan" @{
        source = $TrackFixture; library_ids = @($Review.id)
    } "face-track-selected-library"
    Require (@($Restricted.hits).Count -eq 1 -and $Restricted.hits[0].accepted -and
        $Restricted.hits[0].candidate.person_id -eq $Reviewer.id) "Video search matched a candidate outside the selected library."
    $EmptyScope = @(Invoke-NativeOperation "face.gallery" @{
        input = $FaceImage; library_ids = @()
    } "face-gallery-empty-scope")
    Require ($EmptyScope.Count -eq 0) "An empty library selection searched the whole gallery."
    $Checks.face_library_selection = $true
    $Hit = $Scanned.hits[0]
    $Mark = Invoke-NativeOperation "annotation.write" @{
        session_id = $Hit.session_id; source = $Hit.source; track_id = $Hit.track_id; frame_index = $Hit.frame_index
        action = "associate"; person_id = $Person.id; snapshot_path = $Hit.snapshot_path; face_path = $Hit.face_path; bbox = $Hit.bbox
    }
    $Marks = @(Invoke-NativeOperation "annotations" @{})
    $Export = Invoke-NativeOperation "annotation.export" @{ output = (Join-Path $OutputDirectory "annotation-export") }
    Require ($Mark.recorded -and $Marks.Count -eq 1 -and $Export.count -eq 1 -and
        (Test-Path -LiteralPath $Export.output -PathType Leaf)) "Annotation round trip or ZIP export failed."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $AnnotationZip = [IO.Compression.ZipFile]::OpenRead($Export.output)
    try {
        $Entries = @($AnnotationZip.Entries | ForEach-Object { $_.FullName -replace '^\./', '' })
        $ManifestEntry = $AnnotationZip.Entries | Where-Object { $_.Name -eq "annotations.json" } | Select-Object -First 1
        Require ($null -ne $ManifestEntry) "Annotation ZIP manifest is missing."
        $Reader = [IO.StreamReader]::new($ManifestEntry.Open())
        try { $Manifest = $Reader.ReadToEnd() | ConvertFrom-Json }
        finally { $Reader.Dispose() }
        $Item = $Manifest.annotations[0]
        Require ($Item.media.path -cin $Entries -and $Item.snapshot_path -cin $Entries -and $Item.face_path -cin $Entries) "Annotation ZIP media paths do not match archive entries."
    }
    finally { $AnnotationZip.Dispose() }
    $Checks.annotations = $true
    $Visual = Invoke-NativeOperation "visual.search" @{ input = $VehicleImage; source = $VehicleImage; threshold = .70 }
    Require (@($Visual.hits).Count -gt 0 -and $Visual.hits[0].score -ge .9) "Appearance search failed."
    $Checks.visual_search = $true
    $Detector = Invoke-NativeOperation "objects.scan" @{
        source = $VehicleImage; people = $false; vehicles = $true; output = (Join-Path $OutputDirectory "object-screenshots")
    }
    Require ($Detector.frames -eq 1 -and @($Detector.hits).Count -gt 0 -and (Test-Path -LiteralPath $Detector.hits[0].image)) "Object detection did not produce a frame and crop."
    $ObjectVideo = Invoke-NativeOperation "objects.scan" @{
        source = $FaceVideo; video_output = (Join-Path $OutputDirectory "objects.mp4")
    } "objects-video"
    Require ($ObjectVideo.frames -gt 0 -and (Test-Path -LiteralPath $ObjectVideo.output -PathType Leaf)) "Annotated MP4 was not created."
    $TrackScreens = Invoke-NativeOperation "objects.scan" @{
        source = $FaceVideo; people = $true; vehicles = $false
        output = (Join-Path $OutputDirectory "person-track-screenshots")
    } "person-track-screens"
    $TrackHits = @($TrackScreens.hits)
    $TrackFiles = @(Get-ChildItem -LiteralPath (Join-Path $OutputDirectory "person-track-screenshots") -Filter "*.png")
    Require ($TrackScreens.frames -eq $ObjectVideo.frames -and $TrackHits.Count -gt 0 -and
        $TrackHits.Count -lt $ObjectVideo.detections -and $TrackHits.Count -eq $TrackFiles.Count -and
        @($TrackHits | Select-Object -ExpandProperty track_id -Unique).Count -eq $TrackHits.Count) "Video person tracks returned duplicate crops or no matching screenshots."
    $Checks.object_detection = $true
    $RegionFixture = Join-Path $OutputDirectory "trigger-static.mkv"
    & $FFmpeg -v error -i $FaceVideo -map 0:v:0 -an -vf 'select=eq(n\,0),loop=loop=3:size=1:start=0,setpts=N/(5*TB)' -frames:v 4 -c:v ffv1 $RegionFixture
    Require ($LASTEXITCODE -eq 0) "Could not generate the region trigger fixture."
    $Dimensions = (& $FFprobe -v error -select_streams v:0 -show_entries stream=width,height -of json $RegionFixture |
        ConvertFrom-Json).streams[0]
    $Trigger = Invoke-NativeOperation "objects.scan" @{
        source = $RegionFixture; people = $true; vehicles = $false
        trigger_area = @(0, 0, $Dimensions.width, $Dimensions.height)
        output = (Join-Path $OutputDirectory "trigger-capture")
    } "region-trigger"
    Require ($Trigger.frames -eq 4 -and $Trigger.screenshots -eq 1 -and @($Trigger.failures).Count -eq 0 -and
        @($Trigger.hits).Count -gt 0 -and @($Trigger.hits | Where-Object frame_index -ne 0).Count -eq 0) "A stationary target did not create exactly one region screenshot."
    $Checks.trigger_capture = $true
    $SemanticPayload = @{ source = $VehicleImage; text = "汽车"; threshold = -.5 }
    $Semantic = Invoke-NativeOperation "semantic.search" $SemanticPayload "semantic-first"
    $SemanticAgain = Invoke-NativeOperation "semantic.search" $SemanticPayload "semantic-reuse"
    Require ($Semantic.indexed_sources -eq 1 -and @($Semantic.hits).Count -ge 1 -and $SemanticAgain.reused_sources -eq 1) "Chinese semantic search or persistent index failed."
    $Shots = Invoke-NativeOperation "hits.export" @{ hits = $Semantic.hits; output = (Join-Path $OutputDirectory "semantic-screenshots") }
    Require ($Shots.screenshots -ge 1) "Search result screenshot export failed."
    $SemanticVideo = Join-Path $OutputDirectory "semantic-reordered.mp4"
    & $FFmpeg -v error -loop 1 -framerate 5 -i $VehicleImage -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' `
        -frames:v 5 -c:v mpeg4 -pix_fmt yuv420p -bf 2 -output_ts_offset 120 $SemanticVideo
    Require ($LASTEXITCODE -eq 0) "Could not generate the semantic video fixture."
    $VideoQuery = @{ source = $SemanticVideo; text = "汽车"; threshold = -.5; step = 2 }
    $VideoIndex = Invoke-NativeOperation "semantic.search" $VideoQuery "semantic-video"
    $VideoHit = @($VideoIndex.hits | Where-Object frame_index -eq 2)
    Require ($VideoIndex.indexed_sources -eq 1 -and $VideoHit.Count -gt 0 -and
        [Math]::Abs($VideoHit[0].seconds - .4) -lt .001) "Semantic video hit is missing or is not relative to the video start."
    $VideoShot = Invoke-NativeOperation "hits.export" @{
        hits = @($VideoHit[0]); output = (Join-Path $OutputDirectory "semantic-video-screenshots")
    } "semantic-video-export"
    Require ($VideoShot.screenshots -eq 1) "Semantic video screenshot export failed."
    $Checks.semantic_search = $true
    $Plate = Invoke-NativeOperation "plate.video" @{
        source = $FaceVideo; frame_index = 0; bbox = @(30, 25, 170, 95); count = 4
        output = (Join-Path $OutputDirectory "plate-export")
    }
    Require ($Plate.tracked -ge 1 -and @($Plate.candidates).Count -ge 1) "Multi-frame restoration did not produce a candidate."
    $Checks.video_plate = $true
    $Restored = Join-Path $OutputDirectory "portrait-video.mkv"
    $env:VISUAL_ANALYSIS_BACKEND = "cpu"
    try {
        $Video = Start-Process -FilePath $Executable -WindowStyle Hidden -Wait -PassThru -ArgumentList @(
            "--restore-portrait-video", (Quote-Argument $FaceVideo), "--video-output", (Quote-Argument $Restored)
        )
    } finally { $env:VISUAL_ANALYSIS_BACKEND = $null }
    Require ($Video.ExitCode -eq 0 -and (Test-Path -LiteralPath $Restored -PathType Leaf)) "Portrait video restoration failed."
    $VideoTimesBefore = @(& $FFprobe -v error -select_streams v -show_entries 'frame=pts_time' -of csv=p=0 $FaceVideo)
    $VideoTimesAfter = @(& $FFprobe -v error -select_streams v -show_entries 'frame=pts_time' -of csv=p=0 $Restored)
    $AudioBefore = @(& $FFprobe -v error -select_streams a -show_packets -show_data_hash sha256 -show_entries 'packet=stream_index,pts_time,data_hash' -of csv=p=0 $FaceVideo)
    $AudioAfter = @(& $FFprobe -v error -select_streams a -show_packets -show_data_hash sha256 -show_entries 'packet=stream_index,pts_time,data_hash' -of csv=p=0 $Restored)
    Require ($VideoTimesBefore.Count -gt 0 -and $AudioBefore.Count -gt 0 -and
        ($VideoTimesBefore -join '|') -eq ($VideoTimesAfter -join '|') -and
        ($AudioBefore -join '|') -eq ($AudioAfter -join '|')) "Portrait export changed source frame timing or audio."
    $Checks.portrait_video = $true
    $Vehicle = Invoke-NativeOperation "vehicle.analyze" @{ input = $VehicleImage }
    Require ($Vehicle.reference_images -ge 1 -and @($Vehicle.candidates).Count -ge 1 -and
        $Vehicle.candidates[0].candidate.series -eq $ExpectedVehicleSeries) "Live public image comparison did not rank the fixture model first."
    $Records = @($Vehicle.catalogues | Where-Object source -eq "wikimedia" | ForEach-Object records)
    Require ($Records.Count -ge 1 -and [bool]($Records[0].source_url -match '^https://') -and
        (@($Vehicle.catalogues | Where-Object { $_.queried_at }).Count -ge 1)) "Public source URL or query time is missing."
    $Checks.vehicle_public = $true
    $Offline = Join-Path $OutputDirectory "offline-worker-unavailable"
    New-Item -ItemType Directory -Path $Offline | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $Offline "models") | Out-Null
    foreach ($Library in (Get-ChildItem -LiteralPath (Split-Path -Parent $Executable) -File -Filter "*.dll")) {
        New-Item -ItemType HardLink -Path (Join-Path $Offline $Library.Name) -Target $Library.FullName | Out-Null
    }
    foreach ($Model in (Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent $Executable) "models") -File)) {
        New-Item -ItemType HardLink -Path (Join-Path $Offline "models\$($Model.Name)") -Target $Model.FullName | Out-Null
    }
    $OfflineExe = Join-Path $Offline "VisualAnalysis.exe"
    New-Item -ItemType HardLink -Path $OfflineExe -Target $Executable | Out-Null
    $Local = Invoke-NativeOperation "vehicle.analyze" @{
        input = $VehicleImage; manual = @{ brand = "比亚迪"; series = $ExpectedVehicleSeries }
    } "vehicle-unavailable" $OfflineExe
    Require ($Local.status -eq "本地初判，未联网核验" -and @($Local.candidates).Count -gt 0) "Vehicle offline fallback did not preserve its local result."
    $Checks.vehicle_offline = $true
    $ExpectedInputs = @(Get-ChildItem -LiteralPath $CaseDirectory -File -Filter "*.jpg")
    Require ($ExpectedInputs.Count -eq 3) "Acceptance expects the three local photographs."
    foreach ($Mode in @("standard", "recall")) {
        $CaseReport = Join-Path $OutputDirectory "cases-$Mode.json"
        $Arguments = @("--check-restoration-cases", (Quote-Argument $CaseDirectory),
            "--case-output", (Quote-Argument (Join-Path $OutputDirectory "case-images")),
            "--report", (Quote-Argument $CaseReport))
        if ($Mode -eq "recall") { $Arguments += "--prefer-recall" }
        if ($Mode -eq "recall") { $env:VISUAL_ANALYSIS_BACKEND = "cpu" }
        try { $Run = Start-Process -FilePath $Executable -WindowStyle Hidden -Wait -PassThru -ArgumentList $Arguments }
        finally { $env:VISUAL_ANALYSIS_BACKEND = $null }
        Require ($Run.ExitCode -eq 0) "Native portrait cases failed in $Mode mode."
        $Report = Get-Content -LiteralPath $CaseReport -Raw -Encoding UTF8 | ConvertFrom-Json
        Require ($Report.ok -and @($Report.cases).Count -eq 3) "Portrait case report is incomplete in $Mode mode."
        foreach ($Case in $Report.cases) {
            $Candidates = @($Case.candidates)
            $ExpectedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $CaseDirectory $Case.source)).Hash
            Require ($ExpectedHash -ieq $Case.source_sha256 -and $Candidates.Count -in 1,2) "Case provenance or candidate count is incorrect."
            foreach ($Candidate in $Candidates) {
                Require ($Candidate.score -ge 0 -and $Candidate.score -le 100 -and
                    (Test-Path -LiteralPath $Candidate.file -PathType Leaf)) "Case result is missing a score or image."
            }
            if ($Candidates.Count -eq 2) {
                Require (($Candidates[0].agreement -ge 92) -and ($Candidates[1].agreement -ge 92) -and
                    ((Get-FileHash -LiteralPath $Candidates[0].file).Hash -ne (Get-FileHash -LiteralPath $Candidates[1].file).Hash)) "Near-duplicate or inconsistent portrait alternatives."
            }
        }
        $Checks["portrait_$Mode"] = $true
    }
    $Digest = [ordered]@{}
    foreach ($Model in (Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent $Executable) "models") -File)) {
        $Digest[$Model.Name] = (Get-FileHash -Algorithm SHA256 -LiteralPath $Model.FullName).Hash.ToLowerInvariant()
    }
    $NativeRuntime = [ordered]@{}
    foreach ($Library in (Get-ChildItem -LiteralPath (Split-Path -Parent $Executable) -File -Filter "*.dll")) {
        $NativeRuntime[$Library.Name] = (Get-FileHash -Algorithm SHA256 -LiteralPath $Library.FullName).Hash.ToLowerInvariant()
    }
    $Gpu = [ordered]@{}
    $IntelGpuRuns = [ordered]@{}
    $Profiles = @(Get-ChildItem -LiteralPath $env:VISUAL_ANALYSIS_GPU_PROFILE -File)
    foreach ($Name in @("scrfd_10g_bnkps", "glintr100", "rtdetr_v2_r50vd", "chinese_clip_vision",
                        "chinese_clip_text", "gfpgan-v1.4-fp32", "pcflow-bfr-fp32")) {
        $Completed = @($Profiles | Where-Object Name -Like "$Name-*.gpu")
        $Counts = foreach ($File in $Completed) {
            $Prefix = [IO.Path]::GetFileNameWithoutExtension($File.Name)
            if ($Profiles.Name -contains "$Prefix.fallback") { continue }
            foreach ($Trace in ($Profiles | Where-Object Name -Like "${Prefix}_*.json")) {
                $Profile = Get-Content -LiteralPath $Trace.FullName -Raw -Encoding UTF8
                [regex]::Matches($Profile, '"provider"\s*:\s*"DmlExecutionProvider"').Count
            }
        }
        $Gpu[$Name] = [int](($Counts | Measure-Object -Sum).Sum)
        $IntelRuns = foreach ($File in ($Profiles | Where-Object Name -Like "$Name-*.ovgpu")) {
            if($Profiles.Name -notcontains "$($File.BaseName).fallback") {
                [int](Get-Content -LiteralPath $File.FullName -Raw -Encoding ASCII)
            }
        }
        $IntelGpuRuns[$Name] = [int](($IntelRuns | Measure-Object -Sum).Sum)
        Require (($Gpu[$Name] -gt 0) -or ($IntelGpuRuns[$Name] -gt 0)) "No completed GPU inference was recorded for $Name."
    }
    $Checks.gpu = $true
    [ordered]@{
        ok = $true; exe_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Executable).Hash.ToLowerInvariant()
        worker_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path (Split-Path -Parent $Executable) "VehicleCatalogueWorker.exe")).Hash.ToLowerInvariant()
        models = $Digest
        native_runtime = $NativeRuntime
        directml_node_events = $Gpu
        openvino_gpu_runs = $IntelGpuRuns
        display_adapters = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
        finished = [DateTime]::UtcNow.ToString("o"); checks = $Checks
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory "acceptance.json") -Encoding UTF8
    $Checks
}
finally {
    $env:FACE_MATCHING_HOME = $PreviousHome
    $env:VISUAL_ANALYSIS_GPU_PROFILE = $PreviousProfile
    $env:VISUAL_ANALYSIS_BACKEND = $PreviousBackend
}
