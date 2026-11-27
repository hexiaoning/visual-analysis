[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InputPng,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$ModelDirectory,
    [Parameter(Mandatory)][string]$MomentsExecutable,
    [Parameter(Mandatory)][string]$CodesExecutable,
    [Parameter(Mandatory)][string]$CandidatesExecutable,
    [ValidateSet("GPU", "CPU")][string]$Device = "GPU",
    [switch]$RefineClarity,
    [ValidateScript({ $_ -gt 0.0 -and $_ -lt 0.055 })][double]$CloseSeparation,
    [switch]$TransferGuidedDetail,
    [switch]$ExploreExposureDirections
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InputFile = (Resolve-Path -LiteralPath $InputPng).ProviderPath
$Moments = (Resolve-Path -LiteralPath $MomentsExecutable).ProviderPath
$Codes = (Resolve-Path -LiteralPath $CodesExecutable).ProviderPath
$Candidates = (Resolve-Path -LiteralPath $CandidatesExecutable).ProviderPath
$Models = @{}
$Names = @("jin-240.onnx", "signed-512.onnx", "encode.onnx", "decode.onnx", "hr-nosa-wflw-geometry.onnx")
if ($ExploreExposureDirections) {
    $Names += @("tail-1.onnx", "tail-2.onnx", "original.onnx", "madenn.onnx", "learned-exposure.bin")
}
foreach ($Name in $Names) {
    $Models[$Name] = (Resolve-Path -LiteralPath (Join-Path $ModelDirectory $Name)).ProviderPath
}
$Destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
if (Test-Path -LiteralPath $Destination) {
    throw "Choose a new output directory."
}
$Destination = $Destination.TrimEnd('\', '/')
$Pending = "$Destination.$([Guid]::NewGuid().ToString('N')).pending"
$null = New-Item -ItemType Directory -Path $Pending
$Backend = if ($Device -eq "GPU") { "dml" } else { "cpu" }

function Invoke-NativeStage([string]$Program, [string[]]$Parameters, [string]$Log) {
    & $Program @Parameters | Out-File -LiteralPath $Log -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "Native stage failed with exit code $LASTEXITCODE. See $(Join-Path $Pending $Log)."
    }
}

Push-Location -LiteralPath $Pending
try {
    Invoke-NativeStage $Codes @("validate", $InputFile) "input.log"
    if ($ExploreExposureDirections) {
        Invoke-NativeStage $Moments @("--refine-sequence", $Device, $Models["jin-240.onnx"], $Models["signed-512.onnx"],
            "--additional-refinement", $Models["tail-1.onnx"], "--additional-refinement", $Models["tail-2.onnx"],
            "--positions", "1,3,5,7", $InputFile, "all-seeds") "moments.log"
        $null = New-Item -ItemType Directory -Path "seeds", "exposure-seeds"
        Copy-Item -LiteralPath "all-seeds/candidate-1.png" -Destination "seeds/candidate-1.png"
        Copy-Item -LiteralPath "all-seeds/candidate-4.png" -Destination "seeds/candidate-2.png"
        foreach ($Index in 1..8) {
            Copy-Item -LiteralPath "all-seeds/candidate-$($Index + 4).png" -Destination "exposure-seeds/candidate-$Index.png"
        }
    } else {
        Invoke-NativeStage $Moments @("--refine-sequence", $Device, $Models["jin-240.onnx"], $Models["signed-512.onnx"],
            "--positions", "1,7", $InputFile, "seeds") "moments.log"
    }
    Invoke-NativeStage $Codes @("encode", $Backend, $Models["encode.onnx"], $InputFile, "raw.f32",
        "seeds/candidate-1.png", "first.f32", "seeds/candidate-2.png", "last.f32") "encode.log"
    Invoke-NativeStage $Codes @("blend", "raw.f32", "raw.f32", "raw.bin", "raw.f32", "first.f32", "first.bin",
        "raw.f32", "last.f32", "last.bin") "blend.log"
    Invoke-NativeStage $Codes @("decode", $Backend, $Models["decode.onnx"], $InputFile, "raw.bin", "direct.png",
        $InputFile, "first.bin", "first.png", $InputFile, "last.bin", "last.png") "decode.log"
    $Bank = @("direct.png", "first.png", "last.png")
    if ($TransferGuidedDetail) {
        $GuideDirectories = @("guided-first", "guided-last")
        $Arguments = @("--transfer-guided-detail", "direct.png", "seeds/candidate-1.png", $GuideDirectories[0],
            "direct.png", "seeds/candidate-2.png", $GuideDirectories[1])
        Invoke-NativeStage $Candidates $Arguments "guided-detail-paths.log"
        $GuidedPaths = @(Get-Content -LiteralPath "guided-detail-paths.log" | Where-Object { $_.Trim() })
        if ($GuidedPaths.Count -ne $GuideDirectories.Count) {
            throw "The native detail transfer did not publish both endpoints."
        }
        $GuidedReceipts = @()
        for ($Index = 0; $Index -lt $GuideDirectories.Count; ++$Index) {
            $Run = Split-Path -Leaf $GuidedPaths[$Index]
            $Directory = Join-Path $GuideDirectories[$Index] $Run
            $GuidedReceipt = Join-Path $Directory "transfer.json"
            if (-not (Test-Path -LiteralPath $GuidedReceipt -PathType Leaf)) {
                throw "Missing native detail transfer receipt at $GuidedReceipt."
            }
            foreach ($File in "warped.png", "adjusted.png") {
                $Image = Join-Path $Directory $File
                if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
                    throw "Missing native detail transfer candidate at $Image."
                }
                $Bank += $Image
            }
            $GuidedReceipts += $GuidedReceipt
        }
        Set-Content -LiteralPath "guided-detail-paths.log" -Value $GuidedReceipts -Encoding utf8
    }
    $Arguments = @("--extend-shape-cover", ".055", ".028", "1", "--max-additions", "2")
    if ($PSBoundParameters.ContainsKey("CloseSeparation")) {
        $Arguments += @("--close-separation", $CloseSeparation.ToString("G17", [Globalization.CultureInfo]::InvariantCulture))
    }
    $Arguments += @("--landmarks", $Models["hr-nosa-wflw-geometry.onnx"], $InputFile, "selected") + $Bank
    Invoke-NativeStage $Candidates $Arguments "selection-path.log"

    $Selection = (Get-Content -LiteralPath "selection-path.log" -Raw).Trim()
    $SelectionName = Split-Path -Leaf $Selection
    $Receipt = Join-Path "selected\$SelectionName" "selection.json"
    if (-not (Test-Path -LiteralPath $Receipt -PathType Leaf)) {
        throw "The native selector did not publish a result in $Pending."
    }
    $DisplayInputs = $Bank
    if ($RefineClarity) {
        $Initial = Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
        $SelectedNames = @($Initial.outputs | ForEach-Object { $Bank[[int]$_.input_index] })
        $AddedCount = $SelectedNames.Count - [int]$Initial.kept_count
        $OrderedBank = $SelectedNames + @($Bank | Where-Object { $_ -notin $SelectedNames })
        $DisplayInputs = $OrderedBank
        Set-Content -LiteralPath "refinement-inputs.log" -Value $OrderedBank -Encoding utf8
        $Arguments = @("--refine-clear-representatives", "$($Initial.kept_count)", "$AddedCount",
            $Models["hr-nosa-wflw-geometry.onnx"], $InputFile, "refined") + $OrderedBank
        Invoke-NativeStage $Candidates $Arguments "refinement-path.log"
        $Name = Split-Path -Leaf (Get-Content -LiteralPath "refinement-path.log" -Raw).Trim()
        $Receipt = Join-Path "refined\$Name" "selection.json"
        if (-not (Test-Path -LiteralPath $Receipt -PathType Leaf)) {
            throw "The clarity refinement did not publish a result in $Pending."
        }
        Set-Content -LiteralPath "refinement-path.log" -Value $Receipt -Encoding utf8
    }
    if ($ExploreExposureDirections) {
        Invoke-NativeStage $Codes @("restore", $Backend, $Models["original.onnx"],
            $InputFile, "exposure-first.png") "exposure-first.log"
        Invoke-NativeStage $Codes @("restore", $Backend, $Models["madenn.onnx"],
            $InputFile, "exposure-second.png") "exposure-second.log"
        Invoke-NativeStage $Candidates @("--bounded-exposure-directions", $Models["learned-exposure.bin"],
            $InputFile, "exposure-first.png", "exposure-second.png", "exposure-seeds", "exposure-directions") "exposure-directions.log"
        $DirectionRun = Split-Path -Leaf (Get-Content -LiteralPath "exposure-directions.log" -Raw).Trim()
        $DirectionsRoot = Join-Path "exposure-directions" $DirectionRun
        $DirectionReceipt = Join-Path $DirectionsRoot "directions.json"
        $Directions = Get-Content -LiteralPath $DirectionReceipt -Raw | ConvertFrom-Json
        Set-Content -LiteralPath "exposure-directions.log" -Value $DirectionReceipt -Encoding utf8
        $NewStates = @($Directions.groups | Where-Object eligible | ForEach-Object {
            $_.outputs | ForEach-Object { Join-Path $DirectionsRoot $_.png }
        })
        if ($NewStates.Count -gt 0) {
            $Initial = Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
            $Protected = @($Initial.outputs | ForEach-Object { $DisplayInputs[[int]$_.input_index] })
            $DisplayInputs = $Protected + $NewStates
            $Radius = if ($PSBoundParameters.ContainsKey("CloseSeparation")) { $CloseSeparation } else { 0.025 }
            $Distance = $Radius.ToString("G17", [Globalization.CultureInfo]::InvariantCulture)
            $Arguments = @("--extend-shape-cover", ".055", "1", "$($Protected.Count)", "--max-additions", "2",
                "--close-separation", $Distance, "--landmarks", $Models["hr-nosa-wflw-geometry.onnx"],
                $InputFile, "selected-exposure") + $DisplayInputs
            Invoke-NativeStage $Candidates $Arguments "exposure-selection.log"
            $Name = Split-Path -Leaf (Get-Content -LiteralPath "exposure-selection.log" -Raw).Trim()
            $Receipt = Join-Path ("selected-exposure\" + $Name) "selection.json"
            if (-not (Test-Path -LiteralPath $Receipt -PathType Leaf)) {
                throw "The exposure selector did not publish a result."
            }
            Set-Content -LiteralPath "exposure-selection.log" -Value $Receipt -Encoding utf8
        }
        Set-Content -LiteralPath "exposure-inputs.log" -Value $DisplayInputs -Encoding utf8
    }
    Set-Content -LiteralPath "selection-path.log" -Value $Receipt -Encoding utf8
} finally {
    Pop-Location
}

[IO.Directory]::Move($Pending, $Destination)
Write-Output (Join-Path $Destination $Receipt)
