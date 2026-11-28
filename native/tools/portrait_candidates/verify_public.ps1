[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$FixtureDirectory
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Drawing

function Get-PixelReceipt([string]$File) {
    $bitmap = New-Object System.Drawing.Bitmap($File)
    try {
        $width = $bitmap.Width
        $height = $bitmap.Height
        $bounds = New-Object System.Drawing.Rectangle(0, 0, $width, $height)
        $bits = $bitmap.LockBits($bounds, [Drawing.Imaging.ImageLockMode]::ReadOnly,
            [Drawing.Imaging.PixelFormat]::Format24bppRgb)
        try {
            # GDI+ BGR rows may be padded or use a negative stride. Hash only actual pixel bytes.
            $rowBytes = $width * 3
            $bytes = New-Object byte[] ($height * $rowBytes)
            for ($y = 0; $y -lt $height; ++$y) {
                $address = [IntPtr]::Add($bits.Scan0, $y * $bits.Stride)
                [Runtime.InteropServices.Marshal]::Copy($address, $bytes, $y * $rowBytes, $rowBytes)
            }
            $hash = [Security.Cryptography.SHA256]::Create()
            try {
                $digest = [BitConverter]::ToString($hash.ComputeHash($bytes)).Replace("-", "").ToLowerInvariant()
            } finally {
                $hash.Dispose()
            }
            [pscustomobject]@{ Width = $width; Height = $height; Hash = $digest }
        } finally {
            $bitmap.UnlockBits($bits)
        }
    } finally {
        $bitmap.Dispose()
    }
}

function Test-Pixels([string]$File, [int]$Side, [string]$Expected) {
    $result = Get-PixelReceipt $File
    if ($result.Width -ne $Side -or $result.Height -ne $Side -or $result.Hash -cne $Expected) {
        throw "Unexpected pixels or dimensions in $File. Actual BGR SHA-256 $($result.Hash)."
    }
}

$App = (Resolve-Path -LiteralPath $Executable).ProviderPath
$Graph = (Resolve-Path -LiteralPath $Model).ProviderPath
$Fixtures = (Resolve-Path -LiteralPath $FixtureDirectory).ProviderPath
$Reference = (Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot "public-regression.json") |
    ConvertFrom-Json)
$Cases = $Reference.cases
$TextureModel = Join-Path $PSScriptRoot "texture-guard-v1.json"
$TempRoot = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).ProviderPath
$OutputRoot = Join-Path $TempRoot ("portrait-candidates-" + [guid]::NewGuid().ToString("N"))
New-Item -Path $OutputRoot -ItemType Directory -ErrorAction Stop | Out-Null
$OldEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    foreach ($Case in $Cases) {
        foreach ($route in "source", "codeformer", "amd") {
            $png = Join-Path (Join-Path $Fixtures $Case.id) "$route.png"
            Test-Pixels $png 512 $Case.input_hashes.$route
        }
    }
    foreach ($mode in "default", "fit-duplicate", "texture-guard") {
        $fit = $mode -ne "default"
        $arguments = @()
        if ($mode -eq "fit-duplicate") { $arguments += "--fit-duplicate" }
        if ($mode -eq "texture-guard") { $arguments += @("--texture-guard", $TextureModel) }
        $arguments += $Graph
        foreach ($Case in $Cases) {
            foreach ($route in "source", "codeformer", "amd") {
                $arguments += Join-Path (Join-Path $Fixtures $Case.id) "$route.png"
            }
            $arguments += Join-Path $OutputRoot $Case.id
        }
        $folders = @(& $App @arguments)
        if ($LASTEXITCODE -ne 0 -or $folders.Count -ne $Cases.Count) {
            throw "Batch regression failed with exit code $LASTEXITCODE."
        }
        for ($i = 0; $i -lt $Cases.Count; $i++) {
            $Case = $Cases[$i]
            $folder = $folders[$i]
            $report = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $folder "selection.json") | ConvertFrom-Json
            $expected = $Case.default
            if ($mode -eq "fit-duplicate") { $expected = $Case.fitted }
            if ($mode -eq "texture-guard") {
                $expected = $Case.guarded
                $score = $Case.replacement_texture_score
                if (($report.texture_guard_selected -join ",") -cne ($expected -join ",") -or
                    $report.texture_cutoff -ne $Reference.texture_cutoff -or
                    ($null -eq $report.replacement_texture_score) -ne ($null -eq $score) -or
                    ($null -ne $score -and [Math]::Abs($report.replacement_texture_score - $score) -gt 0.00001)) {
                    throw "Unexpected texture guard result for $($Case.id)."
                }
            } elseif ($null -ne $report.texture_guard_selected -or $null -ne $report.texture_cutoff -or
                      $null -ne $report.replacement_texture_score) {
                throw "Unexpected texture result without a guard model for $($Case.id)."
            }
            if ($report.mode -cne $mode -or ($report.previous_rule_selected -join ",") -cne ($Case.default -join ",") -or
                ($report.fitted_duplicate_rule_selected -join ",") -cne ($Case.fitted -join ",") -or
                $report.fit_duplicate_enabled -ne $fit -or ($report.selected -join ",") -cne ($expected -join ",") -or
                ($report.outputs.route -join ",") -cne ($expected -join ",")) {
                throw "Unexpected selection for sample $($Case.id), mode=$mode."
            }
            Test-Pixels (Join-Path $folder $report.observed_luminance_png) 128 $Case.observed_hash
            foreach ($item in $report.outputs) {
                $route = $item.route
                Test-Pixels (Join-Path $folder $item.png) 512 $Case.input_hashes.$route
                Test-Pixels (Join-Path $folder $item.replay_luminance_png) 128 $Case.replay_hashes.$route
            }
            Write-Output "$($Case.id) mode=$mode selected=$($expected -join ',')"
        }
    }
    $CoverCase = $Cases | Where-Object id -EQ "2762"
    $CoverSource = Join-Path $Fixtures $CoverCase.id
    $Primary = Join-Path $CoverSource "codeformer.png"
    $Alternate = Join-Path $CoverSource "amd.png"
    $CoverFolder = & $App --cover-shapes 0 (Join-Path $OutputRoot "cover") $Primary $Alternate $Primary
    if ($LASTEXITCODE -ne 0) {
        throw "Shape cover regression failed with exit code $LASTEXITCODE."
    }
    $Cover = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $CoverFolder "selection.json") | ConvertFrom-Json
    $CoverDistances = @($Cover.measured_distance)
    $NonzeroDistances = @($CoverDistances | Where-Object { $null -eq $_ -or [double]$_ -ne 0.0 })
    if ($Cover.mode -cne "shape-cover" -or ($Cover.outputs.input_index -join ",") -cne "0,1" -or
        ($Cover.nearest_input_index -join ",") -cne "0,1,0" -or $CoverDistances.Count -ne 3 -or $NonzeroDistances.Count -ne 0) {
        throw "Shape cover should retain both distinct fixtures and collapse the repeated image."
    }
    Test-Pixels (Join-Path $CoverFolder $Cover.outputs[0].png) 512 $CoverCase.input_hashes.codeformer
    Test-Pixels (Join-Path $CoverFolder $Cover.outputs[1].png) 512 $CoverCase.input_hashes.amd
    Write-Output "2762 mode=shape-cover selected=0,1"

    $ExposureSource = Join-Path $OutputRoot "exposure-source.png"
    $Pixel = New-Object System.Drawing.Bitmap(1, 1)
    try {
        $Pixel.SetPixel(0, 0, [Drawing.Color]::FromArgb(0, 0, 1))
        $Pixel.Save($ExposureSource, [Drawing.Imaging.ImageFormat]::Png)
        $Pixel.SetPixel(0, 0, [Drawing.Color]::Black)
        $Black = Join-Path $OutputRoot "exposure-black.png"
        $Pixel.Save($Black, [Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $Pixel.Dispose()
    }
    $ExposureRoot = Join-Path $OutputRoot "short-exposure"
    $ExposureArgs = @("--short-exposure-cover", $ExposureSource, $ExposureRoot, 1, 1, 0, 0, 1)
    $FirstPage = & $App @ExposureArgs
    if ($LASTEXITCODE -ne 0) { throw "Could not generate the first exposure page." }
    $SecondPage = & $App @ExposureArgs $FirstPage
    if ($LASTEXITCODE -ne 0) { throw "Could not resume the exposure page." }
    $SourceReceipt = Get-PixelReceipt $ExposureSource
    $CandidateFiles = @($Black, $ExposureSource)
    $Pages = @($FirstPage, $SecondPage)
    for ($i = 0; $i -lt $Pages.Count; $i++) {
        $Page = $Pages[$i]
        $Last = $i -eq 1
        $Report = Get-Content -Raw -LiteralPath (Join-Path $Page "exposure-cover.json") | ConvertFrom-Json
        $Output = @($Report.outputs)
        if ($Report.continued -ne $Last -or $Report.exhausted -ne $Last -or $Output.Count -ne 1) {
            throw "The quantized exposure has exactly two solutions on two pages."
        }
        Test-Pixels (Join-Path $Page $Output[0].candidate) 1 (Get-PixelReceipt $CandidateFiles[$i]).Hash
        Test-Pixels (Join-Path $Page $Output[0].companion) 1 $SourceReceipt.Hash
    }
    Write-Output "short-exposure-cover pages=2 exhausted=true"

    $SparseArgs = @("--short-exposure-cover", $Black, $ExposureRoot, 1, 1, 0, 0, 4, "--capture-error", 1, 1)
    $FirstPage = & $App @SparseArgs
    if ($LASTEXITCODE -ne 0) { throw "Could not generate the first page with capture error." }
    $LastPage = & $App @SparseArgs $FirstPage
    if ($LASTEXITCODE -ne 0) { throw "Could not resume the page with capture error." }
    $Distinct = [Collections.Generic.HashSet[string]]::new()
    $BlackReceipt = Get-PixelReceipt $Black
    foreach ($Page in @($FirstPage, $LastPage)) {
        $Report = Get-Content -Raw -LiteralPath (Join-Path $Page "exposure-cover.json") | ConvertFrom-Json
        if ($Report.exhausted -ne ($Page -ceq $LastPage)) {
            throw "Unexpected end of the cover with capture error."
        }
        foreach ($Output in $Report.outputs) {
            $Raster = New-Object System.Drawing.Bitmap((Join-Path $Page $Output.candidate))
            try {
                $Value = $Raster.GetPixel(0, 0)
                $Channels = @($Value.R, $Value.G, $Value.B)
                $Nonzero = @($Channels | Where-Object { $_ -gt 0 }).Count
                if ($Nonzero -gt 1 -or ($Channels | Measure-Object -Maximum).Maximum -gt 2 -or
                    $Output.changed_channels -ne $Nonzero -or $Output.capture_error_gray -ne $Nonzero -or
                    -not $Distinct.Add($Channels -join ",")) {
                    throw "The page must emit each feasible color once, spending at most one changed channel."
                }
            } finally {
                $Raster.Dispose()
            }
            Test-Pixels (Join-Path $Page $Output.companion) 1 $BlackReceipt.Hash
        }
    }
    if ($Distinct.Count -ne 7) { throw "The budget allows seven colors across both pages." }
    Write-Output "short-exposure-cover with capture error pages=2 solutions=7"
} finally {
    [Console]::OutputEncoding = $OldEncoding
    # Delete only the unique temporary tree created by this invocation.
    $actual = (Resolve-Path -LiteralPath $OutputRoot).ProviderPath
    if ($actual -ceq $OutputRoot -and [IO.Path]::GetDirectoryName($actual) -ieq $TempRoot.TrimEnd('\')) {
        Remove-Item -LiteralPath $actual -Recurse -Force
    }
}
