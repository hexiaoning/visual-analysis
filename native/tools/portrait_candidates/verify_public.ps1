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
$Cases = (Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot "public-regression.json") |
    ConvertFrom-Json).cases
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
    foreach ($fit in $false, $true) {
        $arguments = @()
        if ($fit) { $arguments += "--fit-duplicate" }
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
            $expected = if ($fit) { $Case.fitted } else { $Case.default }
            if (($report.previous_rule_selected -join ",") -cne ($Case.default -join ",") -or
                ($report.fitted_duplicate_rule_selected -join ",") -cne ($Case.fitted -join ",") -or
                $report.fit_duplicate_enabled -ne $fit -or ($report.selected -join ",") -cne ($expected -join ",") -or
                ($report.outputs.route -join ",") -cne ($expected -join ",")) {
                throw "Unexpected selection for sample $($Case.id), fit=$fit."
            }
            Test-Pixels (Join-Path $folder $report.observed_luminance_png) 128 $Case.observed_hash
            foreach ($item in $report.outputs) {
                $route = $item.route
                Test-Pixels (Join-Path $folder $item.png) 512 $Case.input_hashes.$route
                Test-Pixels (Join-Path $folder $item.replay_luminance_png) 128 $Case.replay_hashes.$route
            }
            Write-Output "$($Case.id) fit=$fit selected=$($expected -join ',')"
        }
    }
} finally {
    [Console]::OutputEncoding = $OldEncoding
    # Delete only the unique temporary tree created by this invocation.
    $actual = (Resolve-Path -LiteralPath $OutputRoot).ProviderPath
    if ($actual -ceq $OutputRoot -and [IO.Path]::GetDirectoryName($actual) -ieq $TempRoot.TrimEnd('\')) {
        Remove-Item -LiteralPath $actual -Recurse -Force
    }
}
