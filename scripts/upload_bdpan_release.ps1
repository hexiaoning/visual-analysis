[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$CredentialsFile,
    [switch]$PrunePrevious
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$settings = (Get-Content -LiteralPath $CredentialsFile -Raw -Encoding UTF8 | ConvertFrom-Json).baidu_netdisk
$cli = (Resolve-Path -LiteralPath $settings.executable).Path
$config = (Resolve-Path -LiteralPath $settings.config_path).Path
$root = [string]$settings.release_directory
if ($root -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') {
    throw "The Baidu Netdisk release directory must be a single folder name."
}
$cliOptions = @("--config-path", $config, "--no-check-update", "--json")

function Invoke-Netdisk {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $raw = & $cli @Arguments @cliOptions
    $exit = $LASTEXITCODE
    $reply = ($raw -join [Environment]::NewLine) | ConvertFrom-Json
    if ($exit -ne 0 -or ($reply -is [pscustomobject] -and
            $reply.PSObject.Properties["code"] -and $reply.code -ne 0)) {
        throw "Baidu Netdisk command failed: $($Arguments[0]) (exit $exit)."
    }
    $reply
}

function Ensure-NetdiskFolder([string]$Parent, [string]$Name) {
    $entry = Invoke-Netdisk -Arguments @("ls", $Parent) |
        Where-Object { $_.server_filename -ceq $Name } | Select-Object -First 1
    if ($null -ne $entry) {
        if ($entry.isdir -ne 1) {
            throw "An existing remote file conflicts with directory $Name."
        }
        return
    }
    $path = if ($Parent -eq ".") { $Name } else { "$Parent/$Name" }
    try {
        $null = Invoke-Netdisk -Arguments @("mkdir", $path)
    }
    catch {
        # A concurrent release may already have created the same folder.
        $created = Invoke-Netdisk -Arguments @("ls", $Parent) |
            Where-Object { $_.server_filename -ceq $Name -and $_.isdir -eq 1 }
        if (-not $created) { throw }
    }
}

function Remove-PreviousNetdiskRelease([object[]]$Previous) {
    foreach ($candidate in $Previous) {
        $oldPath = "$root/$($candidate.server_filename)"
        $oldVersion = ($candidate.server_filename -split "-", 2)[0]
        $oldZip = "FaceMatching-$oldVersion-windows-x64.zip"
        $oldListing = @(Invoke-Netdisk -Arguments @("ls", $oldPath))
        $oldNames = @($oldListing | Where-Object { $_.isdir -eq 0 -and [long]$_.size -gt 0 } |
            Select-Object -ExpandProperty server_filename)
        if ($oldListing.Count -ne 2 -or $oldZip -cnotin $oldNames -or "$oldZip.sha256" -cnotin $oldNames) {
            continue
        }
        $null = Invoke-Netdisk -Arguments @("rm", $oldPath, "--force")
        if (Invoke-Netdisk -Arguments @("ls", $root) | Where-Object { $_.server_filename -ceq $candidate.server_filename }) {
            throw "Baidu Netdisk still lists the previous release $oldPath."
        }
        Write-Host "Removed the previous Baidu Netdisk release: $oldPath"
        break
    }
}

function Save-NetdiskShare([string]$Destination, [string]$Version, [string]$Hash, [string]$ArchiveName) {
    $recordDirectory = Join-Path $env:LOCALAPPDATA "VisualAnalysis"
    $null = New-Item -ItemType Directory -Path $recordDirectory -Force
    $recordPath = Join-Path $recordDirectory "latest-release.json"
    $now = [DateTimeOffset]::UtcNow
    if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
        $existing = Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($existing.sha256 -ceq $Hash -and $existing.netdisk_directory -ceq $Destination -and
                [DateTimeOffset]$existing.expires_at -gt $now) {
            Write-Host "Baidu Netdisk share: $($existing.share_url)"
            Write-Host "Share record: $recordPath"
            return
        }
    }

    $share = Invoke-Netdisk -Arguments @("share", $Destination, "--period", "7")
    $url = [uri]$share.link
    if (-not $url.IsAbsoluteUri -or $url.Scheme -ne "https" -or
            $url.Host -ne "pan.baidu.com" -or $url.AbsolutePath -notmatch '^/s/[^/]+$') {
        throw "Baidu Netdisk did not return a valid release share link."
    }
    $record = [ordered]@{
        version = $Version
        netdisk_directory = $Destination
        archive = $ArchiveName
        checksum = "$ArchiveName.sha256"
        sha256 = $Hash
        share_url = $share.link
        extraction_code = $share.pwd
        created_at = $now.ToString("o")
        expires_at = $now.AddDays(7).ToString("o")
    }
    $temporary = Join-Path $recordDirectory ("latest-release-" + [guid]::NewGuid().ToString("N") + ".tmp")
    try {
        [IO.File]::WriteAllText($temporary, ($record | ConvertTo-Json) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
            [IO.File]::Replace($temporary, $recordPath, [NullString]::Value)
        }
        else {
            [IO.File]::Move($temporary, $recordPath)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary }
    }
    Write-Host "Baidu Netdisk share: $($share.link)"
    Write-Host "Share record: $recordPath"
}

$zip = Get-Item -LiteralPath $Archive
if ($zip.PSIsContainer -or $zip.Name -notmatch '^FaceMatching-(v\d+\.\d+\.\d+)-windows-x64\.zip$') {
    throw "Expected a versioned Visual Analysis release ZIP."
}
$version = $Matches[1]
$checksum = Get-Item -LiteralPath ($zip.FullName + ".sha256")
$manifest = (Get-Content -LiteralPath $checksum.FullName -Raw -Encoding UTF8).Trim() -split '\s+', 2
if ($manifest.Count -ne 2 -or $manifest[0] -notmatch '^[a-fA-F0-9]{64}$' -or $manifest[1] -cne $zip.Name) {
    throw "The release checksum file has an unexpected format."
}
$hash = (Get-FileHash -LiteralPath $zip.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $manifest[0]) { throw "The release ZIP does not match its SHA-256 file." }

$status = Invoke-Netdisk -Arguments @("whoami")
if (-not $status.authenticated -or -not $status.has_valid_token) {
    throw "Baidu Netdisk authorization is unavailable. Sign in on the build machine first."
}

$versionDirectory = "$version-$($hash.Substring(0, 12))"
$destination = "$root/$versionDirectory"
$recordDirectory = Join-Path $env:LOCALAPPDATA "VisualAnalysis"
$null = New-Item -ItemType Directory -Path $recordDirectory -Force
$progressPath = Join-Path $recordDirectory "release-progress.json"
$progress = [ordered]@{
    version = $version; netdisk_directory = $destination; phase = "preparing";
    total_bytes = $zip.Length + $checksum.Length; confirmed_bytes = 0;
    estimate_basis = "process_read_bytes_minus_first_file_pass"
}
. (Join-Path $PSScriptRoot "baidu_upload_progress.ps1")
trap {
    Write-ReleaseProgress "failed" @{ error = $_.Exception.Message }
    throw $_
}
Write-ReleaseProgress "preparing"
Ensure-NetdiskFolder "." $root
$previous = @()
if ($PrunePrevious) {
    $currentVersion = [version]$version.Substring(1)
    $previous = @(Invoke-Netdisk -Arguments @("ls", $root) | Where-Object {
        $match = [regex]::Match($_.server_filename, '^v(\d+\.\d+\.\d+)-[a-f0-9]{12}$')
        $_.isdir -eq 1 -and $match.Success -and $_.server_filename -cne $versionDirectory -and
            [version]$match.Groups[1].Value -le $currentVersion
    } | Sort-Object -Descending -Property @{ Expression = {
        [version](($_.server_filename -split "-", 2)[0].Substring(1))
    } }, server_ctime)
}
Ensure-NetdiskFolder $root $versionDirectory
foreach ($file in @($zip, $checksum)) {
    Write-Host "Uploading $($file.Name) to Baidu Netdisk..."
    $reply = Invoke-NetdiskUpload $file "$destination/$($file.Name)"
    if ($reply.data.action -notin @("created", "unchanged")) {
        throw "Baidu Netdisk did not confirm the requested path for $($file.Name)."
    }
    Write-ReleaseProgress "upload_confirmed" @{
        confirmed_bytes = $progress.confirmed_bytes + $file.Length;
        estimated_current_file_bytes = $null; estimated_remaining_seconds = $null
    }
}

Write-ReleaseProgress "verifying"
$listing = @(Invoke-Netdisk -Arguments @("ls", $destination))
foreach ($file in @($zip, $checksum)) {
    $remote = $listing | Where-Object {
        $_.server_filename -ceq $file.Name -and $_.isdir -eq 0 -and [long]$_.size -eq $file.Length
    }
    if (-not $remote) { throw "Remote listing did not confirm $($file.Name) and its byte count." }
    Write-Host "Baidu Netdisk verified: $($remote.path)"
}

Write-ReleaseProgress "sharing"
Save-NetdiskShare $destination $version $hash $zip.Name
if ($PrunePrevious) {
    Write-ReleaseProgress "pruning_previous"
    Remove-PreviousNetdiskRelease $previous
}
Write-ReleaseProgress "published" @{ file = $null }
