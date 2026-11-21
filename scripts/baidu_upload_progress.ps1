function Write-ReleaseProgress([string]$Phase, [hashtable]$Details = @{}) {
    $progress.phase = $Phase
    $progress.updated_at = [DateTimeOffset]::UtcNow.ToString("o")
    foreach ($key in $Details.Keys) { $progress[$key] = $Details[$key] }
    $temporary = "$progressPath.$PID.tmp"
    try {
        [IO.File]::WriteAllText($temporary, ($progress | ConvertTo-Json) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $progressPath -PathType Leaf) {
            [IO.File]::Replace($temporary, $progressPath, [NullString]::Value)
        }
        else {
            [IO.File]::Move($temporary, $progressPath)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary }
    }
}

function Initialize-UploadCounters {
    if ("VisualAnalysisUploadIO" -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class VisualAnalysisUploadIO {
    [StructLayout(LayoutKind.Sequential)]
    public struct Counters { public ulong Reads, Writes, Others, ReadBytes, WriteBytes, OtherBytes; }
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetProcessIoCounters(IntPtr handle, out Counters counters);
}
'@
}

function Invoke-NetdiskUpload([IO.FileInfo]$File, [string]$Target) {
    Initialize-UploadCounters
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $cli
    $arguments = @("upload", $File.FullName, $Target) + $cliOptions
    if ($arguments -match '"') { throw "A Baidu upload argument contains an invalid quotation mark." }
    $start.Arguments = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join " "
    $start.StandardOutputEncoding = [Text.Encoding]::UTF8
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    Write-ReleaseProgress "uploading" @{
        file = $File.Name; current_file_bytes = $File.Length; process_read_bytes = 0;
        estimated_current_file_bytes = $null; estimated_remaining_seconds = $null
    }

    $process = [Diagnostics.Process]::Start($start)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $previousEstimate = $null
        $timer = [Diagnostics.Stopwatch]::StartNew()
        while (-not $process.WaitForExit(5000)) {
            $counters = [VisualAnalysisUploadIO+Counters]::new()
            if (-not [VisualAnalysisUploadIO]::GetProcessIoCounters($process.Handle, [ref]$counters)) { continue }
            $estimate = $null
            $eta = $null
            if ($counters.ReadBytes -ge $File.Length -and $File.Length -eq $zip.Length) {
                # The CLI first reads the ZIP for hashes. Later reads can also
                # include retries, so this is never a remote byte count.
                $estimate = [Math]::Min([long]$File.Length, [long]$counters.ReadBytes - $File.Length)
                if ($null -ne $previousEstimate -and $estimate -gt $previousEstimate) {
                    $eta = [Math]::Ceiling(($File.Length - $estimate) * $timer.Elapsed.TotalSeconds / ($estimate - $previousEstimate))
                }
            }
            Write-ReleaseProgress "uploading" @{
                process_read_bytes = $counters.ReadBytes;
                estimated_current_file_bytes = $estimate; estimated_remaining_seconds = $eta
            }
            $timer.Restart()
            $previousEstimate = $estimate
        }
        $raw = $stdout.GetAwaiter().GetResult()
        $null = $stderr.GetAwaiter().GetResult()
        $reply = $raw | ConvertFrom-Json
        if ($process.ExitCode -ne 0 -or $reply.code -ne 0) {
            throw "Baidu Netdisk command failed: upload (exit $($process.ExitCode))."
        }
        $reply
    }
    finally {
        if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
        $process.Dispose()
    }
}
