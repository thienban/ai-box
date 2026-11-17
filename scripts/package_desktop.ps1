param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$appDir = Join-Path $root "app"
$desktopDir = Join-Path $root "desktop"
$binDir = Join-Path $desktopDir "src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

Write-Host "1/3 Compilation release de aibox.exe..."
& cargo build --release --manifest-path (Join-Path $appDir "Cargo.toml") -p aibox
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$aiboxExe = Join-Path $appDir "target\release\aibox.exe"
$sidecarTarget = Join-Path $binDir "aibox-x86_64-pc-windows-msvc.exe"
Copy-Item $aiboxExe $sidecarTarget -Force

$gpuDll = Join-Path $appDir "target\release\aibox_gpu.dll"
if (Test-Path $gpuDll) {
    Copy-Item $gpuDll (Join-Path $binDir "aibox_gpu.dll") -Force
}

Write-Host "2/3 Installation et build frontend desktop..."
Push-Location $desktopDir
try {
    & pnpm install
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & pnpm build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not $SkipInstaller) {
        Write-Host "3/3 Packaging installeur Tauri v2 (NSIS / MSI)..."
        & pnpm tauri build
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} finally {
    Pop-Location
}
