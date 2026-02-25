# OmegaCore Build Script — Pure C++ (No PyTorch, No PyBind11)
# Requires: Zig 0.16+ (as C++ compiler only)
# Output: omega_core.dll (Windows) / libomega_core.so (Linux)

$ZIG = "zig"
$CORE_DIR = $PSScriptRoot

Write-Host "`n--- TARS OmegaCore: Pure C++ Build ---" -ForegroundColor Cyan

Push-Location $CORE_DIR

# Single clean command — no torch includes, no python includes, no lib linking
& $ZIG c++ -shared -O3 `
    -std=c++17 `
    -mavx2 -mfma `
    -target x86_64-windows-gnu `
    -static-libgcc -static-libstdc++ `
    omega_core_pure.cpp `
    -o omega_core.dll

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] omega_core.dll compiled." -ForegroundColor Green
    Get-Item omega_core.dll | Select-Object Name, Length, LastWriteTime
} else {
    Write-Host "[ERROR] Build failed ($LASTEXITCODE). Check zig installation." -ForegroundColor Red
}

Pop-Location
