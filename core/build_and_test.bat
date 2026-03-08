@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "C:\Users\Public\Tarsfull\TarsSSM-Py\core"

echo === Compiling TARS Core test_kernels.exe ===
cl /O2 /arch:AVX2 /fp:fast /EHsc /DTARS_HAS_AVX2=1 ^
    /I"C:\Users\Public\Tarsfull\TarsSSM-Py\core" ^
    /Fe:test_kernels.exe ^
    kernels\rmsnorm_fused.cpp ^
    kernels\embedding_lookup.cpp ^
    kernels\bitnet_matmul.cpp ^
    kernels\softmax_fused.cpp ^
    runtime\arena.cpp ^
    tests\test_kernels.cpp

if %ERRORLEVEL% NEQ 0 (
    echo COMPILATION FAILED
    exit /b 1
)

echo === Running tests ===
test_kernels.exe
