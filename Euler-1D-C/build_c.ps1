param(
    [switch]$Profile
)

$ErrorActionPreference = 'Stop'

# Move to the folder containing this script.
Set-Location -Path $PSScriptRoot

$commonFlags = @('-march=native', '-ffast-math', '-fopenmp', '-std=c11')
if ($Profile) {
    Write-Host 'Profiling build enabled (-pg instrumentation).'
    # On Windows/MinGW, ASLR can prevent gprof from mapping samples to symbols.
    # Keep profiling builds less aggressively optimized to preserve function boundaries.
    $compileFlags = @('-O3', '-g', '-fno-omit-frame-pointer', '-fno-inline', '-pg') + $commonFlags
    $linkFlags = @('-fopenmp', '-pg', '-Wl,--disable-dynamicbase', '-Wl,--disable-high-entropy-va')
} else {
    $compileFlags = @('-O3') + $commonFlags
    $linkFlags = @('-fopenmp')
}

Write-Host "[1/4] Cleaning previous C build artifacts..."
if (-not (Test-Path -Path 'build')) {
    New-Item -ItemType Directory -Path 'build' | Out-Null
}

Remove-Item -Path 'build\*.o' -Force -ErrorAction SilentlyContinue
Remove-Item -Path '*.exe' -Force -ErrorAction SilentlyContinue

Write-Host "[2/4] Cleaning previous C simulation output files..."
$caseDirs = Get-ChildItem -Path '.' -Directory -Filter 'cfl*_c' -ErrorAction SilentlyContinue
foreach ($dir in $caseDirs) {
    Remove-Item -Path (Join-Path $dir.FullName 'final_profiles_case*.csv') -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $dir.FullName 'geometry_history_case*.csv') -Force -ErrorAction SilentlyContinue
}

Write-Host "[3/4] Compiling C sources with OpenMP..."

& gcc @compileFlags -c input_vegaE.c -o build\input_vegaE.o
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Error 'Build failed. Check errors above.'
    exit 1
}

& gcc @compileFlags -c riemann_test_cases.c -o build\riemann_test_cases.o
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Error 'Build failed. Check errors above.'
    exit 1
}

& gcc @compileFlags -c Rocket_tester_seq.c -o build\Rocket_tester_seq.o
& gcc @compileFlags -c Rocket_tester_parallel.c -o build\Rocket_tester_parallel.o
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Error 'Build failed. Check errors above.'
    exit 1
}

& gcc @linkFlags build\input_vegaE.o build\riemann_test_cases.o build\Rocket_tester_seq.o -lm -o rocket_tester_seq.exe
& gcc @linkFlags build\input_vegaE.o build\riemann_test_cases.o build\Rocket_tester_parallel.o -lm -o rocket_tester_parallel.exe
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Error 'Build failed. Check errors above.'
    exit 1
}

Write-Host "[4/4] Build completed: rocket_tester_seq.exe and rocket_tester_parallel.exe"
Write-Host ''
Write-Host 'Optional run example:'
Write-Host '  $env:OMP_NUM_THREADS = 8'
Write-Host '  .\rocket_tester_seq.exe'
Write-Host '  .\rocket_tester_parallel.exe'

if ($Profile) {
    Write-Host ''
    Write-Host 'Profiling workflow (gprof):'
    Write-Host '  1) For cleaner CPU hotspots, use one thread:'
    Write-Host '     $env:OMP_NUM_THREADS = 1'
    Write-Host '  2) Run the program to generate gmon.out:'
    Write-Host '     .\rocket_tester_seq.exe'
    Write-Host '  3) Build a readable report:'
    Write-Host '     gprof .\rocket_tester_seq.exe .\gmon.out > .\gprof_report.txt'
    Write-Host '  4) Open report and inspect hottest functions first:'
    Write-Host '     Get-Content .\gprof_report.txt -TotalCount 120'
}
