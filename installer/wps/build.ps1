<#
.SYNOPSIS
    Builds the standalone XSTARS Windows WPS distribution package.

.DESCRIPTION
    1. Ensures PyInstaller is available in the repository .venv.
    2. Builds the frozen Python service and worker using installer/wps/xstars-wps.spec.
    3. Builds the official WPS add-in offline deployment package via wps-addon/scripts/build-offline-publish.cjs.
    4. Compiles XSTARS_WPS_Setup.exe using Inno Setup 6 (ISCC.exe).
    5. Computes SHA-256 checksum for the generated installer.

.PARAMETER SkipPyInstaller
    Skip the PyInstaller build step (uses existing dist/xstars-wps).

.PARAMETER SkipWpsAddon
    Skip building the WPS add-in offline package (uses existing deploy/).

.PARAMETER SkipISCC
    Skip running Inno Setup compiler.

.PARAMETER Clean
    Clean build and output directories before starting.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -SkipPyInstaller
#>

[CmdletBinding()]
param(
    [switch]$SkipPyInstaller,
    [switch]$SkipWpsAddon,
    [switch]$SkipISCC,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  XSTARS Windows WPS Installer Build Pipeline" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Repository root : $RepoRoot"
Write-Host "Script directory: $ScriptDir"

# Locate python executable in repository .venv
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python.exe"
}
Write-Host "Python runner   : $VenvPython"

# Locate Inno Setup Compiler (ISCC)
$IsccCandidates = @(
    "ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)
$IsccPath = $null
foreach ($cand in $IsccCandidates) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        $IsccPath = $cand
        break
    }
    if (Test-Path $cand) {
        $IsccPath = $cand
        break
    }
}

# Directories
$BuildDir = Join-Path $ScriptDir "build"
$DistDir = Join-Path $ScriptDir "dist"
$OutputDir = Join-Path $ScriptDir "output"
$SpecFile = Join-Path $ScriptDir "xstars-wps.spec"
$IssFile = Join-Path $ScriptDir "XSTARS_WPS.iss"
$WpsAddonDir = Join-Path $RepoRoot "wps-addon"

if ($Clean) {
    Write-Host "`n[Clean] Removing previous build directories..." -ForegroundColor Yellow
    Remove-Item -Path $BuildDir, $DistDir, $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
}

# 1. Read add-in version from wps-addon/package.json
$PackageJsonPath = Join-Path $WpsAddonDir "package.json"
$Version = "1.0.4"
if (Test-Path $PackageJsonPath) {
    $PackageJson = Get-Content $PackageJsonPath -Raw | ConvertFrom-Json
    if ($PackageJson.version) {
        $Version = $PackageJson.version
    }
}
Write-Host "Add-in version  : $Version"

# 2. PyInstaller build step
if (-not $SkipPyInstaller) {
    Write-Host "`n[Step 1/3] Building frozen Python service and worker via PyInstaller..." -ForegroundColor Green
    
    # Check if pyinstaller is installed in .venv
    $PyiCheck = & $VenvPython -c "import PyInstaller; print(PyInstaller.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller not found in environment, installing..." -ForegroundColor Yellow
        & $VenvPython -m pip install pyinstaller
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install PyInstaller into $VenvPython"
        }
    } else {
        Write-Host "Using PyInstaller version: $PyiCheck"
    }

    & $VenvPython -m PyInstaller --noconfirm --workpath $BuildDir --distpath $DistDir $SpecFile
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }
    Write-Host "PyInstaller build completed successfully: $DistDir\xstars-wps" -ForegroundColor Green
} else {
    Write-Host "`n[Step 1/3] Skipped PyInstaller build step (-SkipPyInstaller)" -ForegroundColor DarkGray
}

# 3. WPS Add-in offline publish build step
if (-not $SkipWpsAddon) {
    Write-Host "`n[Step 2/3] Building WPS official offline add-in package..." -ForegroundColor Green
    Push-Location $WpsAddonDir
    try {
        $NpmCmd = if (Get-Command "npm.cmd" -ErrorAction SilentlyContinue) { "npm.cmd" } else { "npm" }
        & $NpmCmd run publish:offline
        if ($LASTEXITCODE -ne 0) {
            throw "WPS add-in offline build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    Write-Host "WPS add-in offline packaging completed successfully." -ForegroundColor Green
} else {
    Write-Host "`n[Step 2/3] Skipped WPS add-in offline packaging (-SkipWpsAddon)" -ForegroundColor DarkGray
}

# 4. Inno Setup Compiler (ISCC) step
if (-not $SkipISCC) {
    Write-Host "`n[Step 3/3] Compiling Inno Setup installer..." -ForegroundColor Green
    if (-not $IsccPath) {
        throw "Inno Setup 6 compiler (ISCC.exe) not found. Please install Inno Setup 6."
    }
    Write-Host "Using ISCC compiler: $IsccPath"

    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    $SetupBaseName = "XSTARS_WPS_Setup_v$Version"
    & "$IsccPath" "/DAppVersion=$Version" "/O$OutputDir" "/F$SetupBaseName" $IssFile
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compilation failed with exit code $LASTEXITCODE"
    }

    $TargetExe = Join-Path $OutputDir "$SetupBaseName.exe"
    if (Test-Path $TargetExe) {
        $FileStream = [System.IO.File]::OpenRead($TargetExe)
        $Sha256 = [System.Security.Cryptography.SHA256]::Create()
        $HashBytes = $Sha256.ComputeHash($FileStream)
        $FileStream.Close()
        $Hash = -join ($HashBytes | ForEach-Object { "{0:X2}" -f $_ })
        $ChecksumFile = Join-Path $OutputDir "$SetupBaseName.sha256.txt"
        "$Hash *$SetupBaseName.exe" | Out-File -FilePath $ChecksumFile -Encoding ascii
        
        $Item = Get-Item $TargetExe
        $SizeMB = [math]::Round($Item.Length / 1MB, 2)

        Write-Host "`n====================================================" -ForegroundColor Cyan
        Write-Host "  Build Succeeded!" -ForegroundColor Green
        Write-Host "====================================================" -ForegroundColor Cyan
        Write-Host "Output installer : $TargetExe ($SizeMB MB)"
        Write-Host "SHA-256 Checksum : $Hash"
        Write-Host "Checksum file    : $ChecksumFile"
    }
} else {
    Write-Host "`n[Step 3/3] Skipped Inno Setup build step (-SkipISCC)" -ForegroundColor DarkGray
}
