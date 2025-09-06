@echo off
setlocal

:: Default VST3 installation paths for Windows
set "DEST_USER=%USERPROFILE%\Documents\VST3"
set "DEST_SYSTEM=C:\Program Files\Common Files\VST3"
set "DEST=%DEST_USER%"
set "DRY_RUN=0"
set "CODESIGN=0" :: Codesigning on Windows is more involved and not directly comparable to ad-hoc macOS codesign.
                    :: This option is kept for consistency but won't perform actual codesigning here.

:: Parse command-line arguments
call :parse_args %*


:: Get the directory of the script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Function to print usage
:print_usage
echo Usage: %~nx0 [--system^|--user] [--dry-run] [--codesign]
echo.
echo Build the VST3 plugins and install them to the standard Windows VST3 folder.
echo.
echo Options:
echo   --system   Install to C:\Program Files\Common Files\VST3 (requires administrator privileges)
echo   --user     Install to %USERPROFILE%\Documents\VST3 (default)
echo   --dry-run  Show actions without copying files
echo   --codesign (Not implemented for Windows in this script)
echo   -h, --help Show this help and exit
echo.
goto :eof

:: Parse command-line arguments
:parse_args
if "%~1"=="" goto :args_done

if /i "%~1"=="--system" (
    set "DEST=%DEST_SYSTEM%"
) else if /i "%~1"=="--user" (
    set "DEST=%DEST_USER%"
) else if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
) else if /i "%~1"=="--codesign" (
    set "CODESIGN=1"
) else if /i "%~1"=="-h" (
    call :print_usage
    exit /b 0
) else if /i "%~1"=="--help" (
    call :print_usage
    exit /b 0
) else (
    echo Unknown option: %1 1>&2
    call :print_usage 1>&2
    exit /b 2
)
shift
goto :parse_args
:args_done@echo off
setlocal

:: Default VST3 installation paths for Windows
set "DEST_USER=%USERPROFILE%\Documents\VST3"
set "DEST_SYSTEM=C:\Program Files\Common Files\VST3"
set "DEST=%DEST_USER%"
set "DRY_RUN=0"
set "CODESIGN=0" :: Codesigning on Windows is more involved and not directly comparable to ad-hoc macOS codesign.
                  :: This option is kept for consistency but won't perform actual codesigning here.

:: Function to print usage
:print_usage
echo Usage: %~nx0 [--system^|--user] [--dry-run] [--codesign]
echo.
echo Build the VST3 plugins and install them to the standard Windows VST3 folder.
echo.
echo Options:
echo   --system   Install to C:\Program Files\Common Files\VST3 (requires administrator privileges)
echo   --user     Install to %USERPROFILE%\Documents\VST3 (default)
echo   --dry-run  Show actions without copying files
echo   --codesign (Not implemented for Windows in this script)
echo   -h, --help Show this help and exit
echo.
goto :eof

:: Parse command-line arguments
:parse_args
if "%~1"=="" goto :args_done

if /i "%~1"=="--system" (
    set "DEST=%DEST_SYSTEM%"
    shift
    goto :parse_args
)

if /i "%~1"=="--user" (
    set "DEST=%DEST_USER%"
    shift
    goto :parse_args
)

if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_args
)

if /i "%~1"=="--codesign" (
    set "CODESIGN=1"
    shift
    goto :parse_args
)

if /i "%~1"=="-h" (
    call :print_usage
    exit /b 0
)

if /i "%~1"=="--help" (
    call :print_usage
    exit /b 0
)

echo Unknown option: %1 1>&2
call :print_usage 1>&2
exit /b 2

:args_done

:: Get the directory of the script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Building and bundling VST3 plugins in workspace at: %SCRIPT_DIR%

if "%DRY_RUN%"=="0" (
    echo cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release
    cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release || (echo Error running xtask for FX; exit /b 1)
    echo cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release
    cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release || (echo Error running xtask for Instrument; exit /b 1)
) else (
    echo [dry-run] cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release 1>&2
    echo [dry-run] cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release 1>&2
)

set "TARGET_DIR=%SCRIPT_DIR%target\bundled"

:: Function to find bundle paths
:: This uses PowerShell for more robust pattern matching, similar to `find` on Linux.
:find_bundle
set "BUNDLE_PATH="
set "SEARCH_PATTERN=%1"
set "TEMP_BUNDLE_PATH="
for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-ChildItem -Path '%TARGET_DIR%' -Directory -Filter '%SEARCH_PATTERN%*.vst3' | Select-Object -First 1 -ExpandProperty FullName"`) do (
    set "TEMP_BUNDLE_PATH=%%i"
)
set "BUNDLE_PATH=%TEMP_BUNDLE_PATH%"
goto :eof

:: Find FX Bundle
call :find_bundle "node_audio_vst_fx"
set "FX_BUNDLE=%BUNDLE_PATH%"
if not exist "%FX_BUNDLE%" (
    :: Fallback: search by plugin display names
    for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-ChildItem -Path '%TARGET_DIR%' -Directory -Filter '*NodeAudio*FX*.vst3' | Select-Object -First 1 -ExpandProperty FullName"`) do (
        set "FX_BUNDLE=%%i"
    )
)

:: Find Instrument Bundle
call :find_bundle "node_audio_vst_instrument"
set "INST_BUNDLE=%BUNDLE_PATH%"
if not exist "%INST_BUNDLE%" (
    :: Fallback: search by plugin display names
    for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-ChildItem -Path '%TARGET_DIR%' -Directory -Filter '*NodeAudio*Instrument*.vst3' | Select-Object -First 1 -ExpandProperty FullName"`) do (
        set "INST_BUNDLE=%%i"
    )
)

if not exist "%FX_BUNDLE%" (
    echo Error: Could not find bundled FX plugin in %TARGET_DIR% 1>&2
    exit /b 1
)
if not exist "%INST_BUNDLE%" (
    echo Error: Could not find bundled Instrument plugin in %TARGET_DIR% 1>&2
    exit /b 1
)

echo Found bundled plugins:
echo   FX:         %FX_BUNDLE%
echo   Instrument: %INST_BUNDLE%

echo Installing to: %DEST%

:: Check for administrator privileges if installing to system path
set "REQUIRES_ADMIN=0"
if /i "%DEST%"=="%DEST_SYSTEM%" (
    :: Attempt to create a directory in a system path to test for admin rights
    :: Redirect stderr to NUL to suppress "Access is denied" messages
    mkdir "%DEST%\test_admin_$$" >nul 2>&1
    if exist "%DEST%\test_admin_$$" (
        rmdir "%DEST%\test_admin_$$"
    ) else (
        echo You are attempting to install to a system-wide path (%DEST%).
        echo This operation may require administrator privileges.
        set "REQUIRES_ADMIN=1"
    )
)

:: Function to install a bundle
:install_bundle
set "BUNDLE_SRC=%1"
set "DEST_DIR=%2"
for %%f in ("%BUNDLE_SRC%") do set "BUNDLE_NAME=%%~nxf"

if "%DRY_RUN%"=="0" (
    if "%REQUIRES_ADMIN%"=="1" (
        echo Launching with administrator privileges to install "%BUNDLE_NAME%"...
        powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/c \"del /s /q \"%DEST_DIR%\%BUNDLE_NAME%\" >nul 2>&1 & xcopy /e /i /y \"%BUNDLE_SRC%\" \"%DEST_DIR%\%BUNDLE_NAME%\\\"\"'"
        if errorlevel 1 (
            echo Failed to install "%BUNDLE_NAME%" with administrator privileges. Aborting. 1>&2
            exit /b 1
        )
    ) else (
        echo rmdir /s /q "%DEST_DIR%\%BUNDLE_NAME%"
        rmdir /s /q "%DEST_DIR%\%BUNDLE_NAME%" >nul 2>&1
        echo xcopy /e /i /y "%BUNDLE_SRC%" "%DEST_DIR%\%BUNDLE_NAME%\" 
        xcopy /e /i /y "%BUNDLE_SRC%" "%DEST_DIR%\%BUNDLE_NAME%\" || (echo Error copying "%BUNDLE_NAME%"; exit /b 1)
    )
) else (
    echo [dry-run] rmdir /s /q "%DEST_DIR%\%BUNDLE_NAME%"
    echo [dry-run] xcopy /e /i /y "%BUNDLE_SRC%" "%DEST_DIR%\%BUNDLE_NAME%\" 
)
goto :eof

:: Create destination directory if it doesn't exist
if "%DRY_RUN%"=="0" (
    if not exist "%DEST%" (
        if "%REQUIRES_ADMIN%"=="1" (
            echo Launching with administrator privileges to create "%DEST%"...
            powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/c \"mkdir \"%DEST%\"\"'"
            if errorlevel 1 (
                echo Failed to create directory "%DEST%" with administrator privileges. Aborting. 1>&2
                exit /b 1
            )
        ) else (
            echo mkdir "%DEST%"
            mkdir "%DEST%" || (echo Error creating directory "%DEST%"; exit /b 1)
        )
    )
) else (
    echo [dry-run] mkdir "%DEST%"
)

call :install_bundle "%FX_BUNDLE%" "%DEST%"
call :install_bundle "%INST_BUNDLE%" "%DEST%"

echo Done. If your DAW is open, rescan plugins or restart it.

endlocal
exit /b 0


:: Get the directory of the script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Building and bundling VST3 plugins in workspace at: %SCRIPT_DIR%

if "%DRY_RUN%"=="0" (
    echo cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release
    cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release || (echo Error running xtask for FX; exit /b 1)
    echo cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release
    cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release || (echo Error running xtask for Instrument; exit /b 1)
) else (
    echo [dry-run] cargo run --package xtask --release -- bundle -p node_audio_vst_fx --release 1>&2
    echo [dry-run] cargo run --package xtask --release -- bundle -p node_audio_vst_instrument --release 1>&2
)

set "TARGET_DIR=%SCRIPT_DIR%target\bundled"

:: Function to find bundle paths
:: This uses PowerShell for more robust pattern matching, similar to `find` on Linux.
:find_bundle
set "BUNDLE_PATH="
set "SEARCH_PATTERN=%1"
set "TEMP_BUNDLE_PATH="
for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-ChildItem -Path '%TARGET_DIR%' -Directory -Filter '%SEARCH_PATTERN%*.vst3' | Select-Object -First 1 -ExpandProperty FullName"`) do (
    set "TEMP_BUNDLE_PATH=%%i"
)
set "BUNDLE_PATH=%TEMP_BUNDLE_PATH%"
goto :eof


:: Find FX Bundle
call :find_bundle "node_audio_vst_fx"
set "FX_BUNDLE=%BUNDLE_PATH%"
if not exist "%FX_BUNDLE%" (
    :: Fallback: search by plugin display names
    for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-ChildItem -Path '%TARGET_DIR%' -Directory -Filter '*NodeAudio*FX*.vst3' | Select-Object -First 1 -ExpandProperty FullName"`) do (
        set "FX_BUNDLE=%%i"
    )
)

:: Find Instrument Bundle
call :find_bundle "node_audio_vst_instrument"
set "INST_BUNDLE=%BUNDLE_PATH%"
if not exist "%INST_BUNDLE%" (
    :: Fallback: search by plugin display names
    for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-ChildItem -Path '%TARGET_DIR%' -Directory -Filter '*NodeAudio*Instrument*.vst3' | Select-Object -First 1 -ExpandProperty FullName"`) do (
        set "INST_BUNDLE=%%i"
    )
)

if not exist "%FX_BUNDLE%" (
    echo Error: Could not find bundled FX plugin in %TARGET_DIR% 1>&2
    exit /b 1
)
if not exist "%INST_BUNDLE%" (
    echo Error: Could not find bundled Instrument plugin in %TARGET_DIR% 1>&2
    exit /b 1
)

echo Found bundled plugins:
echo   FX:         %FX_BUNDLE%
echo   Instrument: %INST_BUNDLE%

echo Installing to: %DEST%

:: Check for administrator privileges if installing to system path
set "REQUIRES_ADMIN=0"
if /i "%DEST%"=="%DEST_SYSTEM%" (
    :: Attempt to create a directory in a system path to test for admin rights
    :: Redirect stderr to NUL to suppress "Access is denied" messages
    mkdir "%DEST%\test_admin_$$" >nul 2>&1
    if exist "%DEST%\test_admin_$$" (
        rmdir "%DEST%\test_admin_$$"
    ) else (
        echo You are attempting to install to a system-wide path (%DEST%).
        echo This operation may require administrator privileges.
        set "REQUIRES_ADMIN=1"
    )
)

:: Function to install a bundle
:install_bundle
set "BUNDLE_SRC=%1"
set "DEST_DIR=%2"
for %%f in ("%BUNDLE_SRC%") do set "BUNDLE_NAME=%%~nxf"

if "%DRY_RUN%"=="0" (
    if "%REQUIRES_ADMIN%"=="1" (
        echo Launching with administrator privileges to install "%BUNDLE_NAME%"...
        powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/c \"del /s /q \"%DEST_DIR%\%BUNDLE_NAME%\" >nul 2>&1 & xcopy /e /i /y \"%BUNDLE_SRC%\" \"%DEST_DIR%\%BUNDLE_NAME%\\\"\"'"
        if errorlevel 1 (
            echo Failed to install "%BUNDLE_NAME%" with administrator privileges. Aborting. 1>&2
            exit /b 1
        )
    ) else (
        echo rmdir /s /q "%DEST_DIR%\%BUNDLE_NAME%"
        rmdir /s /q "%DEST_DIR%\%BUNDLE_NAME%" >nul 2>&1
        echo xcopy /e /i /y "%BUNDLE_SRC%" "%DEST_DIR%\%BUNDLE_NAME%\"
        xcopy /e /i /y "%BUNDLE_SRC%" "%DEST_DIR%\%BUNDLE_NAME%\" || (echo Error copying "%BUNDLE_NAME%"; exit /b 1)
    )
) else (
    echo [dry-run] rmdir /s /q "%DEST_DIR%\%BUNDLE_NAME%"
    echo [dry-run] xcopy /e /i /y "%BUNDLE_SRC%" "%DEST_DIR%\%BUNDLE_NAME%\"
)
goto :eof

:: Create destination directory if it doesn't exist
if "%DRY_RUN%"=="0" (
    if not exist "%DEST%" (
        if "%REQUIRES_ADMIN%"=="1" (
            echo Launching with administrator privileges to create "%DEST%"...
            powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/c \"mkdir \"%DEST%\"\"'"
            if errorlevel 1 (
                echo Failed to create directory "%DEST%" with administrator privileges. Aborting. 1>&2
                exit /b 1
            )
        ) else (
            echo mkdir "%DEST%"
            mkdir "%DEST%" || (echo Error creating directory "%DEST%"; exit /b 1)
        )
    )
) else (
    echo [dry-run] mkdir "%DEST%"
)


call :install_bundle "%FX_BUNDLE%" "%DEST%"
call :install_bundle "%INST_BUNDLE%" "%DEST%"

echo Done. If your DAW is open, rescan plugins or restart it.

endlocal
exit /b 0