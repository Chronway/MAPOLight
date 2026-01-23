@echo off
setlocal enabledelayedexpansion

set "source_dir=%CD%"

for %%F in ("%source_dir%\*real*.csv") do (
    set "filename=%%~nxF"
    set "newname=!filename:_real=!"
    ren "%%F" "!newname!"
)

echo Renaming complete.
pause