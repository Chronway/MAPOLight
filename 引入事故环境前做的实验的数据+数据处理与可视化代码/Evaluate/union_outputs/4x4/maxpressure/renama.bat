@echo off
setlocal enabledelayedexpansion

set "source_dir=%CD%"

for %%F in ("%source_dir%\*4x4*.csv") do (
    set "filename=%%~nxF"
    set "newname=!filename:_4x4=!"
    ren "%%F" "!newname!"
)

echo Renaming complete.
pause