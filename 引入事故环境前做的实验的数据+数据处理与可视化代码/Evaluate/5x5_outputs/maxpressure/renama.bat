@echo off
setlocal enabledelayedexpansion

set "source_dir=%CD%"

for %%F in ("%source_dir%\*5x5*.csv") do (
    set "filename=%%~nxF"
    set "newname=!filename:_5x5=!"
    ren "%%F" "!newname!"
)

echo Renaming complete.
pause