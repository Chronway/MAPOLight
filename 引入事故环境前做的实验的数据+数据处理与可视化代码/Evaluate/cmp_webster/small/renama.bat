@echo off
setlocal enabledelayedexpansion

set "source_dir=%CD%"
set "name=small"

for %%F in ("%source_dir%\*%name%*.csv") do (
    set "filename=%%~nxF"
    set "newname=!filename:_%name%=!"
    ren "%%F" "!newname!"
)
