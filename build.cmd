@echo off
cls

.paket\paket.bootstrapper.exe
if errorlevel 1 (
  exit /b %errorlevel%
)

.paket\paket.exe restore
if errorlevel 1 (
  exit /b %errorlevel%
)
msbuild DiffSharp.sln
nunit-console "tests\DiffSharp.Tests\bin\Debug\DiffSharp.Tests.dll" 
