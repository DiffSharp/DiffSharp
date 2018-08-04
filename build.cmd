@echo off
cls

REM dotnet --version
dotnet --version

dotnet build DiffSharp.sln -c debug 
dotnet test tests/DiffSharp.Tests  -c debug 
