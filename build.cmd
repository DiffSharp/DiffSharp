@echo off
cls

REM dotnet --version
dotnet --version

dotnet build DiffSharp.sln -c release -v:n
dotnet test tests/DiffSharp.Tests  -c release -f netcoreapp2.0 -v:n
