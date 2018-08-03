@echo off
cls

dotnet build DiffSharp.sln -c debug 
dotnet test tests/DiffSharp.Tests  -c debug 
