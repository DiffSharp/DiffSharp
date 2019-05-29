cls

dotnet --version
dotnet build DiffSharp.sln -c release -v:n
dotnet test tests/DiffSharp.Tests  -c release -v:n
dotnet pack DiffSharp.sln -c release