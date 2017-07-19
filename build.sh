mono .paket/paket.bootstrapper.exe && \
mono .paket/paket.exe restore && \
msbuild /p:Configuration=Release DiffSharp.sln && \
mono ./packages/NUnit.Runners/tools/nunit-console.exe ./tests/DiffSharp.Tests/bin/Release/DiffSharp.Tests.dll