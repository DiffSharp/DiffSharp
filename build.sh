#!/bin/bash

case "$(uname -s)" in

   Darwin)
     brew install openblas
     ;;

   CYGWIN*|MINGW32*|MSYS*|Linux)
    echo 'Linux, Cygwin, etc' 
#    sudo apt-get -y update
#    sudo apt-get -y install libopenblas-dev
     ;;

   *)
     echo 'other OS' 
     ;;
esac

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/OpenBLAS-v0.2.8-Linux64;/usr/local/lib/OpenBLAS-v0.2.8-Linux64

if [ -d "MONO" ]; then
   msbuild DiffSharp.sln /p:Configuration=Release
   msbuild /p:Configuration=Release DiffSharp.sln
   # not currently testing on mono
   # mono ./packages/NUnit.Runners/tools/nunit-console.exe ./tests/DiffSharp.Tests/bin/Release/DiffSharp.Tests.dll
else
   dotnet --version
   dotnet build DiffSharp.sln -c debug 
   dotnet test tests/DiffSharp.Tests  -c release -f netcoreapp2.0
   dotnet pack DiffSharp.sln -c release
fi

