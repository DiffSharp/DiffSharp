

# Running notebooks in MyBinder

The `Dockerfile` and `NuGet.config` allow us to run generated notebooks in [MyBinder](https://mybinder.org)

* `master` branch of diffsharp/diffsharp.github.io:  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master)

# Generating docs


To Iterate on Literate Docs and API Docs (requires evaluation off since DLLs get locked)

    dotnet fsdocs watch 

To use a local build of FSharp.Formatting:

       git clone https://github.com/fsprojects/FSharp.Formatting  ../FSharp.Formatting
       pushd ..\FSharp.Formatting
       .\build
       popd

Then:

       ..\FSharp.Formatting\src\FSharp.Formatting.CommandTool\bin\Debug\net5.0\fsdocs.exe watch --eval
       ..\FSharp.Formatting\src\FSharp.Formatting.CommandTool\bin\Debug\net5.0\fsdocs.exe build --clean --eval

## Generated Notebooks

Notebooks are generated for all .md and .fsx files under docs as part of the build.

* Dockerfile - see https://github.com/dotnet/interactive/blob/master/docs/CreateBinder.md

* NuGet.config - likewise

See MyBinder for creating URLs
