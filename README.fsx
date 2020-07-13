(**
# Running notebooks in MyBinder

The `Dockerfile` and `NuGet.config` allow us to run generated notebooks in [MyBinder](https://mybinder.org)

Since the generated docs are not yet "up", a recent version may be found in the gh-pages branch of dsyme/DiffSharp:

* `gh-pages` branch of dsyme/DiffSharp:  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dsyme/DiffSharp/gh-pages)

* `index.ipynb` for `dev` branch of dsyme/DiffSharp: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dsyme/DiffSharp/gh-pages?filepath=notebooks/index.ipynb)

This published version of the docs can be refreshed by

1. Build FSharp.Formatting

       git clone https://github.com/fsprojects/FSharp.Formatting  ../tmp/FSharp.Formatting
       cd ..\FSharp.Formatting
       .\build

2. Generate Docs (after building)

    rmdir /s /q ..\tmp
    git clone https://github.com/DiffSharp/DiffSharp ../tmp/DiffSharp-docs -b tmpdocs --depth 1
    pushd ..\tmp\DiffSharp-docs
    git rm -fr *
    popd ..\..\DiffSharp
    ..\FSharp.Formatting\src\FSharp.Formatting.CommandTool\bin\Debug\netcoreapp3.1\fsdocs.exe build --output ..\tmp\DiffSharp-docs
    pushd ..\tmp\DiffSharp-docs
    git add .
    git commit -a -m "commit docs"
    git push -f https://github.com/DiffSharp/DiffSharp tmpdocs
    git push -f https://github.com/dsyme/DiffSharp tmpdocs
    popd ..\..\DiffSharp


Typical URL and badges when docs are published:

* `gh-pages` branch of DiffSharp/DiffSharp:  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiffSharp/DiffSharp/gh-pages)

* `index.ipynb` for `dev` branch of DiffSharp/DiffSharp: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiffSharp/DiffSharp/gh-pages?filepath=notebooks/index.ipynb)



# How it works

Notebooks are generated for all .md and .fsx files under docs as part of the build.

* Dockerfile - see https://github.com/dotnet/interactive/blob/master/docs/CreateBinder.md

* NuGet.config - likewise

See MyBinder for creating URLs

*)
