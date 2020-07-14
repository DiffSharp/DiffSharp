

# Running notebooks in MyBinder

The `Dockerfile` and `NuGet.config` allow us to run generated notebooks in [MyBinder](https://mybinder.org)

* `gh-pages` branch of dsyme/DiffSharp:  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dsyme/DiffSharp/gh-pages)

* `index.ipynb` for `dev` branch of dsyme/DiffSharp: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dsyme/DiffSharp/gh-pages?filepath=index.ipynb)

# Generating docs

This published version of the docs can be refreshed by these commands:

1. Prepare one off

    dotnet tool restore
    git clone https://github.com/dsyme/DiffSharp ../DiffSharp-docs -b gh-pages --depth 1

2. Build

    dotnet build
    dotnet fsdocs build --clean --output ../DiffSharp-docs
    bash -c "(cd ../DiffSharp-docs && git add . && git commit -a -m doc-update && git push -f https://github.com/dsyme/DiffSharp gh-pages)"

To use a local builg of FSharp.Formatting:

       git clone https://github.com/fsprojects/FSharp.Formatting  ../FSharp.Formatting
       cd ..\FSharp.Formatting
       .\build

# How it works

Notebooks are generated for all .md and .fsx files under docs as part of the build.

* Dockerfile - see https://github.com/dotnet/interactive/blob/master/docs/CreateBinder.md

* NuGet.config - likewise

See MyBinder for creating URLs
