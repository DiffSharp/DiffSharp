# DiffSharp 

[![Build Status](https://travis-ci.org/DiffSharp/DiffSharp.svg?branch=dev)](https://travis-ci.org/DiffSharp/DiffSharp)
[![codecov](https://codecov.io/gh/DiffSharp/DiffSharp/branch/dev/graph/badge.svg)](https://codecov.io/gh/DiffSharp/DiffSharp)

This is the development branch of DiffSharp 1.0.0.

You can clone this repository to your machine as follows:
```
git clone --branch dev https://github.com/DiffSharp/DiffSharp.git
cd DiffSharp
```

## Run tests

Required:
- Install [.NET Core SDK](https://dotnet.microsoft.com/download) for your system

Use the following command in the root directory of this repository:
```
dotnet test
```

## Build DiffSharp in Docker

Required:
- Install [Docker](https://hub.docker.com/search/?type=edition&offering=community) for your system

Build a Docker image called `diffsharp`. This will work without any local .NET Core installation and build DiffSharp inside the image.
```
docker build -t diffsharp .
```

Use the following to instantiate a Docker container from the `diffsharp` image and run the tests inside:
```
docker run --rm diffsharp dotnet test
```

## Using local TorchSharp packages


To add features you may have extend TorchSharp to make extra features of LibTorch available.

The build is set up to look for a parallel build of TorchSharp, e.g.

    C:\GitHub\dsyme\DiffSharp
    C:\GitHub\dsyme\TorchSharp

To build, test and pack TorchSharp in that repo do this:

    .\build build
    .\build test
    .\build pack

You will see something like this

    Packing LibTorch.Cuda.10.2.Redist nupkg (takes a long time!)...
    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg'.
    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/LibTorch.Cuda.10.2.Redist.0.3.0-local-Debug-20200520.nupkg'.
    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/LibTorch.Redist.0.3.0-local-Debug-20200520.nupkg'.

with warning:

    warning : Packages will be incomplete and unusable on other platforms...

To consume the packages into DiffSHarp adust TorchSharpVersion in DIrectory.Build.props.

When rebuilding the TorchSHarp you will need to vlear your package cache to pick up the new nuget package with the same version id, e.g.

    rmdir /q /s %USERPROFILE%\.nuget\packages\torchsharp
    rmdir /q /s %USERPROFILE%\.nuget\packages\LibTorch.Redist
    rmdir /q /s %USERPROFILE%\.nuget\packages\LibTorch.Cuda.10.2.Redist
    dotnet restore

The LibTorch packages are quite large and you may need to watch disk space.




