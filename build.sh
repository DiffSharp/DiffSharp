#!/usr/bin/env bash
git -C src/TorchSharp pull || git clone https://github.com/DiffSharp/TorchSharp.git src/TorchSharp
dotnet build