#!/usr/bin/env -S dotnet fsi

#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"
// #r "nuget: libtorch-cuda-10.2-linux-x64, 1.7.0.1"
System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim
open DiffSharp.Util
open DiffSharp.Data
open System.Net
open System.IO

dsharp.config(backend=Backend.Torch, device=Device.CPU)

download "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt" "./shakespeare.txt"
