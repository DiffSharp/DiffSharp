#!/usr/bin/env -S dotnet fsi

#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
// #r "nuget: libtorch-cuda-11.1-win-x64, 1.8.0.7"
#r "nuget: libtorch-cuda-11.1-linux-x64, 1.8.0.7"
// Option B: you can use a local libtorch installation
// System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open DiffSharp
open DiffSharp.Model
open DiffSharp.Compose
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.Util

dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(0)


let classifier =
    Conv2d(1, 32, 3, 2)
    --> dsharp.relu
    --> Conv2d(32, 64, 3, 2)
    --> dsharp.relu
    --> dsharp.maxpool2d(2)
    --> dsharp.dropout(0.25)
    --> dsharp.flatten(1)
    --> Linear(576, 128)
    --> dsharp.relu
    --> dsharp.dropout(0.5)
    --> Linear(128, 10)
    --> dsharp.logsoftmax(dim=1)

let epochs = 2
let batchSize = 64
let numSamples = 32

let urls = ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"]

let trainSet = MNIST("../data", urls=urls, train=true, transform=id, n=200)
let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)
let validSet = MNIST("../data", urls=urls, train=false, transform=id, n=10)
let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)


printfn "Model: %A" classifier

let optimizer = Adam(classifier, lr=dsharp.tensor(0.001))

for epoch = 1 to epochs do
    for i, data, target in trainLoader.epoch() do
        classifier.reverseDiff()
        let output = data --> classifier
        let l = dsharp.nllLoss(output, target)
        l.reverse()
        optimizer.step()
        printfn "Epoch: %A/%A minibatch: %A/%A loss: %A" epoch epochs i trainLoader.length (float(l))

    classifier.noDiff()
    let mutable validLoss = dsharp.zero()
    for j, data, target in validLoader.epoch() do
        print j
        let output = data --> classifier
        validLoss <- validLoss + dsharp.nllLoss(output, target, reduction="sum")
    validLoss <- validLoss / validSet.length
    printfn "Validation loss: %A" (float validLoss)