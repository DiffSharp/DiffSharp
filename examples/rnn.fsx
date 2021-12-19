#!/usr/bin/env -S dotnet fsi

#I "../tests/DiffSharp.Tests/bin/Debug/net6.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
#r "nuget: TorchSharp-cpu, 0.93.5"
// #r "nuget: TorchSharp-cuda-linux, 0.93.5"
// #r "nuget: TorchSharp-cuda-windows, 0.93.5"
// Option B: you can use a local libtorch installation
// System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim
open DiffSharp.Util
open DiffSharp.Distributions

open System.IO

dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(1)


// let corpus = "A merry little surge of electricity piped by automatic alarm from the mood organ beside his bed awakened Rick Deckard."
download "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt" "./shakespeare.txt"
let corpus = System.IO.File.ReadAllText("./shakespeare.txt")

let seqLen = 32
let batchSize = 16

let dataset = TextDataset(corpus, seqLen)
let loader = dataset.loader(batchSize=batchSize, shuffle=true)

let rnn = RNN(dataset.numChars, 512, numLayers=2, batchFirst=true)
let languageModel =
    rnn
    --> dsharp.view([-1; 512])
    --> Linear(512, dataset.numChars)

printfn "%s" (languageModel.summary())

let modelFileName = "rnn_language_model.params"
if File.Exists(modelFileName) then 
    printfn "Resuming training from existing model params found: %A" modelFileName
    languageModel.loadState(modelFileName)

let predict (text:string) len =
    rnn.reset()
    let mutable prediction = text
    let mutable last = text
    for i in 1..len do
        let lastTensor = last |> dataset.textToTensor
        let nextCharProbs = lastTensor.unsqueeze(0) --> languageModel --> dsharp.slice([-1]) --> dsharp.softmax(-1)
        last <- Categorical(nextCharProbs).sample() |> int |> dataset.indexToChar |> string
        prediction <- prediction + last
    prediction

let optimizer = Adam(languageModel, lr=dsharp.tensor(0.001))

let losses = ResizeArray()

let epochs = 10
let validInterval = 100

let start = System.DateTime.Now
for epoch = 1 to epochs do
    for i, x, t in loader.epoch() do
        let input =  x.[*,..seqLen-2]
        let target = t.[*,1..]
        rnn.reset()
        languageModel.reverseDiff()
        let output = input --> languageModel
        let loss = dsharp.crossEntropyLoss(output, target.view(-1))
        loss.reverse()
        optimizer.step()
        losses.Add(float loss)
        printfn "%A Epoch: %A/%A minibatch: %A/%A loss: %A" (System.DateTime.Now - start) epoch epochs (i+1) loader.length (float loss)

        if i % validInterval = 0 then
            printfn "\nSample from language model:\n%A\n" (predict "We " 512)

            languageModel.saveState(modelFileName)

            let plt = Pyplot()
            plt.plot(losses |> dsharp.tensor)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.tightLayout()
            plt.savefig (sprintf "rnn_loss_epoch_%A_minibatch_%A.pdf" epoch (i+1))