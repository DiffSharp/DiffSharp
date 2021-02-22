#!/usr/bin/env -S dotnet fsi

(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Torch.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // IPYNB

(*** condition: fsx ***)
#if FSX
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/win-x64/native/torch_cpu.dll")
#r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
#endif // FSX

(*** condition: ipynb ***)
#if IPYNB
// This is a workaround for https://github.com/dotnet/fsharp/issues/10136, necessary in F# scripts and .NET Interactive
System.Runtime.InteropServices.NativeLibrary.Load(let path1 = System.IO.Path.GetDirectoryName(typeof<DiffSharp.dsharp>.Assembly.Location) in if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/linux-x64/native/libtorch.so" else path1 + "/../../../../libtorch-cpu/1.5.6/runtimes/win-x64/native/torch_cpu.dll")

// Set up formatting for notebooks
Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim
open DiffSharp.Util

dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(0)

let nz = 100
let ngf = 2
let ndf = 2
let nc = 3

let generator =
    dsharp.view([-1; nz; 1; 1])
    --> ConvTranspose2d(nz, ngf*8, 2, 1, 0, bias=false)
    --> BatchNorm2d(ngf*8)
    --> dsharp.relu
    --> ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=false)
    --> BatchNorm2d(ngf*4)
    --> dsharp.relu
    --> ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=false)
    --> BatchNorm2d(ngf*2)
    --> dsharp.relu
    --> ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=false)
    --> BatchNorm2d(ngf)
    --> dsharp.relu
    --> ConvTranspose2d(ngf, nc, 4, 2, 1, bias=false)
    --> dsharp.tanh

let discriminator =
    Conv2d(nc, ndf, 2, 2, 1, bias=false)
    --> dsharp.leakyRelu 0.2
    --> Conv2d(ndf, ndf*2, 2, 2, 1, bias=false)
    --> BatchNorm2d(ndf*2)
    --> dsharp.leakyRelu 0.2
    --> Conv2d(ndf*2, ndf*4, 2, 2, 1, bias=false)
    --> BatchNorm2d(ndf*4)
    --> dsharp.leakyRelu 0.2
    --> Conv2d(ndf*4, ndf*8, 2, 2, 1, bias=false)
    --> BatchNorm2d(ndf*8)
    --> dsharp.leakyRelu 0.2
    --> Conv2d(ndf*8, 1, 2, 2, 0, bias=false)
    --> dsharp.sigmoid
    --> dsharp.view([-1; 1])

// let x = dsharp.randn([16; 3; 32; 32])
// let d = x --> discriminator
// print d.shape
// print discriminator.nparameters

// let z = dsharp.randn([16; nz])
// let g = z --> generator
// print g.shape
// print generator.nparameters
// g.saveImage("g_sample.png", normalize=true)

let epochs = 2
let batchSize = 32
let validInterval = 250
let numSamples = 32

let cifar10 = CIFAR10("../data", train=false)
let cifar10bird = cifar10.filter(fun _ t -> cifar10.classNames.[int t] = "bird")

let loader = cifar10bird.loader(batchSize=batchSize, shuffle=true)

let gopt = Adam(generator, lr=dsharp.tensor(0.001))
let dopt = Adam(discriminator, lr=dsharp.tensor(0.001))

for epoch = 1 to epochs do
    for i, x, _ in loader.epoch() do

        discriminator.reverseDiff()
        let label = dsharp.full([batchSize; 1], 1)
        let output = x --> discriminator
        let l = dsharp.bceLoss(output, label)
        // l.reverse()
        // dopt.step()

        printfn "Epoch: %A/%A minibatch: %A/%A loss: %A" epoch epochs i loader.length (float(l))

        // if i % validInterval = 0 then
        //     let mutable validLoss = dsharp.zero()
        //     for _, x, _ in validLoader.epoch() do
        //         validLoss <- validLoss + model.loss(x, normalize=false)
        //     validLoss <- validLoss / validSet.length
        //     printfn "Validation loss: %A" (float validLoss)
        //     let fileName = sprintf "vae_samples_epoch_%A_minibatch_%A.png" epoch i
        //     printfn "Saving %A samples to %A" numSamples fileName
        //     let samples = model.sample(numSamples).view([-1; 1; 28; 28])
        //     samples.saveImage(fileName)
