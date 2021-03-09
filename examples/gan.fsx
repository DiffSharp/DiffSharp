#!/usr/bin/env -S dotnet fsi

#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"
// #r "nuget: libtorch-cuda-10.2-linux-x64, 1.7.0.1"
//System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim
open DiffSharp.Util

let nz = 128


dsharp.config(backend=Backend.Torch, device=Device.CPU)
let train (lrg, lrd, betag, betad, nepochs, maxbatches) = 
    dsharp.seed(4)
    let maxbatches = defaultArg maxbatches System.Int32.MaxValue
    let mnist = MNIST("../data", train=true, transform=fun t -> (t - 0.5) / 0.5)

    let generator =
        dsharp.view([-1;nz])
        --> Linear(nz, 256)
        --> dsharp.leakyRelu(0.2)
        --> Linear(256, 512)
        --> dsharp.leakyRelu(0.2)
        --> Linear(512, 1024)
        --> dsharp.leakyRelu(0.2)
        --> Linear(1024, 28*28)
        --> dsharp.tanh

    let discriminator =
        dsharp.view([-1; 28*28])
        --> Linear(28*28, 1024)
        --> dsharp.leakyRelu(0.2)
        --> dsharp.dropout(0.3)
        --> Linear(1024, 512)
        --> dsharp.leakyRelu(0.2)
        --> dsharp.dropout(0.3)
        --> Linear(512, 256)
        --> dsharp.leakyRelu(0.2)
        --> dsharp.dropout(0.3)
        --> Linear(256, 1)
        --> dsharp.sigmoid

    print "Generator"
    print generator

    print "Discriminator"
    print discriminator

    let batchSize = 64
    let validInterval = 100

    let fixedNoise = dsharp.randn([batchSize; nz])

    let glosses = ResizeArray()
    let dlosses = ResizeArray()
    let dxs = ResizeArray()
    let dgzs = ResizeArray()

    let loader = mnist.loader(batchSize=batchSize, shuffle=true)

    let tag = GlobalNestingLevel.Next()
    let gopt = Adam(generator, lr=lrg, beta1=betag, reversible=true)
    let dopt = Adam(discriminator, lr=lrd, beta1=betad, reversible=true)
    let start = System.DateTime.Now
    for epoch = 1 to nepochs do
      for i, x, _ in loader.epoch() |> Seq.truncate maxbatches do
        let labelReal = dsharp.ones([batchSize; 1])
        let labelFake = dsharp.zeros([batchSize; 1])

        // update discriminator
        //printfn "i = %d" i
        if i <> 0 then generator.stripDiff()
        discriminator.reverseDiff(tag=tag)

        //printfn $"1. generator = {generator.parametersVector.summary()}"
        //printfn $"2. discriminator = {discriminator.parametersVector.summary()}"

        let doutput = x --> discriminator
        let dx = doutput.mean() |> float
        let dlossReal = dsharp.bceLoss(doutput, labelReal)

        let z = dsharp.randn([batchSize; nz])
        let goutput = z --> generator
        let doutput = goutput --> discriminator
        let dgz = doutput.mean() |> float
        let dlossFake = dsharp.bceLoss(doutput, labelFake)

        let dloss = dlossReal + dlossFake
        dloss.reverse()
        dopt.step()
        dlosses.Add(dloss.primal)
        dxs.Add(float dx)
        dgzs.Add(float dgz)

        // update generator 
        generator.reverseDiff(tag=tag)
        discriminator.stripDiff()

        //printfn $"3. generator = {generator.parametersVector.summary()}"
        //printfn $"4. discriminator = {discriminator.parametersVector.summary()}"

        let goutput = z --> generator
        let doutput = goutput --> discriminator
        let gloss = dsharp.bceLoss(doutput, labelReal)
        gloss.reverse()
        gopt.step()
        glosses.Add(gloss.primal)

        //printfn $"5. goutput = {goutput.summary()}"
        //printfn $"6. doutput = {doutput.summary()}"
        //printfn $"7. gloss = {gloss.summary()}, {float gloss}" 
        //printfn $"7. gloss.primal = {gloss.primal.summary()}, {float gloss.primal}"

        printfn "%A lrg: %A lrd: %A betag: %A betad: %A Epoch: %A/%A minibatch: %A/%A gloss: %A dloss: %A d(x): %A d(g(z)): %A" (System.DateTime.Now - start) lrg lrd betag betad epoch nepochs (i+1) loader.length (float gloss) (float dloss) dx dgz

        if i % validInterval = 0 then
            let realFileName = sprintf "gan_real_samples_epoch_%A_minibatch_%A.png" epoch (i+1)
            printfn "Saving real samples to %A" realFileName
            ((x+1)/2).saveImage(realFileName, normalize=false)
            let fakeFileName = sprintf "gan_fake_samples_epoch_%A_minibatch_%A.png" epoch (i+1)
            printfn "Saving fake samples to %A" fakeFileName
            let goutput = fixedNoise --> generator
            ((goutput.view([-1;1;28;28])+1)/2).saveImage(fakeFileName, normalize=false)

            let plt = Pyplot()
            plt.plot(glosses |> dsharp.tensor, label="Generator")
            plt.plot(dlosses |> dsharp.tensor, label="Discriminator")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.tightLayout()
            plt.savefig (sprintf "gan_loss_epoch_%A_minibatch_%A.pdf" epoch (i+1))

            let plt = Pyplot()
            plt.plot(dxs |> dsharp.tensor, label="d(x)")
            plt.plot(dgzs |> dsharp.tensor, label="d(g(z))")
            plt.xlabel("Iterations")
            plt.ylabel("Score")
            plt.legend()
            plt.tightLayout()
            plt.savefig (sprintf "gan_score_epoch_%A_minibatch_%A.pdf" epoch (i+1))       
    Seq.sum glosses + Seq.sum dlosses

//train (dsharp.tensor(0.0002))
//printfn "---------------"
//train (dsharp.tensor(0.0002))
//printfn "---------------"
let v (x: double) = dsharp.tensor(x)

// Use the unoptimized learning rates over the hole data set for one epoch
train (v 0.0003, v 0.0003, v 0.5, v 0.5, 1, None)

// "Optimize" the learning rates using SGD and the first 640 shuffled images only 
//Optim.optim.sgd ((fun hyp -> train(hyp.[0], hyp.[1], v 0.5, v 0.5, 1, Some 10)), x0=dsharp.tensor([0.0003,0.0003]), lr=dsharp.tensor(0.0000000005))

// Answer is 0.0000487555, v 0.000340143 

// Use the optimized learning rates over the hole data se4t for one epoch
//train (v 0.0000487555, v 0.000340143, v 0.5, v 0.5, 1, None)

//Optim.optim.sgd (train, x0=dsharp.tensor([4.87555e-05, 0.000340143, 0.5, 0.5]), lr=dsharp.tensor(0.0000000001))
//Optim.optim.sgd ((fun hyp ->  train (dsharp.tensor(4.87555e-05), dsharp.tensor(0.000340143), hyp.[0], hyp.[1], 1, Some 20)), x0=dsharp.tensor([0.5, 0.5]), lr=dsharp.tensor(0.0001))
//train (dsharp.tensor([0.0000487555, 0.000340143, 0.5, 0.5], 1, None))


// Optimize the learning rates using Adam and the first 640 shuffled images only 
// Optim.optim.adam ((fun hyp -> train(hyp.[0], hyp.[1], v 0.5, v 0.5, 1, Some 10)), x0=dsharp.tensor([0.0003,0.0003]), lr=dsharp.tensor(0.000005))

// Answer is TBD - looks about the same
