#!/usr/bin/env -S dotnet fsi

(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
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
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Util
open DiffSharp.Data


type VAE(xDim:int, zDim:int, ?hDims:seq<int>, ?nonlinearity:Tensor->Tensor, ?nonlinearityLast:Tensor->Tensor) =
    inherit Model()
    let hDims = defaultArg hDims (let d = (xDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let nonlinearity = defaultArg nonlinearity dsharp.relu
    let nonlinearityLast = defaultArg nonlinearityLast dsharp.sigmoid
    let dims =
        if hDims.Length = 0 then
            [|xDim; zDim|]
        else
            Array.append (Array.append [|xDim|] hDims) [|zDim|]
            
    let enc = Array.append [|for i in 0..dims.Length-2 -> Linear(dims.[i], dims.[i+1])|] [|Linear(dims.[dims.Length-2], dims.[dims.Length-1])|]
    let dec = [|for i in 0..dims.Length-2 -> Linear(dims.[i+1], dims.[i])|] |> Array.rev
    do 
        base.add([for m in enc -> box m])
        base.add([for m in dec -> box m])

    let encode x =
        let mutable x = x
        for i in 0..enc.Length-3 do
            let v = enc.[i].forward(x)
            x <- nonlinearity <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let sampleLatent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode z =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- nonlinearity <| dec.[i].forward(h)
        nonlinearityLast <| dec.[dec.Length-1].forward(h)

    member _.encodeDecode(x:Tensor) =
        let mu, logVar = encode (x.view([-1; xDim]))
        let z = sampleLatent mu logVar
        decode z, mu, logVar

    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x

    override _.ToString() = sprintf "VAE(%A, %A, %A)" xDim hDims zDim

    static member loss(xRecon:Tensor, x:Tensor, mu:Tensor, logVar:Tensor) =
        let bce = dsharp.bceLoss(xRecon, x.viewAs(xRecon), reduction="sum")
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    member m.loss(x, ?normalize:bool) =
        let normalize = defaultArg normalize true
        let xRecon, mu, logVar = m.encodeDecode x
        let loss = VAE.loss(xRecon, x, mu, logVar)
        if normalize then loss / x.shape.[0] else loss

    member _.sample(?numSamples:int) = 
        let numSamples = defaultArg numSamples 1
        dsharp.randn([|numSamples; zDim|]) |> decode


dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(0)

let v (x: double) = dsharp.tensor(x)

let trainSet = MNIST("../data", train=true, transform=id)
let validSet = MNIST("../data", train=false, transform=id)
let train(lr, beta1, beta2, numEpochs, dataLimit, hd) = 
    if float lr < 0.0 then failwith "learning rate went below zero"
    let numEpochs = defaultArg numEpochs 10
    let dataLimit = defaultArg dataLimit System.Int32.MaxValue
    let batchSize = 128
    let validInterval = 250
    let numSamples = 32

    let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)

    let model = VAE(28*28, 20, [400])
    printfn "Model: %A lr: %A" model lr

    let optimizer = Adam(model, lr=lr, beta1=beta1, beta2=beta2, hyperdescent=hd, hyperlr=v 0.05)

    let score() =
        let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)
        let mutable validLoss = dsharp.zero()
        for _, x, _ in validLoader.epoch() do
            validLoss <- validLoss + model.loss(x, normalize=false)
        validLoss <- validLoss / validSet.length
        float validLoss

    let tag = GlobalNestingLevel.Next()
    let losses = ResizeArray()
    for epoch = 1 to numEpochs do
      for i, x, _ in trainLoader.epoch() |> Seq.truncate dataLimit do
        model.reverseDiff(tag)
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        model.stripDiff()
        //let l = dsharp.log l
        //printfn $"l.primal = {l.primal}"
        //if l.primal.isForwardDiff() then printfn $"l.primal.derivative = {l.primal.derivative}"
        losses.Add(l.primal)

        printfn $"lr: %f{float optimizer.learningRate} beta1: %f{float beta1} beta2: %f{float beta2} epoch: {epoch}/{numEpochs} minibatch: {i}/{trainLoader.length} loss: %f{float l}" 

        //if i % validInterval = 0 then
        //    //let validLoss = score()
        //    //printfn "Validation loss: %A" (float validLoss)
        //    //let fileName = sprintf "vae_samples_epoch_%A_minibatch_%A.png" epoch i
        //    //printfn "Saving %A samples to %A" numSamples fileName
        //    //let samples = model.sample(numSamples).view([-1; 1; 28; 28])
        //    //samples.saveImage(fileName)

        //    let plt = Pyplot()
        //    plt.plot(losses |> dsharp.tensor, label="Losses")
        //    plt.xlabel("Iterations")
        //    plt.ylabel("Loss")
        //    plt.legend()
        //    plt.tightLayout()
        //    plt.savefig (sprintf "vae_loss_graph_epoch_%A_minibatch_%A_lr_%f.pdf" epoch (i+1) (float lr))

    let validLoss = score()
    printfn "Validation loss: %A" (float validLoss)

    let fileName = sprintf "vae_samples_loss_lr_%f.txt" (float lr)
    System.IO.File.WriteAllText(fileName, sprintf "Validation loss: %A" (float validLoss))

    let avgLosses = losses.ToArray()  |> Seq.map float |> Seq.windowed 5 |> Seq.map Seq.average 
    let fileName = sprintf "vae_samples_losses_lr_%f.txt" (float lr)
    System.IO.File.WriteAllLines(fileName, Array.mapi (fun i k -> sprintf $"%d{i}, %f{k}") (Array.ofSeq avgLosses))

    // Optimize the learning rate via the sum of losses on the final epoch
    //losses |> Seq.skip ((Seq.length losses / numEpochs) * (numEpochs-1)) |> Seq.sum
    
    // Returning running averages
    losses.ToArray()  |> Seq.map float |> Seq.windowed 10 |> Seq.map Seq.average 


// Optimize the learning rate starting with a very poor learning rate

// Optim.optim.sgd ((fun hyp -> train(hyp.exp(), v 0.9, v 0.999, Some 2, None)), x0=dsharp.tensor(log 0.001), lr=dsharp.tensor(0.0001))

//Optim.optim.sgd ((fun hyp -> train(hyp, 1, None)), x0=dsharp.tensor(0.000783917), lr=dsharp.tensor(0.00000000001))
//Optim.optim.sgd (train, x0=v 0.00001, lr=dsharp.tensor(0.0000000002))

// Optimize the learning rate starting with the one given in the paper
//Optim.optim.sgd ((fun hyp -> train(hyp, 2, None)), x0=dsharp.tensor(0.001), lr=dsharp.tensor(0.00000000001))
//Optim.optim.sgd ((fun hyp -> train(hyp, 2, None)), x0=dsharp.tensor(0.00136211), lr=dsharp.tensor(0.0000000002))

// Unoptimized learning rate

// Hyperdescent of learning rate
let losses1 = train (v 0.001, v 0.9, v 0.999, None, None, true)
//let losses2 = train (v 0.0001, v 0.9, v 0.999, None, None, true)
let losses3 = train (v 0.001, v 0.9, v 0.999, None, None, false)
//let losses4 = train (v 0.0001, v 0.9, v 0.999, None, None, false)

//let losses1 = train (v 0.0001, 2, None) // a poor learning rate
//let losses2 = train (v 0.001, 2, None)   // the one in the sample python code
//let losses3 = train (v 0.00136211, 2, None)  // the hyper-optimized one 

//let plt = Pyplot()
//plt.plot(losses1 |> dsharp.tensor, label="lr=0.00010")
//plt.plot(losses2 |> dsharp.tensor, label="lr=0.00100")
//plt.plot(losses3 |> dsharp.tensor, label="lr=0.00136")
//plt.xlabel("Iterations")
//plt.ylabel("Loss")
//plt.legend()
//plt.tightLayout()
//plt.savefig (sprintf "vae_loss_graph.pdf")

//----------------------------------------------------------
// Optimize learningRate and beta1 at same time
// 
// Note requires forward-on-reverse multi-pass

//Optim.optim.sgd ((fun hyp -> train(hyp.[0].exp(), 1.0-hyp.[1].exp(), v 0.999, None, None)), 
//                 x0=dsharp.tensor([log 0.001; log (1.0-0.9)]), lr=v 0.0001)

// 0.01:29:58 |   7 | 2.100153e+005  (Best min is 210015.2969 at x = tensor([-6.42431, -2.47364]))
//
//lr = 0.001621651832
//beta1 = 0.9157224702
//Validation loss: 109.3937988
// First under 120: 548   v.  629
// First under 130: 325   v.  390

//let losses1 = train (v 0.001621651832, v 0.9157224702, v 0.999, None, None)
//let losses2 = train (v 0.001, v 0.9, v 0.999, None, None)

//----------------------------------------------------------
// Optimize learningRate and beta1 and beta2 at same time
// 
// Note requires forward-on-reverse multi-pass
//
// First under 120: 399   v.  629
// First under 130: 299   v.  390

//Optim.optim.sgd ((fun hyp -> train(hyp.[0].exp(), 1.0-hyp.[1].exp(), 1.0-hyp.[2].exp(), None, None)), 
//                 x0=dsharp.tensor([log 0.001; log (1.0-0.9); log (1.0-0.999)]), lr=v 0.0001)

//lr: 0.001185 beta1: 0.587597 beta2: 0.999432
//let lr = 0.001075
//let beta1 = 0.777373
//let beta2 = 0.999467


//lr: 0.002416485222056508 beta1: 0.8328096866607666 beta2: 0.9990000128746033 epoch: 2/2 minibatch: 99/1875 loss: 143.86788940429688
//lr: 0.002434812020510435 beta1: 0.8374529480934143 beta2: 0.9990000128746033 epoch: 1/2 minibatch: 0/1875 loss: 549.8197021484375
//Validation loss: 141.219986
//v = tensor(15054.5), g = tensor([-75.5554, 281.651])
//0.00:07:06 |   9 | 1.505446e+004 +
//Model: VAE(784, [|400|], 20) lr: tensor(0.00243481, fwd)

//let losses1 = train (v lr, v beta1, v beta2, None, None, false)
//let losses2 = train (v 0.001, v 0.9, v 0.999, None, None, false)
let plt = Pyplot()
plt.plot(losses1 |> dsharp.tensor, label="lr=0.001 (adaptive)")
//plt.plot(losses2 |> dsharp.tensor, label="lr=0.0001 (adaptive)")
plt.plot(losses3 |> dsharp.tensor, label="lr=0.001")
//plt.plot(losses4 |> dsharp.tensor, label="lr=0.0001")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tightLayout()
plt.savefig (sprintf "vae_loss_graph_multi_opt.pdf")
