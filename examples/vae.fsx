#!/usr/bin/env -S dotnet fsi

(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests.ShapeChecking/bin/Debug/net5.0"
#r "Microsoft.Z3.dll"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.ShapeChecking.dll"
#r "System.Runtime.dll"
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

#compilertool @"e:\GitHub\dsyme\FSharp.Compiler.PortaCode\FSharp.Tools.LiveChecks.Analyzer\bin\Debug\netstandard2.0"

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.ShapeChecking

let Assert b = if not b then failwith "assertion constraint failed"

[<ShapeCheck("N,M")>]
let f (x: Tensor) = 
   let res = x.transpose(0,1)
   //let res = dsharp.cat[x;x;x;x] 
   res


(*

/// Variational auto-encoder example in DiffSharp (shape-aware)
//
type VAE(xDim:Int, yDim: Int, zDim:Int, ?hDims:seq<Int>, ?nonlinearity:Tensor->Tensor, ?nonlinearityLast:Tensor->Tensor) =
    inherit Model()
    let xyDim = xDim * yDim 
    do Assert (xDim >~ Int 0 ) 
    do Assert (yDim >~ Int 0 ) 
    let hDims = defaultArg hDims (let d = (xyDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let nonlinearity = defaultArg nonlinearity dsharp.relu
    let nonlinearityLast = defaultArg nonlinearityLast dsharp.sigmoid
    let dims = [| yield xyDim; yield! hDims; yield zDim |]
            
    let ndims = dims.Length
    let enc = [| for i in 0..ndims-2 do
                    Linear(dims.[i], dims.[i+1])
                 Linear(dims.[ndims-2], dims.[ndims-1]) |]
    let dec = [|for i in 0..ndims-2 -> Linear(dims.[i+1], dims.[i])|] |> Array.rev
    do 
        base.add([for m in enc -> box m])
        base.add([for m in dec -> box m])

    let encode (x: Tensor) =
        let mutable x = x
        for i in 0..enc.Length-3 do
            x <- nonlinearity <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let sampleLatent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode (z: Tensor) =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- nonlinearity <| dec.[i].forward(h)
        nonlinearityLast <| dec.[dec.Length-1].forward(h)

    member _.encodeDecode(x:Tensor) =
        let mu, logVar = encode (x.view([-1; xDim]))
        let z = sampleLatent mu logVar
        decode z, mu, logVar

    [<ShapeCheck( [| "𝐵"; "𝑋"; "𝑌" |] , ReturnShape=[| "𝐵"; "𝑋*𝑌" |] )>]
    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x

    override _.ToString() = sprintf "VAE(%A, %A, %A)" xDim hDims zDim

    //[<ShapeCheck( "𝑁" , ReturnShape=[| "𝑁"; "𝑋*𝑌" |] )>]
    static member loss(xRecon:Tensor, x:Tensor, mu:Tensor, logVar:Tensor) =
        let bce = dsharp.bceLoss(xRecon, x.viewAs(xRecon), reduction="sum")
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    member _.sample(?numSamples:Int) = 
        let numSamples = defaultArg numSamples (Int 1)
        dsharp.randn(Shape [|numSamples; zDim|]) |> decode

    override _.ToString() = sprintf "VAE(%A, %A, %A)" xyDim hDims zDim

    new (xDim:int, yDim:int, zDim:int, ?hDims:seq<int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
        VAE(Int xDim, Int yDim, Int zDim, ?hDims = Option.map (Seq.map Int) hDims, ?activation=activation, ?activationLast=activationLast)



dsharp.config(backend=Backend.Reference, device=Device.CPU)
dsharp.seed(0)

let epochs = 2
let batchSize = 32
let validInterval = 250
let numSamples = 32

let trainSet = MNIST("../data", train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)
let validSet = MNIST("../data", train=false, transform=id)
let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)

let model = VAE(28*28, 20, [400])
printfn "Model: %A" model

let optimizer = Adam(model, learningRate=dsharp.tensor(0.001))

for epoch = 1 to epochs do
    for i, x, _ in trainLoader.epoch() do
        printfn "loader: x.shapex = %A" x.shapex
        model.reverseDiff()
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        printfn "Epoch: %A/%A minibatch: %A/%A loss: %A" epoch epochs i trainLoader.length (float(l))

        if i % validInterval = 0 then
            let mutable validLoss = dsharp.zero()
            for _, x, _ in validLoader.epoch() do
                validLoss <- validLoss + model.loss(x, normalize=false)
            validLoss <- validLoss / validSet.length
            printfn "Validation loss: %A" (float validLoss)
            let fileName = sprintf "vae_samples_epoch_%A_minibatch_%A.png" epoch i
            printfn "Saving %A samples to %A" numSamples fileName
            let samples = model.sample(numSamples).view([-1; 1; 28; 28])
            samples.saveImage(fileName)

*)
