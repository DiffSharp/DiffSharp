(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.0"
#r "Microsoft.Z3.dll"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Torch.dll"
#r "DiffSharp.Backends.ShapeChecking.dll"
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

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.ShapeChecking

let Assert b = if not b then failwith "assertion constraint failed"

/// Variational auto-encoder example in DiffSharp (shape-aware)
//
// See https://www.compart.com/en/unicode/block/U+1D400 for nice italic characters
//[<ShapeCheck( "𝑋", "𝑌", "𝑍" )>]
[<ShapeCheck>]
type VAE(xDim:Int, yDim: Int, zDim:Int, ?hDims:seq<Int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
    inherit Model()
    let xyDim = xDim * yDim 
    do Assert (xDim >~ Int 0 ) 
    do Assert (yDim >~ Int 0 ) 
    //do if not (xDim =~= yDim ) then failwith "over constrained"
    let hDims = defaultArg hDims (let d = (xyDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let activation = defaultArg activation dsharp.relu
    let activationLast = defaultArg activationLast dsharp.sigmoid
    let dims = [| yield xyDim; yield! hDims; yield zDim |]
            
    let ndims = dims.Length
    let enc = [| for i in 0..ndims-2 do
                    Linear(dims.[i], dims.[i+1])
                 Linear(dims.[ndims-2], dims.[ndims-1])|]
    let dec = [|for i in 0..ndims-2 -> Linear(dims.[i+1], dims.[i])|] |> Array.rev
    do 
        base.add([for m in enc -> box m])
        base.add([for m in dec -> box m])

    let encode (x: Tensor) =
        let mutable x = x
        for i in 0..enc.Length-3 do
            x <- activation <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let latent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode (z: Tensor) =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- activation <| dec.[i].forward(h)
        activationLast <| dec.[dec.Length-1].forward(h)

    member _.encodeDecode(x:Tensor) =
        let mu, logVar = encode (x.viewx(Shape [|Int -1; xyDim|]))
        let z = latent mu logVar
        decode z, mu, logVar

    [<ShapeCheck( [| "𝐵"; "𝑋"; "𝑌" |], ReturnShape=[| |])>]
    member m.loss(x: Tensor) =
        let xRecon, mu, logVar = m.encodeDecode x
        let target = x.view(Shape [|Int -1; xyDim|])
        let bce = dsharp.bceLoss(xRecon, target, reduction="sum") 
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    //[<ShapeCheck( "𝑁" , ReturnShape=[| "𝑁"; "𝑋*𝑌" |] )>]
    member _.sample(?numSamples:Int) = 
        let numSamples = defaultArg numSamples (Int 1)
        dsharp.randn(Shape [|numSamples; zDim|]) |> decode

    //[<ShapeCheck( [| "𝐵"; "𝑋"; "𝑌" |] , ReturnShape=[| "𝐵"; "𝑋*𝑌" |] )>]
    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x

    override _.ToString() = sprintf "VAE(%A, %A, %A)" xyDim hDims zDim

    new (xDim:int, yDim:int, zDim:int, ?hDims:seq<int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
        VAE(Int xDim, Int yDim, Int zDim, ?hDims = Option.map (Seq.map Int) hDims, ?activation=activation, ?activationLast=activationLast)

dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(0)

let trainSet = MNIST("./mnist", train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=32, shuffle=true)

let model = VAE(28, 28, 16, [512; 256])
printfn "%A" model

let optimizer = Adam(model, lr=dsharp.tensor(0.001))

let epochs = 2
for epoch = 0 to epochs do
    for i, x, _ in trainLoader.epoch() do
        printfn "loader: x.shapex = %A" x.shapex
        model.reverseDiff()
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        printfn "epoch: %A/%A minibatch: %A/%A loss: %A" epoch epochs i trainLoader.length (float(l))

        if i % 250 = 249 then
            printfn "Saving samples"
            let samples = model.sample(Int 64).view([-1; 1; 28; 28])
            samples.saveImage(sprintf "samples_%A_%A.png" epoch i)

