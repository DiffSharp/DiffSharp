#targetfx "netcore"
#time;;
(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests.Gpu/bin/Release/netcoreapp3.0"
#r "TorchSharp.dll"
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

System.Runtime.InteropServices.NativeLibrary.Load(@"E:\GitHub\dsyme\DiffSharp\tests\DiffSharp.Tests.Gpu\bin\Release\netcoreapp3.0\runtimes\win-x64\native\torch_cuda.dll")
System.Runtime.InteropServices.NativeLibrary.Load(@"E:\GitHub\dsyme\DiffSharp\tests\DiffSharp.Tests.Gpu\bin\Release\netcoreapp3.0\runtimes\win-x64\native\nvrtc-builtins64_102.dll")
System.Runtime.InteropServices.NativeLibrary.Load(@"E:\GitHub\dsyme\DiffSharp\tests\DiffSharp.Tests.Gpu\bin\Release\netcoreapp3.0\runtimes\win-x64\native\caffe2_nvrtc.dll")
System.Runtime.InteropServices.NativeLibrary.Load(@"E:\GitHub\dsyme\DiffSharp\tests\DiffSharp.Tests.Gpu\bin\Release\netcoreapp3.0\runtimes\win-x64\native\nvrtc64_102_0.dll")
TorchSharp.Torch.IsCudaAvailable()

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data


/// VAE(28*28, 16, [512; 256])
type VAE(xDim:int, zDim:int, ?hDims:seq<int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
    inherit Model()
    let hDims = defaultArg hDims (let d = (xDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let activation = defaultArg activation dsharp.relu
    let activationLast = defaultArg activationLast dsharp.sigmoid
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
            x <- activation <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let latent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode z =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- activation <| dec.[i].forward(h)
        activationLast <| dec.[dec.Length-1].forward(h)

    member _.encodeDecode(x:Tensor) =
        let mu, logVar = encode (x.view([-1; xDim]))
        let z = latent mu logVar
        decode z, mu, logVar

    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x

    override _.ToString() = sprintf "VAE(%A, %A, %A)" xDim hDims zDim

    static member loss(xRecon:Tensor, x:Tensor, mu:Tensor, logVar:Tensor) =
        let bce = dsharp.bceLoss(xRecon, x.view([|-1; 28*28|]), reduction="sum")
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    member m.loss(x) =
        let xRecon, mu, logVar = m.encodeDecode x
        VAE.loss(xRecon, x, mu, logVar)

    member _.sample(?numSamples:int) = 
        let numSamples = defaultArg numSamples 1
        dsharp.randn([|numSamples; zDim|]) |> decode


TorchSharp.Torch.IsCudaAvailable()
dsharp.config(backend=Backend.Torch, device=Device.GPU)
dsharp.seed(0)
dsharp.tensor([1..10])
let trainSet = MNIST("./mnist", train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=2048, shuffle=true)

let model = VAE(28*28, 16, [512; 256])
printfn "%A" model
let optimizer = Adam(model, lr=dsharp.tensor(0.001))

let epochs = 0
for epoch = 0 to epochs do
    let t = Diagnostics.Stopwatch()
    t.Start()
    let mutable tlast = t.Elapsed
    for i, x, _ in trainLoader.epoch() do
        printfn "trainLoader.epoch(): %A" (t.Elapsed - tlast)
        tlast <- t.Elapsed
        model.reverseDiff()
        printfn "reverseDiff(): %A" (t.Elapsed - tlast)
        tlast <- t.Elapsed
        let l = model.loss(x)
        printfn "model.loss(): %A" (t.Elapsed - tlast)
        tlast <- t.Elapsed
        l.reverse()
        printfn "l.reverse(): %A" (t.Elapsed - tlast)
        tlast <- t.Elapsed
        optimizer.step()
        printfn "optimizer.step: %A" (t.Elapsed - tlast)
        tlast <- t.Elapsed
        printfn "epoch: %A/%A minibatch: %A/%A loss: %A time: %A" epoch epochs i trainLoader.length (float(l)) t.Elapsed

        if i % 250 = 0 && i <> 0 then
            printfn "Saving samples"
            let samples = model.sample(64).view([-1; 1; 28; 28])
            samples.move(Device.CPU).saveImage(sprintf "samples_%A_%A.png" epoch i)
            printfn "Done saving samples"

(*

CPU: 

    typical trainLoader.epoch(): 00:00:00.3543867
    typical optimizer.step     : 00:00:00.0350334
    Real: 00:00:25.789, CPU: 00:01:41.250, GC gen0: 27, gen1: 12, gen2: 1

GPU with data loading mod: 

    typical trainLoader.epoch(): 00:00:00.2713589
    optimizer.step: 00:00:00.0743182
    Real: 00:00:19.201, CPU: 00:00:22.203, GC gen0: 27, gen1: 11, gen2: 1

GPU without data loading mod: 

    typical trainLoader.epoch(): 00:00:05.1245604
    optimizer.step: 00:00:00.0843537
    Real: 00:02:36.755, CPU: 00:02:37.109, GC gen0: 30, gen1: 13, gen2: 1


*)

