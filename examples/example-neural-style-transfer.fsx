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

open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking

// See https://www.compart.com/en/unicode/block/U+1D400 for nice italic characters

let Model ps f = Model.create ps f

let Assert b = if not b then failwith "assertion constraint failed"

type Tensor with 
    member t.mean (dims: seq<int>, ?keepDim: bool) =
           (t, Array.rev (Array.sort (Seq.toArray dims))) ||> Array.fold (fun input dim -> dsharp.mean(input, dim, ?keepDim=keepDim))

    member t.stddev (dims: seq<int>, ?keepDim: bool) =
           (t, Array.rev (Array.sort (Seq.toArray dims))) ||> Array.fold (fun input dim -> dsharp.stddev(input, dim, ?keepDim=keepDim))

    member t.moments (dims: seq<int>, ?keepDim: bool) =
           t.mean(dims, ?keepDim=keepDim), t.stddev(dims, ?keepDim=keepDim)


[<ShapeCheck>]
module Test =

    [<ShapeCheck("N,C,H,W", "K,C,F1,F2")>]
    //[<ShapeCheck("N,C,748,748", "32,C,9,9")>]
    let someFunction (t1: Tensor, t2: Tensor) = 
           let res = t1.conv2d(t2, stride=2I, padding=4I)
           res


           

[<ShapeCheck(3)>]
type NeuralStyles(C: Int) =
    inherit Model()

    let instance_norm (channels: Int) = 
        let shift = Parameter (Weight.uniform ([ channels ], 0.0))
        let scale = Parameter (Weight.uniform ([ channels ], 1.0))
        Model [shift; scale] (fun input  ->
            let mu, sigma_sq = input.moments(dims= [2;3]) 
            let epsilon = 0.001
            let normalized =  (input - mu) / sqrt (sigma_sq + epsilon)
            scale.value.view([ channels; 1I; 1I ]) * normalized + shift.value.view([ channels; 1I; 1I ])
        )

    let conv_layer (in_channels: Int, out_channels: Int, filter_size: Int, stride: Int) = 
        let filters = Parameter (Weight.uniform ([ out_channels; in_channels; filter_size; filter_size], 0.1))
        let inorm = instance_norm (out_channels) 
        Model [filters; inorm] (fun input  ->
            dsharp.conv2d (input, filters.value, stride=stride, padding=filter_size/2)
            |> inorm.forward
        )

    let conv_transpose_layer (in_channels: Int, out_channels:Int, filter_size:Int, stride) =
        let filters = Parameter (Weight.uniform ([ in_channels; out_channels; filter_size; filter_size], 0.1))
        let inorm = instance_norm (out_channels) 
        Model [filters; inorm] (fun input  ->
            dsharp.convTranspose2d(input, filters.value, stride=stride, padding=filter_size/stride, outputPadding=filter_size % stride) 
            |> inorm.forward
        )

    let residual_block filter_size = 
        let conv1 = conv_layer (128I, 128I, filter_size, 1I)
        let conv2 = conv_layer (128I, 128I, filter_size, 1I) 
        Model [conv1; conv2] (fun input  ->
            input + conv1.forward input |> dsharp.relu |> conv2.forward
        )

    let to_pixel_value (input: Tensor) = 
        dsharp.tanh input * 150.0 + (255.0 / 2.0)
        
    let clip min max input = 
        dsharp.clamp(input, min, max)

    let model : Model =
        conv_layer (C, 32I, 9I, 1I) --> dsharp.relu
        --> conv_layer (32I, 64I, 3I, 2I) --> dsharp.relu
        --> conv_layer (64I, 128I, 3I, 2I) --> dsharp.relu
        --> residual_block 3I
        --> residual_block 3I
        --> residual_block 3I
        --> residual_block 3I
        --> residual_block 3I
        --> conv_transpose_layer (128I, 64I, 3I, 2I) --> dsharp.relu
        --> conv_transpose_layer (64I, 32I, 3I, 2I) --> dsharp.relu
        --> conv_layer (32I, C, 9I, 1I)
        --> to_pixel_value 
        --> clip 0.0 255.0

    [<ShapeCheck( "5,3,65,65", ReturnShape="5,3,65,65")>]
    //[<ShapeCheck( "5,C,65,65")>]
    override _.forward(input) = 
        model.forward(input) 
        
         


//[<ShapeCheck("𝐶")>]
//[<ShapeCheck( "5,3,65,65", ReturnShape="5,3,68,68")>]
//[<ShapeCheck( "5,3,68,68", ReturnShape="5,3,68,68")>]
//[<ShapeCheck("𝑁,𝐶,𝐻,𝑊")>] 
//[<ShapeCheck( "5,3,68,68")>]
 //, ReturnShape="5,3,68,68")>]
//[<ShapeCheck( "5,3,68,68")>] //, ReturnShape="5,3,68,68")>]

//[<ShapeCheck(1, "𝑁,𝐶,𝐻,𝑊")>]
//[<ShapeCheck( "𝑁,𝐶,𝐻,𝑊", ReturnShape="𝑁,𝐶,𝐻,𝑊")>]
//    [<ShapeCheck( "5,3,64,64", ReturnShape="5,3,64,64")>]
//    [<ShapeCheck( "5,3,65,65", ReturnShape="5,3,68,68")>]
//    [<ShapeCheck( "5,3,66,66", ReturnShape="5,3,68,68")>]
//[<ShapeCheck( "5,3,68,68")>] //, ReturnShape="5,3,68,68")>]
//[<ShapeCheck( "5,3,67,67")>] //, ReturnShape="5,3,68,68")>]
//[<ShapeCheck( "5,3,67,67", ReturnShape="5,3,68,68")>]
//[<ShapeCheck( "5,3,68,68", ReturnShape="5,3,68,68")>]

(*
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



*)