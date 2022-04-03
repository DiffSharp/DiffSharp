#!/usr/bin/env -S dotnet fsi

#I "../tests/DiffSharp.Tests/bin/Debug/net6.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
// #r "nuget: libtorch-cuda-11.1-win-x64, 1.8.0.7"
// #r "nuget: libtorch-cuda-11.1-linux-x64, 1.8.0.7"
// Option B: you can use a local libtorch installation
System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open DiffSharp
open DiffSharp.Model

dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(4)


let checkRNNCellShape (input:Tensor) (hidden:Tensor) inputSize hiddenSize =
    // input: [batchSize, inputSize]
    // hidden: [batchSize, hiddenSize]
    if input.dim <> 2 then failwithf "Expecting the input to be of shape [batchSize; inputSize] but got %A" input.shape
    if hidden.dim <> 2 then failwithf "Expecting the hidden to be of shape [batchSize; hiddenSize] but got %A" hidden.shape
    if input.shape[0] <> hidden.shape[0] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[0]
    if input.shape[1] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[1]
    if hidden.shape[1] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[1]

let checkRNNCellShapeSequence (input:Tensor) (hidden:Tensor) inputSize hiddenSize =
    // input: [seqLen, batchSize, inputSize]
    // hidden: [batchSize, hiddenSize]
    if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
    if hidden.dim <> 2 then failwithf "Expecting the hidden to be of shape [batchSize; hiddenSize] but got %A" hidden.shape
    if input.shape[0] <> hidden.shape[0] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[0]
    if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]
    if hidden.shape[1] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[1]

let checkRNNShapeSequence (input:Tensor) (hidden:Tensor) inputSize hiddenSize batchFirst numLayers numDirections =
    // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
    // hidden: [numLayers*numDirections, batchSize, hiddenSize]
    if hidden.dim <> 3 then failwithf "Expecting the hidden to be of shape [numLayers*numDirections; batchSize; hiddenSize] but got %A" hidden.shape
    let input =
        if batchFirst then
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [batchSize; sequenceLength; inputSize] but got %A" input.shape
            input.transpose(0, 1)
        else
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
            input
    // input: [seqLen, batchSize, inputSize]
    if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]
    if hidden.shape[2] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[2]
    if input.shape[1] <> hidden.shape[1] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[1] hidden.shape[1]
    if hidden.shape[0] <> numLayers*numDirections then failwithf "Expecting the hidden shape[0] to be %A but got %A" (numLayers*numDirections) hidden.shape[0]
    let seqLen, batchSize = input.shape[0], input.shape[1]
    input, seqLen, batchSize


type RNNCell2(inputSize, hiddenSize, ?nonlinearity, ?bias) =
    inherit Model<Tensor*Tensor, Tensor>()
    let nonlinearity = defaultArg nonlinearity dsharp.tanh
    let bias = defaultArg bias true
    let k = 1./sqrt (float hiddenSize)
    let wih = Parameter(Weight.uniform([inputSize; hiddenSize], k))
    let whh = Parameter(Weight.uniform([hiddenSize; hiddenSize], k))
    let b = Parameter(if bias then Weight.uniform([hiddenSize], k) else dsharp.tensor([]))
    do base.addParameter((wih, "RNNCell2-weight-ih"), (whh, "RNNCell2-weight-hh"), (b, "RNNCell2-bias"))

    override _.ToString() = sprintf "RNNCell2(%A, %A)" inputSize hiddenSize

    override _.forward((input, hidden)) =
        // input: [batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // returns: [batchSize, hiddenSize]
        // TODO: we can disable shape checking when .forward is used internally
        checkRNNCellShape input hidden inputSize hiddenSize
        let h = dsharp.matmul(input, wih.value) + dsharp.matmul(hidden, whh.value)
        let h = if bias then h + b.value else h
        let h = nonlinearity(h)
        h

    member r.forwardSequence((input, hidden)) =
        // input: [seqLen, batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // returns: [seqLen, batchSize, hiddenSize]
        // TODO: we can disable shape checking when .forwardSequence is used internally
        checkRNNCellShapeSequence input hidden inputSize hiddenSize
        let seqLen = input.shape[1]
        let output = Array.create seqLen (dsharp.tensor([]))
        for i = 0 to seqLen-1 do
            let h = if i = 0 then hidden else output[i-1]
            output[i] <- r.forward(input[i], h)
        dsharp.stack(output)


type RNN2(inputSize, hiddenSize, ?numLayers, ?nonlinearity, ?bias, ?batchFirst, ?dropout, ?bidirectional) =
    inherit Model<Tensor*Tensor, Tensor*Tensor>()
    let numLayers = defaultArg numLayers 1
    let dropout = defaultArg dropout 0.
    let bidirectional = defaultArg bidirectional false
    let batchFirst = defaultArg batchFirst false
    let numDirections = if bidirectional then 2 else 1
    let makeLayers () = Array.init numLayers (fun i -> if i = 0 then RNNCell2(inputSize, hiddenSize, ?nonlinearity=nonlinearity, ?bias=bias) else RNNCell2(hiddenSize, hiddenSize, ?nonlinearity=nonlinearity, ?bias=bias))
    let layers = makeLayers()
    let layersReverse = if bidirectional then makeLayers() else [||]
    let dropoutLayer = Dropout(dropout)
    do
        base.addModel(layers |> Array.mapi (fun i l -> l :> ModelBase, sprintf "RNN-layer-%A" i))
        if bidirectional then base.addModel(layersReverse |> Array.mapi (fun i l -> l :> ModelBase, sprintf "RNN-layer-reverse-%A" i))
        if dropout > 0. then base.addModel(dropoutLayer, "RNN-dropout")

    override _.ToString() = sprintf "RNN(%A, %A, numLayers:%A, bidirectional:%A)" inputSize hiddenSize numLayers bidirectional

    override _.forward((input, hidden)) =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // hidden: [numLayers*numDirections, batchSize, hiddenSize]
        // returns: {[seqLen, batchSize, hiddenSize] or [batchSize, seqLen, hiddenSize]}, [numLayers*numDirections, batchSize, hiddenSize]
        let input, _, _ = checkRNNShapeSequence input hidden inputSize hiddenSize batchFirst numLayers numDirections
        // input: [seqLen, batchSize, inputSize]
        let newHidden = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let mutable hFwd = input
        for i in 0..numLayers-1 do
            let h = hidden[i]
            hFwd <- layers[i].forwardSequence(hFwd, h)
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.forward(hFwd)
            // TODO: newHidden[i] <- 
        let output =
            if bidirectional then
                let mutable hRev = input.flip([0])
                for i in 0..numLayers-1 do
                    let h = hidden[numLayers+i]
                    hRev <- layersReverse[i].forwardSequence(hRev, h)
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.forward(hRev)
                    // TODO: newHidden[numLayers+i] <- 
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        let output = if batchFirst then output.transpose(0, 1) else output
        let hidden = dsharp.stack(newHidden)
        output, hidden