#!/usr/bin/env -S dotnet fsi

#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Torch.dll"
// #r "nuget: libtorch-cuda-11.1-linux-x64, 1.8.0.7"
System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim
open DiffSharp.Util

open System.Collections.Generic

dsharp.config(backend=Backend.Torch, device=Device.CPU)
dsharp.seed(1)

let rnnShape (value:Tensor) inFeatures batchFirst =
    let value =
        if batchFirst then
            if value.dim <> 3 then failwithf "Expecting the input to be of shape batchSize x seqLen x inFeatures, but received input with shape %A" value.shape
            value.transpose(0, 1)
        else
            if value.dim <> 3 then failwithf "Expecting the input to be of shape seqLen x batchSize x inFeatures, but received input with shape %A" value.shape
            value
    if value.shape.[2] <> inFeatures then failwithf "Expecting input to have %A features, but received input with shape %A" inFeatures value.shape
    let seqLen, batchSize = value.shape.[0], value.shape.[1]
    value, seqLen, batchSize


type RNNCell(inFeatures, outFeatures, ?nonlinearity, ?bias, ?batchFirst) =
    inherit Model()
    let nonlinearity = defaultArg nonlinearity dsharp.tanh
    let bias = defaultArg bias true
    let batchFirst = defaultArg batchFirst false
    let k = 1./sqrt (float outFeatures)
    let wih = Parameter(Weight.uniform([|inFeatures; outFeatures|], k))
    let whh = Parameter(Weight.uniform([|outFeatures; outFeatures|], k))
    let b = Parameter(if bias then Weight.uniform([|outFeatures|], k) else dsharp.tensor([]))
    let h = Parameter <| dsharp.tensor([])
    do base.add([wih;whh;b],["RNNCell-weight-ih";"RNNCell-weight-hh";"RNNCell-bias"])

    member _.hidden 
        with get () = h.value
        and set v = h.value <- v

    override _.getString() = sprintf "RNNCell(%A, %A)" inFeatures outFeatures

    member r.reset() = r.hidden <- dsharp.tensor([])

    override r.forward(value) =
        let value, seqLen, batchSize = rnnShape value inFeatures batchFirst
        if r.hidden.nelement = 0 then r.hidden <- dsharp.zeros([batchSize; outFeatures])
        let output = Array.create seqLen (dsharp.tensor([]))
        for i in 0..seqLen-1 do
            let v = value.[i]
            r.hidden <- dsharp.matmul(v, wih.value) + dsharp.matmul(h.value, whh.value)
            if bias then r.hidden <- r.hidden + b.value
            r.hidden <- nonlinearity r.hidden
            output.[i] <- r.hidden
        let output = dsharp.stack output
        if batchFirst then output.transpose(0, 1) else output


type RNN(inFeatures, outFeatures, ?numLayers, ?nonlinearity, ?bias, ?batchFirst, ?dropout, ?bidirectional) =
    inherit Model()
    let numLayers = defaultArg numLayers 1
    let dropout = defaultArg dropout 0.
    let bidirectional = defaultArg bidirectional false
    let batchFirst = defaultArg batchFirst false
    let numDirections = if bidirectional then 2 else 1
    let makeLayers () = Array.init numLayers (fun i -> if i = 0 then RNNCell(inFeatures, outFeatures, ?nonlinearity=nonlinearity, ?bias=bias) else RNNCell(outFeatures, outFeatures, ?nonlinearity=nonlinearity, ?bias=bias))
    let layers = makeLayers()
    let layersReverse = if bidirectional then makeLayers() else [||]
    let dropoutLayer = Dropout(dropout)
    let hs = Parameter <| dsharp.tensor([])
    do 
        base.add(layers |> Array.map box, Array.init numLayers (fun i -> sprintf "RNN-layer-%A" i))
        if bidirectional then base.add(layersReverse |> Array.map box, Array.init numLayers (fun i -> sprintf "RNN-layer-reverse-%A" i))
        if dropout > 0. then base.add([dropoutLayer], ["RNN-dropout"])

    member _.hidden
        with get () = hs.value
        and set v = hs.value <- v

    override _.getString() = sprintf "RNN(%A, %A, numLayers:%A)" inFeatures outFeatures numLayers

    member r.reset() = r.hidden <- dsharp.tensor([])

    override r.forward(value) =
        let value, _, batchSize = rnnShape value inFeatures batchFirst
        if r.hidden.nelement = 0 then r.hidden <- dsharp.zeros([numLayers*numDirections; batchSize; outFeatures])
        let newhs = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let mutable hFwd = value
        for i in 0..numLayers-1 do 
            layers.[i].hidden <- r.hidden.[i]
            hFwd <- layers.[i].forward(hFwd)
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.forward(hFwd)
            newhs.[i] <- layers.[i].hidden
        let output = 
            if bidirectional then
                let mutable hRev = value.flip([0])
                for i in 0..numLayers-1 do 
                    layersReverse.[i].hidden <- r.hidden.[numLayers+i]
                    hRev <- layersReverse.[i].forward(hRev)
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.forward(hRev)
                    newhs.[numLayers+i] <- layersReverse.[i].hidden
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        r.hidden <- dsharp.stack(newhs)
        if batchFirst then output.transpose(0, 1) else output

// type Tokenizer(?example) = 
//     let example = defaultArg example """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ """
//     let chars = example |> Seq.distinct |> Seq.toArray
//     let onehot = memoize dsharp.onehot
//     member _.length = chars.Length
//     member _.charToIndex(c) =
//         let i = 
//             try
//                 Array.findIndex ((=) c) chars
//             with
//             | _ -> failwithf "Given char '%A' is not a part of this tokenizer %A" c chars
//         i
//     member t.textToIndices(text:string) = text |> Seq.map t.charToIndex |> Seq.toArray
//     member t.indicesToTensor(indices) = indices |> Array.map (fun i -> onehot(t.length, i)) |> dsharp.stack
//     member t.indexToChar(index) = chars.[index]
//     member t.textToTensor(texts:string[]) = texts |> Array.map (t.textToIndices >> t.indicesToTensor) |> dsharp.stack
//     member t.tensorToText(tensor:Tensor) =
//         if tensor.dim <> 3 then failwithf "Expecting a 3d tensor with shape batchSize x seqLen x features, received tensor with shape %A" tensor.shape 
//         let t2text (tens:Tensor) = [|for i in 0..tens.shape.[0]-1 do tens.[i].argmax().[0]|] |> Array.map t.indexToChar |> System.String |> string
//         [|for i in 0..tensor.shape.[0]-1 do tensor.[i] |> t2text|]
//     member t.dataset(text:string, seqLength) =
//         if seqLength > text.Length then failwithf "Expecting text.Length (%A) >= seqLength (%A)" text.Length seqLength
//         let sequences = [|for i in 0..(text.Length - seqLength + 1)-1 do text.Substring(i, seqLength)|] |> Array.map t.textToIndices
//         let data = sequences |> Array.map t.indicesToTensor |> dsharp.stack
//         let target = sequences |> Array.map dsharp.tensor |> dsharp.stack
//         TensorDataset(data, target)


type TextDataset(text:string, seqLength) =
    inherit Dataset()
    let _chars = text |> Seq.distinct |> Seq.toArray |> Array.sort
    let onehot = memoize dsharp.onehot
    let charToIndexDict =
        let d = new Dictionary<char,int>()
        for c in _chars do d.[c] <- Array.findIndex ((=) c) _chars
        d
    let charToIndex(c) = charToIndexDict.[c]
    let indexToChar(index) = _chars.[index]
    let textToIndices(text:string) = text |> Seq.map charToIndex |> Seq.toArray
    let indicesToTensor(indices) = indices |> Array.map (fun i -> onehot(_chars.Length, i)) |> dsharp.stack
    let sequences = 
        if seqLength > text.Length then failwithf "Expecting text.Length (%A) >= seqLength (%A)" text.Length seqLength
        [|for i in 0..(text.Length - seqLength + 1)-1 do text.Substring(i, seqLength)|] |> Array.map textToIndices

    member d.textToTensor(text:string) = text |> textToIndices |> indicesToTensor
    member d.tensorToText(tensor:Tensor) =
        if tensor.dim <> 2 then failwithf "Expecting a 2d tensor with shape seqLen x features, received tensor with shape %A" tensor.shape 
        let t2text (tens:Tensor) = [|for i in 0..tens.shape.[0]-1 do tens.[i].argmax().[0]|] |> Array.map indexToChar |> System.String |> string
        tensor |> t2text

    member d.chars = _chars
    override d.length = sequences.Length
    override d.item(i) =
        let data = sequences.[i] |> indicesToTensor
        let target = sequences.[i] |> dsharp.tensor
        data, target

let seqLen = 32

// let text = "A merry little surge of electricity piped by automatic alarm from the mood organ beside his bed awakened Rick Deckard."
download "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt" "./shakespeare.txt"
let corpus = System.IO.File.ReadAllText("./shakespeare.txt").[..50]

let dataset = TextDataset(corpus, 32)
let loader = dataset.loader(batchSize=5, shuffle=true)

let rnn = RNN(dataset.chars.Length, dataset.chars.Length, numLayers=2, batchFirst=true)
// let net =
    // rnn
    // --> dsharp.view([-1; 512])
    // --> Linear(512, dataset.chars.Length)
let net = rnn

print rnn
print net
let optimizer = Adam(net, lr=dsharp.tensor(0.001))


let losses = ResizeArray()

let epochs = 1
let validInterval = 100

let start = System.DateTime.Now
for epoch = 1 to epochs do
    for i, x, t in loader.epoch() do
        let input =  x.[*,..seqLen-2]
        let target = t.[*,1..]
        rnn.reset()
        net.reverseDiff()
        let output = input --> net
        let loss = dsharp.crossEntropyLoss(output.view([-1;dataset.chars.Length]), target.view(-1))
        loss.reverse()
        optimizer.step()
        losses.Add(float loss)
        printfn "%A Epoch: %A/%A minibatch: %A/%A goss: %A" (System.DateTime.Now - start) epoch epochs (i+1) loader.length (float loss)

        if i % validInterval = 0 then
            let plt = Pyplot()
            plt.plot(losses |> dsharp.tensor)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.tightLayout()
            plt.savefig (sprintf "rnn_loss_epoch_%A_minibatch_%A.pdf" epoch (i+1))

// WORK IN PROGRESS

let mutable text = "F"
let len = 10

// for i in 0..len-1 do
let c = text.[0] |> string |> dataset.textToTensor |> dsharp.unsqueeze(0)
print c
print c.shape
let n = c --> net// --> dsharp.softmax(1)
print n
print n.shape
print dataset.chars.Length
// let nc = n |> dataset.tensorToText

