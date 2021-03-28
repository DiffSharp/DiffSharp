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
open DiffSharp.Distributions

open System.IO

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
    let h = Parameter <| dsharp.tensor([]) // Not a paramter to be trained, this is for keeping hidden state
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

type LSTMCell(inFeatures, outFeatures, ?bias, ?batchFirst) =
    inherit Model()
    let bias = defaultArg bias true
    let batchFirst = defaultArg batchFirst false
    let k = 1./sqrt (float outFeatures)
    let wih = Parameter(Weight.uniform([|inFeatures; outFeatures*4|], k))
    let whh = Parameter(Weight.uniform([|outFeatures; outFeatures*4|], k))
    let b = Parameter(if bias then Weight.uniform([|outFeatures*4|], k) else dsharp.tensor([]))
    let h = Parameter <| dsharp.tensor([]) // Not a paramter to be trained, this is for keeping hidden state
    let c = Parameter <| dsharp.tensor([]) // Not a paramter to be trained, this is for keeping hidden state
    do base.add([wih;whh;b],["LSTMCell-weight-ih";"LSTMCell-weight-hh";"LSTMCell-bias"])

    member _.hidden 
        with get () = h.value
        and set v = h.value <- v

    member _.cell
        with get () = c.value
        and set v = c.value <- v

    override _.getString() = sprintf "LSTMCell(%A, %A)" inFeatures outFeatures

    member r.reset() = r.hidden <- dsharp.tensor([])

    override r.forward(value) =
        let value, seqLen, batchSize = rnnShape value inFeatures batchFirst
        if r.hidden.nelement = 0 then r.hidden <- dsharp.zeros([batchSize; outFeatures])
        if r.cell.nelement = 0 then r.cell <- dsharp.zeros([batchSize; outFeatures])
        let output = Array.create seqLen (dsharp.tensor([]))
        for i in 0..seqLen-1 do
            let v = value.[i]
            let x2h = dsharp.matmul(v, wih.value)
            let h2h = dsharp.matmul(h.value, whh.value)
            let mutable pre = x2h + h2h
            if bias then pre <- pre + b.value
            let pretan = pre.[*,..outFeatures-1].tanh()
            let presig = pre.[*,outFeatures..].sigmoid()
            let inputGate = presig.[*,..outFeatures-1]
            let forgetGate = presig.[*,outFeatures..(2*outFeatures)-1]
            let outputGate = presig.[*,(2*outFeatures)..]
            r.cell <- (inputGate*pretan) + (forgetGate*c.value)
            r.hidden <- outputGate*c.value.tanh()
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
    let hs = Parameter <| dsharp.tensor([]) // Not a parameter to be trained, it is for keeping hidden state
    do 
        base.add(layers |> Array.map box, Array.init numLayers (fun i -> sprintf "RNN-layer-%A" i))
        if bidirectional then base.add(layersReverse |> Array.map box, Array.init numLayers (fun i -> sprintf "RNN-layer-reverse-%A" i))
        if dropout > 0. then base.add([dropoutLayer], ["RNN-dropout"])

    member _.hidden
        with get () = hs.value
        and set v = hs.value <- v

    override _.getString() = sprintf "RNN(%A, %A, numLayers:%A, bidirectional:%A)" inFeatures outFeatures numLayers bidirectional

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


type LSTM(inFeatures, outFeatures, ?numLayers, ?bias, ?batchFirst, ?dropout, ?bidirectional) =
    inherit Model()
    let numLayers = defaultArg numLayers 1
    let dropout = defaultArg dropout 0.
    let bidirectional = defaultArg bidirectional false
    let batchFirst = defaultArg batchFirst false
    let numDirections = if bidirectional then 2 else 1
    let makeLayers () = Array.init numLayers (fun i -> if i = 0 then LSTMCell(inFeatures, outFeatures, ?bias=bias) else LSTMCell(outFeatures, outFeatures, ?bias=bias))
    let layers = makeLayers()
    let layersReverse = if bidirectional then makeLayers() else [||]
    let dropoutLayer = Dropout(dropout)
    let hs = Parameter <| dsharp.tensor([]) // Not a parameter to be trained, it is for keeping hidden state
    let cs = Parameter <| dsharp.tensor([]) // Not a parameter to be trained, it is for keeping hidden state
    do 
        base.add(layers |> Array.map box, Array.init numLayers (fun i -> sprintf "LSTM-layer-%A" i))
        if bidirectional then base.add(layersReverse |> Array.map box, Array.init numLayers (fun i -> sprintf "LSTM-layer-reverse-%A" i))
        if dropout > 0. then base.add([dropoutLayer], ["LSTM-dropout"])

    member _.hidden
        with get () = hs.value
        and set v = hs.value <- v

    member _.cell
        with get () = cs.value
        and set v = cs.value <- v

    override _.getString() = sprintf "LSTM(%A, %A, numLayers:%A, bidirectional:%A)" inFeatures outFeatures numLayers bidirectional

    member r.reset() =
        r.hidden <- dsharp.tensor([])
        r.cell <- dsharp.tensor([])

    override r.forward(value) =
        let value, _, batchSize = rnnShape value inFeatures batchFirst
        if r.hidden.nelement = 0 then r.hidden <- dsharp.zeros([numLayers*numDirections; batchSize; outFeatures])
        if r.cell.nelement = 0 then r.cell <- dsharp.zeros([numLayers*numDirections; batchSize; outFeatures])
        let newhs = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let newcs = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let mutable hFwd = value
        for i in 0..numLayers-1 do 
            layers.[i].hidden <- r.hidden.[i]
            layers.[i].cell <- r.cell.[i]
            hFwd <- layers.[i].forward(hFwd)
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.forward(hFwd)
            newhs.[i] <- layers.[i].hidden
            newcs.[i] <- layers.[i].cell
        let output = 
            if bidirectional then
                let mutable hRev = value.flip([0])
                for i in 0..numLayers-1 do 
                    layersReverse.[i].hidden <- r.hidden.[numLayers+i]
                    layersReverse.[i].cell <- r.cell.[numLayers+i]
                    hRev <- layersReverse.[i].forward(hRev)
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.forward(hRev)
                    newhs.[numLayers+i] <- layersReverse.[i].hidden
                    newcs.[numLayers+i] <- layersReverse.[i].cell
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        r.hidden <- dsharp.stack(newhs)
        r.cell <- dsharp.stack(newcs)
        if batchFirst then output.transpose(0, 1) else output


type TextDataset(text:string, seqLength, ?chars) =
    inherit Dataset()
    // """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ """
    let _chars = (defaultArg chars text) |> Seq.distinct |> Seq.toArray |> Array.sort
    let onehot = memoize dsharp.onehot
    let _charToIndex = memoize (fun c -> try Array.findIndex ((=) c) _chars with _ -> failwithf "Character %A not found in this TextDataset (chars: %A)" c _chars)
    let _indexToChar(index) = _chars.[index]
    let textToIndices(text:string) = text |> Seq.map _charToIndex |> Seq.toArray
    let indicesToTensor(indices) = indices |> Array.map (fun i -> onehot(_chars.Length, i)) |> dsharp.stack
    let sequences = 
        if seqLength > text.Length then failwithf "Expecting text.Length (%A) >= seqLength (%A)" text.Length seqLength
        [|for i in 0..(text.Length - seqLength + 1)-1 do text.Substring(i, seqLength)|] |> Array.map textToIndices

    member d.indexToChar(i) = _indexToChar(i)
    member d.charToIndex(c) = _charToIndex(c)
    member d.textToTensor(text:string) = text |> textToIndices |> indicesToTensor
    member d.tensorToText(tensor:Tensor) =
        if tensor.dim <> 2 then failwithf "Expecting a 2d tensor with shape seqLen x features, received tensor with shape %A" tensor.shape 
        let t2text (tens:Tensor) = [|for i in 0..tens.shape.[0]-1 do tens.[i].argmax().[0]|] |> Array.map _indexToChar |> System.String |> string
        tensor |> t2text

    member d.chars = _chars
    member d.numChars = _chars.Length
    override d.length = sequences.Length
    override d.item(i) =
        let data = sequences.[i] |> indicesToTensor
        let target = sequences.[i] |> dsharp.tensor
        data, target

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

print languageModel

let modelFileName = "rnn_language_model.params"
if File.Exists(modelFileName) then 
    printfn "Resuming training from existing model params found: %A" modelFileName
    languageModel.loadParameters(modelFileName)

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

            languageModel.saveParameters(modelFileName)

            let plt = Pyplot()
            plt.plot(losses |> dsharp.tensor)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.tightLayout()
            plt.savefig (sprintf "rnn_loss_epoch_%A_minibatch_%A.pdf" epoch (i+1))