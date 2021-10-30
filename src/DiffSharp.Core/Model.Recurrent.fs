// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp

[<AutoOpen>]
module ModelRecurrentAutoOpens =
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


/// <summary>Unit cell of a recurrent neural network. Prefer using the RNN class instead, which can combine RNNCells in multiple layers.</summary>
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
    do base.addParameter([wih;whh;b],["RNNCell-weight-ih";"RNNCell-weight-hh";"RNNCell-bias"])
    do base.addBuffer([h], ["RNNCell-hidden"])

    member _.hidden 
        with get () = h.value
        and set v = h.value <- v

    override _.ToString() = sprintf "RNNCell(%A, %A)" inFeatures outFeatures

    member r.reset() = r.hidden <- dsharp.tensor([])

    override r.run(value) =
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

/// <summary>Unit cell of a long short-term memory (LSTM) recurrent neural network. Prefer using the RNN class instead, which can combine RNNCells in multiple layers.</summary>
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
    do base.addParameter([wih;whh;b],["LSTMCell-weight-ih";"LSTMCell-weight-hh";"LSTMCell-bias"])
    do base.addBuffer([h;c],["LSTMCell-hidden";"LSTMCell-cell"])

    member _.hidden 
        with get () = h.value
        and set v = h.value <- v

    member _.cell
        with get () = c.value
        and set v = c.value <- v

    override _.ToString() = sprintf "LSTMCell(%A, %A)" inFeatures outFeatures

    member r.reset() = r.hidden <- dsharp.tensor([])

    override r.run(value) =
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

/// <summary>Recurrent neural network.</summary>
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

    override _.ToString() = sprintf "RNN(%A, %A, numLayers:%A, bidirectional:%A)" inFeatures outFeatures numLayers bidirectional

    member r.reset() = r.hidden <- dsharp.tensor([])

    override r.run(value) =
        let value, _, batchSize = rnnShape value inFeatures batchFirst
        if r.hidden.nelement = 0 then r.hidden <- dsharp.zeros([numLayers*numDirections; batchSize; outFeatures])
        let newhs = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let mutable hFwd = value
        for i in 0..numLayers-1 do 
            layers.[i].hidden <- r.hidden.[i]
            hFwd <- layers.[i].run(hFwd)
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.run(hFwd)
            newhs.[i] <- layers.[i].hidden
        let output = 
            if bidirectional then
                let mutable hRev = value.flip([0])
                for i in 0..numLayers-1 do 
                    layersReverse.[i].hidden <- r.hidden.[numLayers+i]
                    hRev <- layersReverse.[i].run(hRev)
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.run(hRev)
                    newhs.[numLayers+i] <- layersReverse.[i].hidden
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        r.hidden <- dsharp.stack(newhs)
        if batchFirst then output.transpose(0, 1) else output


/// <summary>Long short-term memory (LSTM) recurrent neural network.</summary>
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

    override _.ToString() = sprintf "LSTM(%A, %A, numLayers:%A, bidirectional:%A)" inFeatures outFeatures numLayers bidirectional

    member r.reset() =
        r.hidden <- dsharp.tensor([])
        r.cell <- dsharp.tensor([])

    override r.run(value) =
        let value, _, batchSize = rnnShape value inFeatures batchFirst
        if r.hidden.nelement = 0 then r.hidden <- dsharp.zeros([numLayers*numDirections; batchSize; outFeatures])
        if r.cell.nelement = 0 then r.cell <- dsharp.zeros([numLayers*numDirections; batchSize; outFeatures])
        let newhs = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let newcs = Array.create (numLayers*numDirections) (dsharp.tensor([]))
        let mutable hFwd = value
        for i in 0..numLayers-1 do 
            layers.[i].hidden <- r.hidden.[i]
            layers.[i].cell <- r.cell.[i]
            hFwd <- layers.[i].run(hFwd)
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.run(hFwd)
            newhs.[i] <- layers.[i].hidden
            newcs.[i] <- layers.[i].cell
        let output = 
            if bidirectional then
                let mutable hRev = value.flip([0])
                for i in 0..numLayers-1 do 
                    layersReverse.[i].hidden <- r.hidden.[numLayers+i]
                    layersReverse.[i].cell <- r.cell.[numLayers+i]
                    hRev <- layersReverse.[i].run(hRev)
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.run(hRev)
                    newhs.[numLayers+i] <- layersReverse.[i].hidden
                    newcs.[numLayers+i] <- layersReverse.[i].cell
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        r.hidden <- dsharp.stack(newhs)
        r.cell <- dsharp.stack(newcs)
        if batchFirst then output.transpose(0, 1) else output