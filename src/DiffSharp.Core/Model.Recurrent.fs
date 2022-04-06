// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp

module RecurrentShape =
    let RNNCell (input:Tensor) inputSize =
        // input: [batchSize, inputSize]
        if input.dim <> 2 then failwithf "Expecting the input to be of shape [batchSize; inputSize] but got %A" input.shape
        if input.shape[1] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[1]

    let RNNCellSequence (input:Tensor) inputSize =
        // input: [seqLen, batchSize, inputSize]
        if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
        if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]

    let RNNCellWithHidden (input:Tensor) (hidden:Tensor) inputSize hiddenSize =
        // input: [batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        if input.dim <> 2 then failwithf "Expecting the input to be of shape [batchSize; inputSize] but got %A" input.shape
        if hidden.dim <> 2 then failwithf "Expecting the hidden to be of shape [batchSize; hiddenSize] but got %A" hidden.shape
        if input.shape[0] <> hidden.shape[0] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[0]
        if input.shape[1] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[1]
        if hidden.shape[1] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[1]

    let RNNCellSequenceWithHidden (input:Tensor) (hidden:Tensor) inputSize hiddenSize =
        // input: [seqLen, batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
        if hidden.dim <> 2 then failwithf "Expecting the hidden to be of shape [batchSize; hiddenSize] but got %A" hidden.shape
        if input.shape[1] <> hidden.shape[0] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[0]
        if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]
        if hidden.shape[1] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[1]

    let RNN (input:Tensor) inputSize batchFirst =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        if batchFirst then
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [batchSize; sequenceLength; inputSize] but got %A" input.shape
        else
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
        if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]

    let RNNWithHidden (input:Tensor) (hidden:Tensor) inputSize hiddenSize batchFirst numLayers numDirections =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // hidden: [numLayers*numDirections, batchSize, hiddenSize]
        if hidden.dim <> 3 then failwithf "Expecting the hidden to be of shape [numLayers*numDirections; batchSize; hiddenSize] but got %A" hidden.shape
        if batchFirst then
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [batchSize; sequenceLength; inputSize] but got %A" input.shape
            if input.shape[0] <> hidden.shape[1] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[1]
        else
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
            if input.shape[1] <> hidden.shape[1] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[1] hidden.shape[1]

        if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]
        if hidden.shape[2] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[2]
        if hidden.shape[0] <> numLayers*numDirections then failwithf "Expecting the hidden shape[0] to be %A but got %A" (numLayers*numDirections) hidden.shape[0]

    let LSTMCellWithHidden (input:Tensor) (hidden:Tensor) (cell:Tensor) inputSize hiddenSize =
        // input: [batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // cell: [batchSize, hiddenSize]
        if input.dim <> 2 then failwithf "Expecting the input to be of shape [batchSize; inputSize] but got %A" input.shape
        if hidden.dim <> 2 then failwithf "Expecting the hidden to be of shape [batchSize; hiddenSize] but got %A" hidden.shape
        if input.shape[0] <> hidden.shape[0] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[0]
        if input.shape[1] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[1]
        if hidden.shape[1] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[1]
        if hidden.shape <> cell.shape then failwithf "Expecting the hidden and cell to have the same shape but got %A and %A" hidden.shape cell.shape

    let LSTMCellSequenceWithHidden (input:Tensor) (hidden:Tensor) (cell:Tensor) inputSize hiddenSize =
        // input: [seqLen, batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // cell: [batchSize, hiddenSize]
        if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
        if hidden.dim <> 2 then failwithf "Expecting the hidden to be of shape [batchSize; hiddenSize] but got %A" hidden.shape
        if input.shape[1] <> hidden.shape[0] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[0]
        if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]
        if hidden.shape[1] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[1]
        if hidden.shape <> cell.shape then failwithf "Expecting the hidden and cell to have the same shape but got %A and %A" hidden.shape cell.shape

    let LSTMWithHidden (input:Tensor) (hidden:Tensor) (cell:Tensor) inputSize hiddenSize batchFirst numLayers numDirections =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // hidden: [numLayers*numDirections, batchSize, hiddenSize]
        // cell: [numLayers*numDirections, batchSize, hiddenSize]
        if hidden.dim <> 3 then failwithf "Expecting the hidden to be of shape [numLayers*numDirections; batchSize; hiddenSize] but got %A" hidden.shape
        if cell.dim <> 3 then failwithf "Expecting the cell to be of shape [numLayers*numDirections; batchSize; hiddenSize] but got %A" cell.shape
        if batchFirst then
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [batchSize; sequenceLength; inputSize] but got %A" input.shape
            if input.shape[0] <> hidden.shape[1] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[0] hidden.shape[1]
            if input.shape[0] <> cell.shape[1] then failwithf "Expecting the batch size to be the same for the input and cell but got %A and %A" input.shape[0] cell.shape[1]
        else
            if input.dim <> 3 then failwithf "Expecting the input to be of shape [sequenceLength; batchSize; inputSize] but got %A" input.shape
            if input.shape[1] <> hidden.shape[1] then failwithf "Expecting the batch size to be the same for the input and hidden but got %A and %A" input.shape[1] hidden.shape[1]
            if input.shape[1] <> cell.shape[1] then failwithf "Expecting the batch size to be the same for the input and cell but got %A and %A" input.shape[1] cell.shape[1]
        if input.shape[2] <> inputSize then failwithf "Expecting the input size to be %A but got %A" inputSize input.shape[2]
        if hidden.shape[2] <> hiddenSize then failwithf "Expecting the hidden size to be %A but got %A" hiddenSize hidden.shape[2]
        if hidden.shape[0] <> numLayers*numDirections then failwithf "Expecting the hidden shape[0] to be %A but got %A" (numLayers*numDirections) hidden.shape[0]
        if cell.shape[2] <> hiddenSize then failwithf "Expecting the cell size to be %A but got %A" hiddenSize cell.shape[2]
        if cell.shape[0] <> numLayers*numDirections then failwithf "Expecting the cell shape[0] to be %A but got %A" (numLayers*numDirections) cell.shape[0]


/// <summary>Unit cell of a recurrent neural network. Prefer using the RNN class instead, which can combine RNNCells in multiple layers.</summary>
type RNNCell(inputSize, hiddenSize, ?nonlinearity, ?bias, ?checkShapes) =
    inherit Model()
    let checkShapes = defaultArg checkShapes true
    let nonlinearity = defaultArg nonlinearity dsharp.tanh
    let bias = defaultArg bias true
    let k = 1./sqrt (float hiddenSize)
    let wih = Parameter(Weight.uniform([inputSize; hiddenSize], k))
    let whh = Parameter(Weight.uniform([hiddenSize; hiddenSize], k))
    let b = Parameter(if bias then Weight.uniform([hiddenSize], k) else dsharp.empty())
    do base.addParameter((wih, "RNNCell-weight-ih"), (whh, "RNNCell-weight-hh"), (b, "RNNCell-bias"))

    member _.inputSize = inputSize
    member _.hiddenSize = hiddenSize

    member _.newHidden(batchSize) =
        dsharp.zeros([batchSize; hiddenSize])

    override _.ToString() = sprintf "RNNCell(%A, %A)" inputSize hiddenSize

    override r.forward(input) =
        // input: [batchSize, inputSize]
        // returns: [batchSize, hiddenSize]    
        if checkShapes then RecurrentShape.RNNCell input inputSize
        let batchSize = input.shape[0]
        let hidden = r.newHidden(batchSize)
        r.forwardWithHidden(input, hidden)

    member r.forwardSequence(input) =
        // input: [seqLen, batchSize, inputSize]
        // returns: [seqLen, batchSize, hiddenSize]    
        if checkShapes then RecurrentShape.RNNCellSequence input inputSize
        let batchSize = input.shape[0]
        let hidden = r.newHidden(batchSize)
        r.forwardSequenceWithHidden(input, hidden)

    member _.forwardWithHidden(input:Tensor, hidden:Tensor) =
        // input: [batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // returns: [batchSize, hiddenSize]
        if checkShapes then RecurrentShape.RNNCellWithHidden input hidden inputSize hiddenSize
        let h = dsharp.matmul(input, wih.value) + dsharp.matmul(hidden, whh.value)
        let h = if bias then h + b.value else h
        let h = nonlinearity(h)
        h

    member r.forwardSequenceWithHidden(input:Tensor, hidden:Tensor) =
        // input: [seqLen, batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // returns: [seqLen, batchSize, hiddenSize]
        RecurrentShape.RNNCellSequenceWithHidden input hidden inputSize hiddenSize
        let seqLen = input.shape[0]
        let output = Array.create seqLen (dsharp.empty())
        for i = 0 to seqLen-1 do
            let h = if i = 0 then hidden else output[i-1]
            output[i] <- r.forwardWithHidden(input[i], h)
        dsharp.stack(output)

/// <summary>Unit cell of a long short-term memory (LSTM) recurrent neural network. Prefer using the RNN class instead, which can combine RNNCells in multiple layers.</summary>
type LSTMCell(inputSize, hiddenSize, ?bias, ?checkShapes) =
    inherit Model()
    let checkShapes = defaultArg checkShapes true
    let bias = defaultArg bias true
    let k = 1./sqrt (float hiddenSize)
    let wih = Parameter(Weight.uniform([inputSize; hiddenSize*4], k))
    let whh = Parameter(Weight.uniform([hiddenSize; hiddenSize*4], k))
    let b = Parameter(if bias then Weight.uniform([hiddenSize*4], k) else dsharp.tensor([]))
    do base.addParameter((wih, "LSTMCell-weight-ih"), (whh, "LSTMCell-weight-hh"), (b, "LSTMCell-bias"))

    member _.inputSize = inputSize
    member _.hiddenSize = hiddenSize

    member _.newHidden(batchSize) =
        dsharp.zeros([batchSize; hiddenSize])

    override _.ToString() = sprintf "LSTMCell(%A, %A)" inputSize hiddenSize

    override r.forward(input) =
        // input: [batchSize, inputSize]
        // returns: [batchSize, hiddenSize]    
        if checkShapes then RecurrentShape.RNNCell input inputSize
        let batchSize = input.shape[0]
        let hidden = r.newHidden(batchSize)
        let cell = r.newHidden(batchSize)
        r.forwardWithHidden(input, hidden, cell) |> fst

    member r.forwardSequence(input) =
        // input: [seqLen, batchSize, inputSize]
        // returns: [seqLen, batchSize, hiddenSize]    
        if checkShapes then RecurrentShape.RNNCellSequence input inputSize
        let batchSize = input.shape[0]
        let hidden = r.newHidden(batchSize)
        let cell = r.newHidden(batchSize)
        r.forwardSequenceWithHidden(input, hidden, cell)

    member r.forwardWithHidden(input:Tensor, hidden:Tensor, cell:Tensor) =
        // input: [batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // cell: [batchSize, hiddenSize]
        // returns: [batchSize, hiddenSize], [batchSize, hiddenSize]
        if checkShapes then RecurrentShape.LSTMCellWithHidden input hidden cell inputSize hiddenSize
        let x2h = dsharp.matmul(input, wih.value)
        let h2h = dsharp.matmul(hidden, whh.value)
        let mutable pre = x2h + h2h
        if bias then pre <- pre + b.value
        let pretan = pre[*,..hiddenSize-1].tanh()
        let presig = pre[*,hiddenSize..].sigmoid()
        let inputGate = presig[*,..hiddenSize-1]
        let forgetGate = presig[*,hiddenSize..(2*hiddenSize)-1]
        let outputGate = presig[*,(2*hiddenSize)..]
        let cell = (inputGate*pretan) + (forgetGate*cell)
        let hidden = outputGate*cell.tanh()
        hidden, cell

    member r.forwardSequenceWithHidden(input:Tensor, hidden:Tensor, cell:Tensor) =
        // input: [seqLen, batchSize, inputSize]
        // hidden: [batchSize, hiddenSize]
        // cell: [batchSize, hiddenSize]
        // returns: [seqLen, batchSize, hiddenSize], [seqLen, batchSize, hiddenSize]
        RecurrentShape.LSTMCellSequenceWithHidden input hidden cell inputSize hiddenSize
        let seqLen = input.shape[0]
        let hs = Array.create seqLen (dsharp.empty())
        let cs = Array.create seqLen (dsharp.empty())
        for i = 0 to seqLen-1 do
            let h, c = if i = 0 then hidden, cell else hs[i-1], cs[i-1]
            let h, c = r.forwardWithHidden(input[i], h, c)
            hs[i] <- h
            cs[i] <- c
        dsharp.stack(hs), dsharp.stack(cs) 


/// <summary>Recurrent neural network.</summary>
type RNN(inputSize, hiddenSize, ?numLayers, ?nonlinearity, ?bias, ?batchFirst, ?dropout, ?bidirectional) =
    inherit Model()
    let numLayers = defaultArg numLayers 1
    let dropout = defaultArg dropout 0.
    let bidirectional = defaultArg bidirectional false
    let batchFirst = defaultArg batchFirst false
    let numDirections = if bidirectional then 2 else 1
    let makeLayers () = Array.init numLayers (fun i -> if i = 0 then RNNCell(inputSize, hiddenSize, ?nonlinearity=nonlinearity, ?bias=bias, checkShapes=false) else RNNCell(hiddenSize, hiddenSize, ?nonlinearity=nonlinearity, ?bias=bias, checkShapes=false))
    let layers = makeLayers()
    let layersReverse = if bidirectional then makeLayers() else [||]
    let dropoutLayer = Dropout(dropout)
    do
        base.addModel(layers |> Array.mapi (fun i l -> l :> ModelBase, sprintf "RNN-layer-%A" i))
        if bidirectional then base.addModel(layersReverse |> Array.mapi (fun i l -> l :> ModelBase, sprintf "RNN-layer-reverse-%A" i))
        if dropout > 0. then base.addModel(dropoutLayer, "RNN-dropout")

    member _.inputSize = inputSize
    member _.hiddenSize = hiddenSize

    member _.newHidden(batchSize) =
        dsharp.zeros([numLayers*numDirections; batchSize; hiddenSize])

    override _.ToString() = sprintf "RNN(%A, %A, numLayers:%A, bidirectional:%A)" inputSize hiddenSize numLayers bidirectional

    override r.forward(input) =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // returns: [seqLen, batchSize, hiddenSize] or [batchSize, seqLen, hiddenSize]
        RecurrentShape.RNN input inputSize batchFirst
        let batchSize = if batchFirst then input.shape[0] else input.shape[1]
        let hidden = r.newHidden(batchSize)
        r.forwardWithHidden(input, hidden) |> fst

    member _.forwardWithHidden(input, hidden) =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // hidden: [numLayers*numDirections, batchSize, hiddenSize]
        // returns: {[seqLen, batchSize, hiddenSize] or [batchSize, seqLen, hiddenSize]}, [numLayers*numDirections, batchSize, hiddenSize]
        RecurrentShape.RNNWithHidden input hidden inputSize hiddenSize batchFirst numLayers numDirections
        let input = if batchFirst then input.transpose(0, 1) else input
        // input: [seqLen, batchSize, inputSize]
        let outHidden = Array.create (numLayers*numDirections) (dsharp.empty())
        let mutable hFwd = input
        for i in 0..numLayers-1 do
            let h = hidden[i]
            hFwd <- layers[i].forwardSequenceWithHidden(hFwd, h)
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.forward(hFwd)
            outHidden[i] <- hFwd[-1]
        let output =
            if bidirectional then
                let mutable hRev = input.flip([0])
                for i in 0..numLayers-1 do
                    let h = hidden[numLayers+i]
                    hRev <- layersReverse[i].forwardSequenceWithHidden(hRev, h)
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.forward(hRev)
                    outHidden[numLayers+i] <- hRev[-1]
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        let output = if batchFirst then output.transpose(0, 1) else output
        let outHidden = dsharp.stack(outHidden)
        output, outHidden


/// <summary>Long short-term memory (LSTM) recurrent neural network.</summary>
type LSTM(inputSize, hiddenSize, ?numLayers, ?bias, ?batchFirst, ?dropout, ?bidirectional) =
    inherit Model()
    let numLayers = defaultArg numLayers 1
    let dropout = defaultArg dropout 0.
    let bidirectional = defaultArg bidirectional false
    let batchFirst = defaultArg batchFirst false
    let numDirections = if bidirectional then 2 else 1
    let makeLayers () = Array.init numLayers (fun i -> if i = 0 then LSTMCell(inputSize, hiddenSize, ?bias=bias, checkShapes=false) else LSTMCell(hiddenSize, hiddenSize, ?bias=bias, checkShapes=false))
    let layers = makeLayers()
    let layersReverse = if bidirectional then makeLayers() else [||]
    let dropoutLayer = Dropout(dropout)
    do 
        base.addModel(layers |> Array.mapi (fun i l -> l :>ModelBase, sprintf "LSTM-layer-%A" i))
        if bidirectional then base.addModel(layersReverse |> Array.mapi (fun i l -> l :>ModelBase, sprintf "LSTM-layer-reverse-%A" i))    
        if dropout > 0. then base.addModel(dropoutLayer, "LSTM-dropout")

    member _.inputSize = inputSize
    member _.hiddenSize = hiddenSize

    member _.newHidden(batchSize) =
        dsharp.zeros([numLayers*numDirections; batchSize; hiddenSize])

    override _.ToString() = sprintf "LSTM(%A, %A, numLayers:%A, bidirectional:%A)" inputSize hiddenSize numLayers bidirectional

    override r.forward(input) =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // returns: [seqLen, batchSize, hiddenSize] or [batchSize, seqLen, hiddenSize]
        RecurrentShape.RNN input inputSize batchFirst
        let batchSize = if batchFirst then input.shape[0] else input.shape[1]
        let hidden = r.newHidden(batchSize)
        let cell = r.newHidden(batchSize)
        let output, _, _ = r.forwardWithHidden(input, hidden, cell)
        output

    member r.forwardWithHidden(input, hidden, cell) =
        // input: [seqLen, batchSize, inputSize] or [batchSize, seqLen, inputSize]
        // hidden: [numLayers*numDirections, batchSize, hiddenSize]
        // returns: {[seqLen, batchSize, hiddenSize] or [batchSize, seqLen, hiddenSize]}, [numLayers*numDirections, batchSize, hiddenSize], [numLayers*numDirections, batchSize, hiddenSize]
        RecurrentShape.LSTMWithHidden input hidden cell inputSize hiddenSize batchFirst numLayers numDirections
        let input = if batchFirst then input.transpose(0, 1) else input
        // input: [seqLen, batchSize, inputSize]
        let outHidden = Array.create (numLayers*numDirections) (dsharp.empty())
        let outCell = Array.create (numLayers*numDirections) (dsharp.empty())
        let mutable hFwd = input
        for i in 0..numLayers-1 do
            let h = hidden[i]
            let c = cell[i]
            let h, c = layers[i].forwardSequenceWithHidden(hFwd, h, c)
            hFwd <-h
            if dropout > 0. && i < numLayers-1 then hFwd <- dropoutLayer.forward(hFwd)
            outHidden[i] <- h[-1]
            outCell[i] <- c[-1]
        let output =
            if bidirectional then
                let mutable hRev = input.flip([0])
                for i in 0..numLayers-1 do
                    let h = hidden[numLayers+i]
                    let c = cell[numLayers+i]
                    let h, c = layersReverse[i].forwardSequenceWithHidden(hRev, h, c)
                    hRev <- h
                    if dropout > 0. && i < numLayers-1 then hRev <- dropoutLayer.forward(hRev)
                    outHidden[numLayers+i] <- h[-1]
                    outCell[numLayers+i] <- c[-1]
                dsharp.cat([hFwd; hRev], 2)
            else hFwd
        let output = if batchFirst then output.transpose(0, 1) else output
        let outHidden = dsharp.stack(outHidden)
        let outCell = dsharp.stack(outCell)
        output, outHidden, outCell