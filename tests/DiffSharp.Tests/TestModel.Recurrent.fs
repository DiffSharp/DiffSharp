// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim


[<TestFixture>]
type TestModelRecurrent () =

    [<Test>]
    member _.TestModelRNN () =
        let din = 8
        let dout = 10
        let seqLen = 4
        let batchSize = 16
        let numLayers = 3
        let numDirections = 1

        // Seq first
        let input = dsharp.randn([seqLen; batchSize; din])
        let rnn = RNN(din, dout, numLayers=numLayers, bidirectional=false)
        let output = input --> rnn
        let outputShape = output.shape
        let outputShapeCorrect = [|seqLen; batchSize; dout|]
        Assert.AreEqual(outputShapeCorrect, outputShape)

        // Batch first
        let input = dsharp.randn([batchSize; seqLen; din])
        let rnn = RNN(din, dout, numLayers=numLayers, batchFirst=true, bidirectional=false)
        let output = input --> rnn
        let outputShape = output.shape
        let outputShapeCorrect = [|batchSize; seqLen; dout|]
        Assert.AreEqual(outputShapeCorrect, outputShape)

        let hiddenShape = rnn.newHidden(batchSize).shape
        let hiddenShapeCorrect = [|numLayers*numDirections; batchSize; dout|]
        Assert.AreEqual(hiddenShapeCorrect, hiddenShape)

        let steps = 64
        let lr = 0.01
        let optimizer = Adam(rnn, lr=dsharp.tensor(lr))
        let target = dsharp.randn([batchSize; seqLen; dout])
        let output = input --> rnn
        let mutable loss = dsharp.mseLoss(output, target)
        let loss0 = float loss

        for i in 1..steps do
            rnn.reverseDiff()
            let output = input --> rnn
            loss <- dsharp.mseLoss(output, target)
            loss.reverse()
            optimizer.step()
        let lossFinal = float loss

        Assert.Less(lossFinal, loss0/2.)

    [<Test>]
    member _.TestModelLSTM () =
        let din = 8
        let dout = 10
        let seqLen = 4
        let batchSize = 16
        let numLayers = 2
        let numDirections = 1

        // Seq first
        let input = dsharp.randn([seqLen; batchSize; din])
        let lstm = LSTM(din, dout, numLayers=numLayers, bidirectional=false)
        let output = input --> lstm
        let outputShape = output.shape
        let outputShapeCorrect = [|seqLen; batchSize; dout|]
        Assert.AreEqual(outputShapeCorrect, outputShape)

        // Batch first
        let input = dsharp.randn([batchSize; seqLen; din])
        let lstm = LSTM(din, dout, numLayers=numLayers, batchFirst=true, bidirectional=false)
        let output = input --> lstm
        let outputShape = output.shape
        let outputShapeCorrect = [|batchSize; seqLen; dout|]
        Assert.AreEqual(outputShapeCorrect, outputShape)

        let hiddenShape = lstm.newHidden(batchSize).shape
        let hiddenShapeCorrect = [|numLayers*numDirections; batchSize; dout|]
        Assert.AreEqual(hiddenShapeCorrect, hiddenShape)

        let steps = 128
        let lr = 0.01
        let optimizer = Adam(lstm, lr=dsharp.tensor(lr))
        let target = dsharp.randn([batchSize; seqLen; dout])
        let output = input --> lstm
        let mutable loss = dsharp.mseLoss(output, target)
        let loss0 = float loss

        for i in 1..steps do
            lstm.reverseDiff()
            let output = input --> lstm
            loss <- dsharp.mseLoss(output, target)
            loss.reverse()
            optimizer.step()
        let lossFinal = float loss

        Assert.Less(lossFinal, loss0/2.)

    [<Test>]
    member _.TestModelRNNSaveLoadState () =
        let net = RNN(10, 10)

        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net.state, fileName) // Save pre-use
        let _ = dsharp.randn([10; 10; 10]) --> net // Use
        net.state <- dsharp.load(fileName) // Load after-use

        Assert.True(true)

    [<Test>]
    member _.TestModelLSTMSaveLoadState () =
        let net = LSTM(10, 10)

        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net.state, fileName) // Save pre-use
        let _ = dsharp.randn([10; 10; 10]) --> net // Use
        net.state <- dsharp.load(fileName) // Load after-use

        Assert.True(true)
