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
type TestModelLinear () =

    [<Test>]
    member _.TestModelLinear () =
        // Trains a linear regressor
        let n, din, dout = 4, 100, 10
        let inputs  = dsharp.randn([n; din])
        let targets = dsharp.randn([n; dout])
        let net = Linear(din, dout)

        let lr, steps = 1e-2, 1000
        let loss inputs p = net.asFunction p inputs |> dsharp.mseLoss targets
        for _ in 0..steps do
            let g = dsharp.grad (loss inputs) net.parametersVector
            net.parametersVector <- net.parametersVector - lr * g
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.01))

    [<Test>]
    member _.TestModelLinearSaveLoadState () =
        let inFeatures = 4
        let outFeatures = 4
        let batchSize = 2
        let net = Linear(inFeatures, outFeatures)

        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net.state, fileName)
        let _ = dsharp.randn([batchSize; inFeatures]) --> net
        net.state <- dsharp.load(fileName)
        Assert.True(true)