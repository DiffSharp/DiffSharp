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
type TestModelConv () =

    [<Test>]
    member _.TestModelConvTranspose1d () =
        let x = dsharp.randn([5; 3; 12])
        let m = ConvTranspose1d(3, 4, 3)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|5; 4; 14|]
        Assert.CheckEqual(yShapeCorrect, yShape)

        let x = dsharp.randn([3; 3; 12])
        let m = ConvTranspose1d(3, 5, 2, dilation=5)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|3; 5; 17|]
        Assert.CheckEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestModelConvTranspose2d () =
        let x = dsharp.randn([3; 3; 12; 6])
        let m = ConvTranspose2d(3, 5, 3)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|3; 5; 14; 8|]
        Assert.CheckEqual(yShapeCorrect, yShape)

        let x = dsharp.randn([2; 3; 12; 6])
        let m = ConvTranspose2d(3, 1, 5, stride=2)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|2; 1; 27; 15|]
        Assert.CheckEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestModelConvTranspose3d () =
        let x = dsharp.randn([2; 3; 12; 6; 6])
        let m = ConvTranspose3d(3, 2, 3)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|2; 2; 14; 8; 8|]
        Assert.CheckEqual(yShapeCorrect, yShape)

        let x = dsharp.randn([2; 3; 12; 6; 6])
        let m = ConvTranspose3d(3, 2, 2, padding=1)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|2; 2; 11; 5; 5|]
        Assert.CheckEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestModelConvTranspose1dSaveLoadState () =
        let inChannels = 4
        let outChannels = 4
        let kernelSize = 3
        let batchSize = 2
        let d = 5
        let net = ConvTranspose1d(inChannels, outChannels, kernelSize)

        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net.state, fileName)
        let _ = dsharp.randn([batchSize; inChannels; d]) --> net
        net.state <- dsharp.load(fileName)
        Assert.True(true)

    [<Test>]
    member _.TestModelConvTranspose2dSaveLoadState () =
        let inChannels = 4
        let outChannels = 4
        let kernelSize = 3
        let batchSize = 2
        let d = 5
        let net = ConvTranspose2d(inChannels, outChannels, kernelSize)

        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net.state, fileName)
        let _ = dsharp.randn([batchSize; inChannels; d; d]) --> net
        net.state <- dsharp.load(fileName)
        Assert.True(true)

    [<Test>]
    member _.TestModelConvTranspose3dSaveLoadState () =
        let inChannels = 4
        let outChannels = 4
        let kernelSize = 3
        let batchSize = 2
        let d = 5
        let net = ConvTranspose3d(inChannels, outChannels, kernelSize)

        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net.state, fileName)
        let _ = dsharp.randn([batchSize; inChannels; d; d; d]) --> net
        net.state <- dsharp.load(fileName)
        Assert.True(true)        