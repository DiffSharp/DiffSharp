// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp


[<TestFixture>]
type TestTensorDet () =
    [<Test>]
    member _.TestTensorDet () =
        for combo in Combos.FloatingPointExcept16s do
            let t3x3 = combo.tensor([[ 1.3038, -0.8699,  1.2059],
                                        [ 1.0837, -1.5076, -0.1286],
                                        [-0.9857,  0.3633, -1.0049]])
            let t3x3Det = t3x3.det()
            let t3x3DetCorrect = combo.tensor(-0.3387)

            Assert.True(t3x3DetCorrect.allclose(t3x3Det, 0.01))

            let t4x2x2 = combo.tensor([[[-2.1301, -1.4122],
                                         [-0.4353, -0.6708]],

                                        [[ 0.0696, -1.3661],
                                         [ 0.4162,  0.0663]],

                                        [[-1.3677, -0.6721],
                                         [ 0.6547,  0.5127]],

                                        [[-1.1081,  1.0203],
                                         [-0.1355,  0.0641]]])
            let t4x2x2Det = t4x2x2.det()
            let t4x2x2DetCorrect = combo.tensor([ 0.8141,  0.5732, -0.2612,  0.0672])

            Assert.True(t4x2x2DetCorrect.allclose(t4x2x2Det, 0.01))


[<TestFixture>]
type TestDerivativesDet () =
    [<Test>]
    member _.TestDerivativeDet () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[-0.1219,  1.4357,  0.3839],
                                        [-1.2608, -0.5778, -0.8679],
                                        [ 0.2116, -1.1607, -0.4967]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[ 0.6779, -0.0532,  0.1049],
                                                        [-0.0534, -0.3002, -0.7770],
                                                        [-1.3737, -0.4547,  0.1911]]))
            let fwdz = dsharp.det(fwdx)
            let fwdzCorrect = combo.tensor(-0.4662)
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor(1.6214)

            let revx = combo.tensor([[-0.1219,  1.4357,  0.3839],
                                        [-1.2608, -0.5778, -0.8679],
                                        [ 0.2116, -1.1607, -0.4967]]).reverseDiff()
            let revz = dsharp.det(revx)
            let revzCorrect = combo.tensor(-0.4662)
            revz.reverse(combo.tensor(1.3444))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[-0.9685, -1.0888,  2.1318],
                                                [ 0.3596, -0.0278,  0.2182],
                                                [-1.3770, -0.7929,  2.5283]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeDetBatched () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[[ 1.2799, -0.6491],
                                        [-1.4575,  2.0789]],

                                        [[-1.0350,  0.8558],
                                            [ 1.3920,  1.4445]],

                                        [[-2.0709,  0.2865],
                                            [ 1.0892,  0.5796]]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[[ 2.2755, -1.2585],
                                                        [ 0.6867,  0.9552]],

                                                        [[-0.6031, -0.1197],
                                                            [-1.5058, -0.3416]],

                                                        [[-1.1658,  0.4657],
                                                            [ 1.1314,  0.8895]]]))
            let fwdz = dsharp.det(fwdx)
            let fwdzCorrect = combo.tensor([ 1.7147, -2.6862, -1.5123])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([ 4.5646,  0.9376, -3.3492])

            let revx = combo.tensor([[[ 1.2799, -0.6491],
                                        [-1.4575,  2.0789]],

                                        [[-1.0350,  0.8558],
                                            [ 1.3920,  1.4445]],

                                        [[-2.0709,  0.2865],
                                            [ 1.0892,  0.5796]]]).reverseDiff()
            let revz = dsharp.det(revx)
            let revzCorrect = combo.tensor([ 1.7147, -2.6862, -1.5123])
            revz.reverse(combo.tensor([-0.1814,  1.2643,  1.5553]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[-0.3771, -0.2644],
                                                [-0.1177, -0.2321]],
                                        
                                                [[ 1.8262, -1.7598],
                                                    [-1.0819, -1.3085]],
                                        
                                                [[ 0.9014, -1.6940],
                                                    [-0.4456, -3.2208]]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))