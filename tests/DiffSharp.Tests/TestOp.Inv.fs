// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp


[<TestFixture>]
type TestTensorInv () =
    [<Test>]
    member _.TestTensorInv () =
        for combo in Combos.FloatingPointExcept16s do
            let t3x3 = combo.tensor([[ 1.3038, -0.8699,  1.2059],
                                        [ 1.0837, -1.5076, -0.1286],
                                        [-0.9857,  0.3633, -1.0049]])
            let t3x3Inv = t3x3.inv()
            let t3x3InvCorrect = combo.tensor([[-4.6103,  1.2872, -5.6974],
                                                [-3.5892,  0.3586, -4.3532],
                                                [ 3.2248, -1.1330,  3.0198]])

            Assert.True(t3x3InvCorrect.allclose(t3x3Inv, 0.01))

            let t4x2x2 = combo.tensor([[[-2.1301, -1.4122],
                                         [-0.4353, -0.6708]],

                                        [[ 0.0696, -1.3661],
                                         [ 0.4162,  0.0663]],

                                        [[-1.3677, -0.6721],
                                         [ 0.6547,  0.5127]],

                                        [[-1.1081,  1.0203],
                                         [-0.1355,  0.0641]]])
            let t4x2x2Inv = t4x2x2.inv()
            let t4x2x2InvCorrect = combo.tensor([[[ -0.8239,   1.7344],
                                                     [  0.5346,  -2.6162]],

                                                    [[  0.1156,   2.3836],
                                                     [ -0.7261,   0.1214]],

                                                    [[ -1.9629,  -2.5729],
                                                     [  2.5065,   5.2359]],

                                                    [[  0.9526, -15.1662],
                                                     [  2.0147, -16.4717]]])

            Assert.True(t4x2x2InvCorrect.allclose(t4x2x2Inv, 0.01))


[<TestFixture>]
type TestDerivativesInv () =
    [<Test>]
    member _.TestDerivativeInv () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[-0.4903,  1.2986],
                                        [-0.6858,  0.1967]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[-0.6098,  1.3503],
                                                        [-0.2191, -0.3666]]))
            let fwdz = dsharp.inv(fwdx)
            let fwdzCorrect = combo.tensor([[ 0.2476, -1.6352],
                                                [ 0.8635, -0.6173]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[-0.8578,  0.9154],
                                                [-1.1054,  0.2197]])

            let revx = combo.tensor([[-0.4903,  1.2986],
                                        [-0.6858,  0.1967]]).reverseDiff()
            let revz = dsharp.inv(revx)
            let revzCorrect = combo.tensor([[ 0.2476, -1.6352],
                                            [ 0.8635, -0.6173]])
            revz.reverse(combo.tensor([[ 0.2513, -1.0312],
                                        [ 0.3695,  0.5980]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[ 0.3324, -0.1681],
                                             [ 2.3118,  1.3648]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeInvBatched () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[[-0.9658, -0.7494],
                                         [ 0.0645,  0.4319]],

                                        [[-0.6887,  1.1664],
                                         [ 0.2725, -0.5800]],

                                        [[-1.6706, -0.4705],
                                         [-0.2973, -0.4069]],

                                        [[-0.3529,  1.1275],
                                         [-0.2755, -1.0010]]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[[ 1.0862, -0.5793],
                                                         [-0.1277, -0.0349]],

                                                        [[ 0.0311, -0.7146],
                                                         [ 0.5133,  0.0229]],

                                                        [[ 0.1031, -1.7459],
                                                         [-0.9217,  1.0681]],

                                                        [[-1.1179, -1.5689],
                                                         [ 0.9075, -0.8954]]]))
            let fwdz = dsharp.inv(fwdx)
            let fwdzCorrect = combo.tensor([[[ -1.1711,  -2.0320],
                                             [  0.1749,   2.6188]],

                                            [[ -7.1031, -14.2832],
                                             [ -3.3370,  -8.4342]],

                                            [[ -0.7537,   0.8717],
                                             [  0.5508,  -3.0948]],

                                            [[ -1.5079,  -1.6985],
                                             [  0.4150,  -0.5316]]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[[ -1.3170,  -4.0203],
                                              [ -0.1354,   0.2115]],
                                     
                                             [[-37.7996, -67.8218],
                                              [-24.1760, -44.8357]],
                                     
                                             [[ -1.9017,   7.7217],
                                              [  4.5430, -15.7415]],
                                     
                                             [[ -1.3953,   2.3112],
                                              [ -1.3544,  -1.7004]]])

            let revx = combo.tensor([[[-0.9658, -0.7494],
                                         [ 0.0645,  0.4319]],

                                        [[-0.6887,  1.1664],
                                         [ 0.2725, -0.5800]],

                                        [[-1.6706, -0.4705],
                                         [-0.2973, -0.4069]],

                                        [[-0.3529,  1.1275],
                                         [-0.2755, -1.0010]]]).reverseDiff()
            let revz = dsharp.inv(revx)
            let revzCorrect = combo.tensor([[[ -1.1711,  -2.0320],
                                              [  0.1749,   2.6188]],
                                     
                                             [[ -7.1031, -14.2832],
                                              [ -3.3370,  -8.4342]],
                                     
                                             [[ -0.7537,   0.8717],
                                              [  0.5508,  -3.0948]],
                                     
                                             [[ -1.5079,  -1.6985],
                                              [  0.4150,  -0.5316]]])
            revz.reverse(combo.tensor([[[-1.1475,  0.2692],
                                         [ 1.3103, -0.0504]],

                                        [[ 1.0963,  0.1170],
                                         [-2.1306,  1.1994]],

                                        [[-0.6898, -0.4653],
                                         [ 1.4485, -1.2765]],

                                        [[ 0.6352, -0.6512],
                                         [ 0.6127,  0.7157]]]))            
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[   1.1838,    0.5734],
                                              [   5.3701,    0.7696]],
                                     
                                             [[ -73.8503,  -43.0277],
                                              [-151.9466,  -91.7053]],
                                     
                                             [[   1.3004,   -1.8163],
                                              [  -6.9220,   13.7710]],
                                     
                                             [[   1.1114,    0.9719],
                                              [  -0.8855,    0.9685]]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
