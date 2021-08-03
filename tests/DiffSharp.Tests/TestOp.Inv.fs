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
            if combo.backend = Backend.Reference then
                Assert.True(true)
            else
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