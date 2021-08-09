// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp


[<TestFixture>]
type TestTensorSolve () =
    [<Test>]
    member _.TestTensorSolve () =
        for combo in Combos.FloatingPointExcept16s do
            let t3x3 = combo.tensor([[-1.1606,  0.6579,  1.0674],
                                     [-1.0226,  0.2406, -0.5414],
                                     [ 0.1195,  1.2423,  0.0889]])
            let t3 = combo.tensor([ 0.6791,  0.5497, -0.3624])
            let t3x1 = t3.unsqueeze(1)

            let t3x3Solvet3 = t3x3.solve(t3)
            let t3x3Solvet3Correct = combo.tensor([-0.6392, -0.2364,  0.0869])

            let t3x3Solvet3x1 = t3x3.solve(t3x1)
            let t3x3Solvet3x1Correct = t3x3Solvet3Correct.unsqueeze(1)

            Assert.True(t3x3Solvet3Correct.allclose(t3x3Solvet3, 0.01))
            Assert.True(t3x3Solvet3x1Correct.allclose(t3x3Solvet3x1, 0.01))

            let t3x2 = combo.tensor([[-1.0439,  0.9510],
                                        [-0.9118,  0.5726],
                                        [-0.3161,  0.1080]])

            let t3x3Solvet3x2 = t3x3.solve(t3x2)
            let t3x3Solvet3x2Correct = combo.tensor([[ 0.7754, -0.6068],
                                                        [-0.3341,  0.1347],
                                                        [ 0.0710,  0.1482]])

            Assert.True(t3x3Solvet3x2Correct.allclose(t3x3Solvet3x2, 0.01))

    [<Test>]
    member _.TestTensorSolveBatched () =
        for combo in Combos.FloatingPointExcept16s do
            let t4x2x2 = combo.tensor([[[-0.0184,  0.7381],
                                          [ 0.3093, -0.4847]],
                                 
                                         [[ 0.0368,  1.2592],
                                          [-0.1828, -0.2979]],
                                 
                                         [[-1.4190, -0.8507],
                                          [ 0.7187,  2.7166]],
                                 
                                         [[-0.2591, -1.3985],
                                          [ 1.7918,  1.2014]]])
            let t4x2 = combo.tensor([[ 1.0514, -1.0258],
                                     [ 0.0513,  0.0240],
                                     [-0.7468, -0.6901],
                                     [ 1.7193, -1.5342]])
            let t4x2x1 = t4x2.unsqueeze(2)

            let t4x2x2Solvet4x2 = t4x2x2.solve(t4x2)
            let t4x2x2Solvet4x2Correct = combo.tensor([[-1.1286,  1.3964],
                                                        [-0.2075,  0.0468],
                                                        [ 0.8065, -0.4674],
                                                        [-0.0365, -1.2226]])

            let t4x2x2Solvet4x2x1 = t4x2x2.solve(t4x2x1)
            let t4x2x2Solvet4x2x1Correct = t4x2x2Solvet4x2Correct.unsqueeze(2)

            Assert.True(t4x2x2Solvet4x2Correct.allclose(t4x2x2Solvet4x2, 0.01))
            Assert.True(t4x2x2Solvet4x2x1Correct.allclose(t4x2x2Solvet4x2x1, 0.01))

            let t4x2x3 = combo.tensor([[[-0.6377,  0.2087, -0.8464],
                                         [-0.3898,  1.1024, -0.2743]],

                                        [[-0.9679,  0.0065, -0.8171],
                                         [-1.1492,  0.5241, -0.0426]],

                                        [[-1.9881,  1.1760,  1.2151],
                                         [ 1.0708, -0.0690, -1.1514]],

                                        [[-0.7452,  0.1944,  0.5586],
                                         [-0.2153,  0.0611, -0.3548]]])

            let t4x2x2Solvet4x2x3 = t4x2x2.solve(t4x2x3)
            let t4x2x2Solvet4x2x3Correct = combo.tensor([[[-2.7204,  4.1705, -2.7931],
                                                             [-0.9317,  0.3866, -1.2162]],

                                                            [[ 7.9153, -3.0187,  1.3551],
                                                             [-0.9997,  0.0933, -0.6885]],

                                                            [[ 1.3843, -0.9669, -0.7157],
                                                             [ 0.0279,  0.2304, -0.2345]],

                                                            [[-0.5451,  0.1453,  0.0797],
                                                             [ 0.6338, -0.1659, -0.4142]]])

            Assert.True(t4x2x2Solvet4x2x3Correct.allclose(t4x2x2Solvet4x2x3, 0.01))


[<TestFixture>]
type TestDerivativesSolve () =
    [<Test>]
    member _.TestDerivativeSolveTT () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[ 1.3299,  1.5288],
                                        [-1.7740, -2.0062]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[-2.2180, -1.6499],
                                                        [-1.1382,  0.8492]]))
            let fwdy = combo.tensor([-2.1001, -0.5405])
            let fwdy = fwdy.forwardDiff(combo.tensor([-0.2762, -0.7975]))
            let fwdz = dsharp.solve(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([ 114.7422, -101.1898])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-11495.3730,  10057.0986])

            let revx = combo.tensor([[ 1.3299,  1.5288],
                                        [-1.7740, -2.0062]]).reverseDiff()
            let revy = combo.tensor([-2.1001, -0.5405]).reverseDiff()
            let revz = dsharp.solve(revx, revy)
            let revzCorrect = combo.tensor([ 114.7422, -101.1898])
            revz.reverse(combo.tensor([-1.3181,  0.8040]))            
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[-10634.0508,   9378.0439],
                                                [ -8057.3462,   7105.6787]])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([92.6778, 70.2213])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))

            printfn "revxd %A %A" revxd revxdCorrect
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

            printfn "revyd %A %A" revyd revydCorrect
            Assert.True(revyd.allclose(revydCorrect, 0.01))