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
    member _.TestDerivativeSolveT2T1 () =
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
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSolveT2T2 () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[ 0.2275,  1.1957],
                                        [ 0.3407, -0.0182]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[-1.3603,  0.8890],
                                                        [ 0.4982,  2.2654]]))
            let fwdy = combo.tensor([[ 2.6431,  1.2431, -1.6088],
                                        [ 0.9090,  0.7572,  0.6813]])
            let fwdy = fwdy.forwardDiff(combo.tensor([[ 0.7555,  1.5113, -0.4810],
                                                        [ 1.0133,  0.4124,  0.5419]]))
            let fwdz = dsharp.solve(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([[ 2.7578,  2.2547,  1.9083],
                                            [ 1.6859,  0.6107, -1.7085]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[-12.0109,  -5.9064,  10.2178],
                                                [  4.8008,   4.4986,   1.0950]])

            let revx = combo.tensor([[ 0.2275,  1.1957],
                                        [ 0.3407, -0.0182]]).reverseDiff()
            let revy = combo.tensor([[ 2.6431,  1.2431, -1.6088],
                                        [ 0.9090,  0.7572,  0.6813]]).reverseDiff()
            let revz = dsharp.solve(revx, revy)
            let revzCorrect = combo.tensor([[ 2.7578,  2.2547,  1.9083],
                                             [ 1.6859,  0.6107, -1.7085]])
            revz.reverse(combo.tensor([[-0.2177, -1.4992,  0.4552],
                                        [ 0.4530, -0.0503,  0.2671]]))            
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[-1.2250, -0.1382],
                                                [ 9.9519,  6.1387]])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[ 0.3655, -0.1078,  0.2412],
                                                [-0.8829, -4.3281,  1.1748]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSolveBatchedT3T2 () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[[-0.1101, -0.9294],
                                         [-1.3321, -0.5504]],

                                        [[-1.3495, -1.0639],
                                         [ 0.7233, -1.6117]],

                                        [[ 1.9579,  0.2827],
                                         [-0.5963,  0.3869]]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[[-0.4519,  0.6483],
                                                         [-1.5386, -0.0233]],

                                                        [[ 0.8457,  0.0209],
                                                         [-0.4935, -0.1902]],

                                                        [[ 0.0984, -0.8426],
                                                         [-1.4607,  0.1055]]]))
            let fwdy = combo.tensor([[-0.2632,  0.7203],
                                        [-0.3515,  0.3553],
                                        [ 1.1202,  0.4182]])
            let fwdy = fwdy.forwardDiff(combo.tensor([[ 1.2497,  0.2558],
                                                        [-0.1988,  0.5994],
                                                        [ 0.7058, -1.1530]]))
            let fwdz = dsharp.solve(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([[-0.6916,  0.3651],
                                             [ 0.3208, -0.0765],
                                             [ 0.3403,  1.6054]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[ 0.9587, -0.8671],
                                             [ 0.5250, -0.2255],
                                             [ 1.0979, -0.4412]])

            let revx = combo.tensor([[[-0.1101, -0.9294],
                                         [-1.3321, -0.5504]],

                                        [[-1.3495, -1.0639],
                                         [ 0.7233, -1.6117]],

                                        [[ 1.9579,  0.2827],
                                         [-0.5963,  0.3869]]]).reverseDiff()
            let revy = combo.tensor([[-0.2632,  0.7203],
                                        [-0.3515,  0.3553],
                                        [ 1.1202,  0.4182]]).reverseDiff()
            let revz = dsharp.solve(revx, revy)
            let revzCorrect = combo.tensor([[-0.6916,  0.3651],
                                             [ 0.3208, -0.0765],
                                             [ 0.3403,  1.6054]])
            revz.reverse(combo.tensor([[ 0.0959,  0.5113],
                                        [-1.3353,  2.0850],
                                        [-0.3518, -0.9607]]))            
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[-0.3690,  0.1948],
                                               [-0.0193,  0.0102]],
                                      
                                              [[-0.0702,  0.0167],
                                               [ 0.4612, -0.1100]],
                                      
                                              [[ 0.2605,  1.2290],
                                               [ 0.6547,  3.0883]]])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[-0.5336, -0.0279],
                                              [ 0.2187, -1.4380],
                                              [-0.7655, -1.9237]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSolveBatchedT3T3 () =
        for combo in Combos.FloatingPointExcept16s do
            let fwdx = combo.tensor([[[-1.1657,  0.6982],
                                         [ 1.3803,  1.3023]],

                                        [[-0.1657, -0.7590],
                                         [ 0.8820, -1.1975]],

                                        [[ 0.1085,  0.5656],
                                         [ 0.3442, -1.5360]]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[[-1.0915, -0.6823],
                                                         [ 1.2316, -0.3240]],

                                                        [[-1.0685, -0.2016],
                                                         [-0.4115,  1.8990]],

                                                        [[-0.2579, -1.7715],
                                                         [ 0.8489,  0.3538]]]))
            let fwdy = combo.tensor([[[-0.8791,  0.3885,  0.5178],
                                         [-0.7265, -0.1601, -0.6888]],

                                        [[ 1.4318,  0.8871,  0.1525],
                                         [ 0.3409,  0.9411,  1.0420]],

                                        [[ 1.7354,  1.4994,  0.3758],
                                         [-2.0791, -0.0783,  0.5615]]])
            let fwdy = fwdy.forwardDiff(combo.tensor([[[-0.8317, -0.2277,  1.6558],
                                                         [ 0.7668, -1.5930,  0.1161]],

                                                        [[ 0.4079, -0.0930, -0.5725],
                                                         [-0.5231, -0.3573, -0.1041]],

                                                        [[ 0.1059,  0.7214, -1.3534],
                                                         [-0.8103, -0.5325, -1.0905]]]))
            let fwdz = dsharp.solve(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([[[ 0.2569, -0.2489, -0.4655],
                                              [-0.8302,  0.1409, -0.0356]],
                                     
                                             [[-1.6775, -0.4010,  0.7008],
                                              [-1.5202, -1.0813, -0.3539]],
                                     
                                             [[ 4.1227,  6.2514,  2.4765],
                                              [ 2.2774,  1.4517,  0.1893]]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[[ 0.6376, -0.1375, -0.3988],
                                              [-0.5365, -0.8071,  0.9432]],
                                     
                                             [[ 3.7969,  2.3592,  0.6039],
                                              [ 1.3990,  0.4592, -0.2703]],
                                     
                                             [[14.1105, 10.9066, -6.7154],
                                              [ 6.4925,  6.5800,  0.6176]]])

            let revx = combo.tensor([[[-1.1657,  0.6982],
                                         [ 1.3803,  1.3023]],

                                        [[-0.1657, -0.7590],
                                         [ 0.8820, -1.1975]],

                                        [[ 0.1085,  0.5656],
                                         [ 0.3442, -1.5360]]]).reverseDiff()
            let revy = combo.tensor([[[-0.8791,  0.3885,  0.5178],
                                         [-0.7265, -0.1601, -0.6888]],

                                        [[ 1.4318,  0.8871,  0.1525],
                                         [ 0.3409,  0.9411,  1.0420]],

                                        [[ 1.7354,  1.4994,  0.3758],
                                         [-2.0791, -0.0783,  0.5615]]]).reverseDiff()
            let revz = dsharp.solve(revx, revy)
            let revzCorrect = combo.tensor([[[ 0.2569, -0.2489, -0.4655],
                                              [-0.8302,  0.1409, -0.0356]],
                                     
                                             [[-1.6775, -0.4010,  0.7008],
                                              [-1.5202, -1.0813, -0.3539]],
                                     
                                             [[ 4.1227,  6.2514,  2.4765],
                                              [ 2.2774,  1.4517,  0.1893]]])
            revz.reverse(combo.tensor([[[-0.4159, -0.4639, -0.4976],
                                         [ 1.3844, -0.1542,  0.6644]],

                                        [[ 2.3279, -0.0428,  0.9373],
                                         [-1.0097, -0.2422, -0.0756]],

                                        [[ 2.1178, -0.4263,  1.4700],
                                         [ 0.9668,  0.4886, -0.2448]]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[  0.0790,   0.8206],
                                               [ -0.1074,   0.4774]],
                                      
                                              [[ -2.6920,  -3.4237],
                                               [  3.1574,   3.6927]],
                                      
                                              [[-47.3892, -21.7832],
                                               [-13.2611,  -6.1560]]])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[[ 9.8821e-01,  1.5766e-01,  6.3063e-01],
                                               [ 5.3327e-01, -2.0293e-01,  1.7207e-01]],
                                      
                                              [[-2.1860e+00,  3.0531e-01, -1.2165e+00],
                                               [ 2.2286e+00,  8.7922e-03,  8.3411e-01]],
                                      
                                              [[ 9.9236e+00, -1.3470e+00,  6.0155e+00],
                                               [ 3.0247e+00, -8.1408e-01,  2.3744e+00]]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))