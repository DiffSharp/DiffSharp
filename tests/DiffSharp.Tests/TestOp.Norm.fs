// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp


[<TestFixture>]
type TestTensorNorm () =
    [<Test>]
    member _.TestTensornorm () =
        for combo in Combos.FloatingPointExcept16s do
            let t = combo.tensor([[ -0.7868,  -2.5744,   6.1267, 0.],
                                    [ -6.3106,  -9.7653,   7.7777,  16.2869],
                                    [-10.8601,  15.1932,  0.,   1.7327]])

            let n0 = t.norm(0.)
            let n0Dim0 = t.norm(0., dim=0)
            let n0Dim1 = t.norm(0., dim=1)
            let n0Dim0KeepDim = t.norm(0., dim=0, keepDim=true)
            let n0Dim1KeepDim = t.norm(0., dim=1, keepDim=true)

            let n0Correct = combo.tensor(10.)
            let n0Dim0Correct = combo.tensor([3., 3., 2., 2.])
            let n0Dim1Correct = combo.tensor([3., 4., 3.])
            let n0Dim0KeepDimCorrect = combo.tensor([[3., 3., 2., 2.]])
            let n0Dim1KeepDimCorrect = combo.tensor([[3.], [4.], [3.]])

            Assert.True(n0Correct.allclose(n0, 0.01))
            Assert.True(n0Dim0Correct.allclose(n0Dim0, 0.01))
            Assert.True(n0Dim1Correct.allclose(n0Dim1, 0.01))
            Assert.True(n0Dim0KeepDimCorrect.allclose(n0Dim0KeepDim, 0.01))
            Assert.True(n0Dim1KeepDimCorrect.allclose(n0Dim1KeepDim, 0.01))

            let n1 = t.norm(1.)
            let n1Dim0 = t.norm(1., dim=0)
            let n1Dim1 = t.norm(1., dim=1)
            let n1Dim0KeepDim = t.norm(1., dim=0, keepDim=true)
            let n1Dim1KeepDim = t.norm(1., dim=1, keepDim=true)

            let n1Correct = combo.tensor(77.4144)
            let n1Dim0Correct = combo.tensor([17.9575, 27.5329, 13.9044, 18.0196])
            let n1Dim1Correct = combo.tensor([ 9.4879, 40.1405, 27.7860])
            let n1Dim0KeepDimCorrect = combo.tensor([[17.9575, 27.5329, 13.9044, 18.0196]])
            let n1Dim1KeepDimCorrect = combo.tensor([[ 9.4879], [40.1405], [27.7860]])

            Assert.True(n1Correct.allclose(n1, 0.01))
            Assert.True(n1Dim0Correct.allclose(n1Dim0, 0.01))
            Assert.True(n1Dim1Correct.allclose(n1Dim1, 0.01))
            Assert.True(n1Dim0KeepDimCorrect.allclose(n1Dim0KeepDim, 0.01))
            Assert.True(n1Dim1KeepDimCorrect.allclose(n1Dim1KeepDim, 0.01))

            let n2 = t.norm(2.)
            let n2Dim0 = t.norm(2., dim=0)
            let n2Dim1 = t.norm(2., dim=1)
            let n2Dim0KeepDim = t.norm(2., dim=0, keepDim=true)
            let n2Dim1KeepDim = t.norm(2., dim=1, keepDim=true)

            let n2Correct = combo.tensor(29.2831)
            let n2Dim0Correct = combo.tensor([12.5851, 18.2434,  9.9010, 16.3788])
            let n2Dim1Correct = combo.tensor([ 6.6920, 21.4695, 18.7557])
            let n2Dim0KeepDimCorrect = combo.tensor([[12.5851, 18.2434,  9.9010, 16.3788]])
            let n2Dim1KeepDimCorrect = combo.tensor([[ 6.6920], [21.4695], [18.7557]])

            Assert.True(n2Correct.allclose(n2, 0.01))
            Assert.True(n2Dim0Correct.allclose(n2Dim0, 0.01))
            Assert.True(n2Dim1Correct.allclose(n2Dim1, 0.01))
            Assert.True(n2Dim0KeepDimCorrect.allclose(n2Dim0KeepDim, 0.01))
            Assert.True(n2Dim1KeepDimCorrect.allclose(n2Dim1KeepDim, 0.01))

            let nInf = t.norm(System.Double.PositiveInfinity)
            let nInfDim0 = t.norm(System.Double.PositiveInfinity, dim=0)
            let nInfDim1 = t.norm(System.Double.PositiveInfinity, dim=1)
            let nInfDim0KeepDim = t.norm(System.Double.PositiveInfinity, dim=0, keepDim=true)
            let nInfDim1KeepDim = t.norm(System.Double.PositiveInfinity, dim=1, keepDim=true)

            let nInfCorrect = combo.tensor(16.2869)
            let nInfDim0Correct = combo.tensor([10.8601, 15.1932,  7.7777, 16.2869])
            let nInfDim1Correct = combo.tensor([ 6.1267, 16.2869, 15.1932])
            let nInfDim0KeepDimCorrect = combo.tensor([[10.8601, 15.1932,  7.7777, 16.2869]])
            let nInfDim1KeepDimCorrect = combo.tensor([[ 6.1267], [16.2869], [15.1932]])

            Assert.True(nInfCorrect.allclose(nInf, 0.01))
            Assert.True(nInfDim0Correct.allclose(nInfDim0, 0.01))
            Assert.True(nInfDim1Correct.allclose(nInfDim1, 0.01))
            Assert.True(nInfDim0KeepDimCorrect.allclose(nInfDim0KeepDim, 0.01))
            Assert.True(nInfDim1KeepDimCorrect.allclose(nInfDim1KeepDim, 0.01))

            let nNegInf = t.norm(System.Double.NegativeInfinity)
            let nNegInfDim0 = t.norm(System.Double.NegativeInfinity, dim=0)
            let nNegInfDim1 = t.norm(System.Double.NegativeInfinity, dim=1)
            let nNegInfDim0KeepDim = t.norm(System.Double.NegativeInfinity, dim=0, keepDim=true)
            let nNegInfDim1KeepDim = t.norm(System.Double.NegativeInfinity, dim=1, keepDim=true)

            let nNegInfCorrect = combo.tensor(0.)
            let nNegInfDim0Correct = combo.tensor([0.7868, 2.5744, 0.0000, 0.0000])
            let nNegInfDim1Correct = combo.tensor([0.0000, 6.3106, 0.0000])
            let nNegInfDim0KeepDimCorrect = combo.tensor([[0.7868, 2.5744, 0.0000, 0.0000]])
            let nNegInfDim1KeepDimCorrect = combo.tensor([[0.0000], [6.3106], [0.0000]])

            Assert.True(nNegInfCorrect.allclose(nNegInf, 0.01))
            Assert.True(nNegInfDim0Correct.allclose(nNegInfDim0, 0.01))
            Assert.True(nNegInfDim1Correct.allclose(nNegInfDim1, 0.01))
            Assert.True(nNegInfDim0KeepDimCorrect.allclose(nNegInfDim0KeepDim, 0.01))
            Assert.True(nNegInfDim1KeepDimCorrect.allclose(nNegInfDim1KeepDim, 0.01))

            let nOther = t.norm(3.5)
            let nOtherDim0 = t.norm(3.5, dim=0)
            let nOtherDim1 = t.norm(3.5, dim=1)
            let nOtherDim0KeepDim = t.norm(3.5, dim=0, keepDim=true)
            let nOtherDim1KeepDim = t.norm(3.5, dim=1, keepDim=true)

            let nOtherCorrect = combo.tensor(20.7627)
            let nOtherDim0Correct = combo.tensor([11.3016, 16.0621,  8.6211, 16.2887])
            let nOtherDim1Correct = combo.tensor([ 6.2108, 17.4708, 16.4092])
            let nOtherDim0KeepDimCorrect = combo.tensor([[11.3016, 16.0621,  8.6211, 16.2887]])
            let nOtherDim1KeepDimCorrect = combo.tensor([[ 6.2108], [17.4708], [16.4092]])

            Assert.True(nOtherCorrect.allclose(nOther, 0.01))
            Assert.True(nOtherDim0Correct.allclose(nOtherDim0, 0.01))
            Assert.True(nOtherDim1Correct.allclose(nOtherDim1, 0.01))
            Assert.True(nOtherDim0KeepDimCorrect.allclose(nOtherDim0KeepDim, 0.01))
            Assert.True(nOtherDim1KeepDimCorrect.allclose(nOtherDim1KeepDim, 0.01))