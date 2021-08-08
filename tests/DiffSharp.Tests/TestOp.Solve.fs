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
            if combo.backend = Backend.Torch then
                let t3x3 = combo.tensor([[-1.1606,  0.6579,  1.0674],
                                         [-1.0226,  0.2406, -0.5414],
                                         [ 0.1195,  1.2423,  0.0889]])
                let t3 = combo.tensor([ 0.6791,  0.5497, -0.3624])
                let t3x3Solvet3 = t3x3.solve(t3)
                let t3x3Solvet3Correct = combo.tensor([-0.6392, -0.2364,  0.0869])

                Assert.True(t3x3Solvet3Correct.allclose(t3x3Solvet3, 0.01))
