// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp


[<TestFixture>]
type TestTensorOuter () =
    [<Test>]
    member _.TestTensorOuter () =
        for combo in Combos.FloatingPointExcept16s do
            let a1 = combo.tensor([ 1.7865,  1.2723,  0.2065, -0.4601,  0.3218])
            let b1 = combo.tensor([ 2.1136,  1.0551, -0.4575])

            let a1outerb1 = a1.outer(b1)
            let a1outerb1Correct = combo.tensor([[ 3.7759,  1.8849, -0.8173],
                                                    [ 2.6891,  1.3424, -0.5820],
                                                    [ 0.4365,  0.2179, -0.0945],
                                                    [-0.9725, -0.4854,  0.2105],
                                                    [ 0.6801,  0.3395, -0.1472]])

            Assert.True(a1outerb1Correct.allclose(a1outerb1, 0.01))

            let a2 = combo.tensor([[ 1.7865,  1.2723,  0.2065, -0.4601,  0.3218],
                                    [-0.2400, -0.1650, -1.1463,  0.0578,  1.5240]])
            let b2 = combo.tensor([[ 2.1136,  1.0551, -0.4575],
                                     [ 1.1928, -2.3803,  0.3160]])

            let a2outerb2 = a2.outer(b2)
            let a2outerb2Correct = combo.tensor([[[ 3.7759,  1.8849, -0.8173],
                                                    [ 2.6891,  1.3424, -0.5820],
                                                    [ 0.4365,  0.2179, -0.0945],
                                                    [-0.9725, -0.4854,  0.2105],
                                                    [ 0.6801,  0.3395, -0.1472]],
                                                    
                                                    [[-0.2863,  0.5713, -0.0758],
                                                        [-0.1968,  0.3927, -0.0521],
                                                        [-1.3672,  2.7284, -0.3622],
                                                        [ 0.0690, -0.1376,  0.0183],
                                                        [ 1.8177, -3.6275,  0.4816]]])

            Assert.True(a2outerb2Correct.allclose(a2outerb2, 0.01))