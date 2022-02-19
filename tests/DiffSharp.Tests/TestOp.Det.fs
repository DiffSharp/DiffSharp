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
