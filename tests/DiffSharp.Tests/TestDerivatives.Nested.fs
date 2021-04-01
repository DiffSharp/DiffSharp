// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Util

// #nowarn "0058"

[<TestFixture>]
type TestDerivativesNested () =

    [<Test>]
    member _.TestDerivativesNestedPerturbationConfusion () =
        // 2nd order (fwd-on-fwd)
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        // Page 4
        let x0 = dsharp.tensor(1)
        let y0 = dsharp.tensor(2)
        let d = dsharp.diff (fun x -> x * dsharp.diff (fun y -> x * y) y0) x0
        let dCorrect = dsharp.tensor(2)
        Assert.CheckEqual(dCorrect, d)

    [<Test>]
    member _.TestDerivativesNestedChargedParticle () =
        // 3rd order (fwd-on-fwd-on-rev)
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        // Page 13
        let dt = dsharp.tensor(0.1)
        let x0 = dsharp.tensor([0., 8.])
        let x'0 = dsharp.tensor([0.75, 0.])
        let errorTolerance = 0.1

        let norm (x:Tensor) = x.pow(2).sum().sqrt()

        let naiveEuler (w:Tensor) = 
            let p x = (1 / norm (x - dsharp.tensor([dsharp.tensor(10.) + w*0; 10.-w])))
                    + (1 / norm (x - dsharp.tensor([10, 0])))

            let mutable x = x0
            let mutable x' = x'0
            while float x.[1] > 0. do
                let x'' = -dsharp.grad p x
                x <- x + dt * x'
                x' <- x' + dt * x''

            let dtf = (0-x.[1]) /x'.[1]
            let xtf = x+dtf*x'
            xtf.[0]*xtf.[0]

        let argminNewton f x =
            let mutable x = x
            let mutable i = 0
            let mutable converged = false
            while not converged do
                let dfdx = dsharp.diff f x
                x <- x - dfdx / (dsharp.diff (dsharp.diff f) x)
                i <- i + 1
                if float (dfdx.abs()) < errorTolerance then converged <- true
            i, x

        let w0 = dsharp.tensor(0.)
        let i, wf = argminNewton naiveEuler w0
        let iCorrect, wfCorrect = 4, dsharp.tensor(-0.266521)
        
        Assert.AreEqual(iCorrect, i)
        Assert.True(wfCorrect.allclose(wf, 0.01))
