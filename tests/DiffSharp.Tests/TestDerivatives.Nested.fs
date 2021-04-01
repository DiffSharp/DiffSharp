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
        // Perturbation confusion example
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
        // Nested optimization of a charged particle's trajectory
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

    [<Test>]
    member _.TestDerivativesNestedHessian () =
        // 2nd order (rev-on-fwd and rev-on-rev)
        // Compares Hessian-vector product to vector-Hessian product
        
        let rosenbrock (x:Tensor) = 
            let x, y = x.[0], x.[1]
            (1. - x)**2 + 100. * (y - x**2)**2

        // Analytical Hessian for Rosenbrock
        let rosenbrockHessian (x:Tensor) = 
            let x, y = x.[0], x.[1]
            dsharp.tensor([[2.+1200.*x*x-400.*y, -400.*x],[-400.*x, 200.*dsharp.one()]])

        // Jacobian-vector product (fwd)
        let jacobianv f x v =
            let _, d = dsharp.evalForwardDiff f x v
            d

        // Vector-jacobian product (rev)
        let vjacobian f x v =
            let _, r = dsharp.evalReverseDiff f x
            r v

        // Hessian-vector product (rev-on-fwd)
        let hessianv f x v =
            let gv xx = jacobianv f xx v
            let hv = vjacobian gv x (dsharp.tensor(1.))
            hv

        // Vector-Hessian product (rev-on-rev)
        let vhessian f x v =
            let vg xx = vjacobian f xx (dsharp.tensor(1.))
            let vh = vjacobian vg x v
            vh

        let x = dsharp.randn(2)
        let v = dsharp.randn(2)

        // Should be the same because Hessian is symmetric
        let hv = hessianv rosenbrock x v
        let vh = vhessian rosenbrock x v
        let hvCorrect = rosenbrockHessian(x).matmul(v)

        Assert.True(hvCorrect.allclose(hv, 0.01))
        Assert.True(hvCorrect.allclose(vh, 0.01))
