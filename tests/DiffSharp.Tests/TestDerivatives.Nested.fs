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

        let x0 = dsharp.tensor(1.)
        let y0 = dsharp.tensor(2.)
        let d = dsharp.diff (fun x -> x * dsharp.diff (fun y -> x * y) y0) x0
        let dCorrect = dsharp.tensor(2.)
        Assert.CheckEqual(dCorrect, d)


    [<Test>]
    member _.TestDerivativesNestedMin () =
        // 2nd order (fwd-on-fwd, because fgrad below is using fwd for scalar arguments) 
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        // Page 2

        let rosenbrock (x:Tensor) (y:Tensor) = 
            (1. - x)**2 + 100. * (y - x**2)**2

        let x0 = dsharp.tensor(0.9)

        let min f =
            let lr = 0.001
            let tolerance = 0.1
            let rec iter x =
                let fx, gx = dsharp.fgrad f x
                if float fx <= tolerance then fx
                else iter (x - lr*gx)
            iter x0

        let fmin = min (fun x -> min (fun y -> rosenbrock x y))
        let fminCorrect = dsharp.zero()

        Assert.True(fminCorrect.allclose(fmin, 0.1, 0.1))

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
            while float x[1] > 0. do
                let x'' = -dsharp.grad p x
                x <- x + dt * x'
                x' <- x' + dt * x''

            let dtf = (0-x[1]) /x'[1]
            let xtf = x+dtf*x'
            xtf[0]*xtf[0]

        let argminNewton f x =
            let rec iter i x =
                let dfdx = dsharp.diff f x
                if float (dfdx.abs()) < errorTolerance then i, x
                else iter (i+1) (x - dfdx/(dsharp.diff (dsharp.diff f) x))
            iter 1 x

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
            let x, y = x[0], x[1]
            (1. - x)**2 + 100. * (y - x**2)**2

        // Analytical Hessian for Rosenbrock
        let rosenbrockHessian (x:Tensor) = 
            let x, y = x[0], x[1]
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

    [<Test>]
    member _.TestDerivativesNestedTanh () =
        // 6th order (fwd-on-fwd-on-fwd-on-fwd-on-fwd-on-fwd)
        // Inspired by https://github.com/HIPS/autograd

        let sech (x:Tensor) = 1 / x.cosh()

        let f (x:Tensor) = x.tanh()
        let f' (x:Tensor) = let s = sech x in s*s
        let f'' (x:Tensor) = let s = sech x in -2.*x.tanh()*s*s
        let f''' (x:Tensor) = let t, s = x.tanh(), sech x in 4*t*t*s*s - 2*s*s*s*s
        let f'''' (x:Tensor) = let t, s = x.tanh(), sech x in 16*t*s*s*s*s - 8*t*t*t*s*s
        let f''''' (x:Tensor) = let t, s = x.tanh(), sech x in 8*s*s*(2*t*t*t*t + 2*s*s*s*s - 11*t*t*s*s)
        let f'''''' (x:Tensor) = let t, s = x.tanh(), sech x in -16*t*s*s*(2*t*t*t*t + 17*s*s*s*s - 26*t*t*s*s)

        let x = dsharp.randn(1).squeeze()

        let d = dsharp.diff
        let f'x = d f x
        let f''x = d (d f) x
        let f'''x = d (d (d f)) x
        let f''''x = d (d (d (d f))) x
        let f'''''x = d (d (d (d (d f)))) x
        let f''''''x = d (d (d (d (d (d f))))) x

        let f'xCorrect = f' x
        let f''xCorrect = f'' x
        let f'''xCorrect = f''' x
        let f''''xCorrect = f'''' x
        let f'''''xCorrect = f''''' x
        let f''''''xCorrect = f'''''' x

        Assert.True(f'xCorrect.allclose(f'x, 0.01))
        Assert.True(f''xCorrect.allclose(f''x, 0.01))
        Assert.True(f'''xCorrect.allclose(f'''x, 0.01))
        Assert.True(f''''xCorrect.allclose(f''''x, 0.01))
        Assert.True(f'''''xCorrect.allclose(f'''''x, 0.01))
        Assert.True(f''''''xCorrect.allclose(f''''''x, 0.01))


// Random pipelines of fwd-fwd-rev-fwd, etc.