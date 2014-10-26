//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Reference for "triplets": Dixon L., 2001, Automatic Differentiation: Calculation of the Hessian (http://dx.doi.org/10.1007/0-306-48332-7_17)
//

#light

/// Forward AD module, keeping vectors of gradient components and matrices of Hessian components
module DiffSharp.AD.ForwardGH

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// DualGH numeric type, keeping a triplet of primal value, a vector of gradient components, and a matrix of Hessian components
// NOT FULLY OPTIMIZED
type DualGH =
    | DualGH of float * Vector * Matrix
    override d.ToString() = let (DualGH(p, g, h)) = d in sprintf "DualGH (%f, %A, %A)" p g h
    static member op_Explicit(p) = DualGH(p, Vector.Zero, Matrix.Zero)
    static member op_Explicit(DualGH(p, _, _)) = p
    static member DivideByInt(DualGH(p, g, m), i:int) = DualGH(p / float i, g / float i, m / float i)
    static member Zero = DualGH(0., Vector.Zero, Matrix.Zero)
    static member One = DualGH(1., Vector.Zero, Matrix.Zero)
    // DualGH - DualGH binary operations
    static member (+) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a + b, ag + bg, Matrix.CreateSymmetric(ah.Rows, fun i j -> ah.[i, j] + bh.[i, j]))
    static member (-) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a - b, ag - bg, Matrix.CreateSymmetric(ah.Rows, fun i j -> ah.[i, j] - bh.[i, j]))
    static member (*) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a * b, ag * b + a * bg, Matrix.CreateSymmetric(ah.Rows, fun i j -> ag.[j] * bg.[i] + a * bh.[i, j] + bg.[j] * ag.[i] + b * ah.[i, j]))
    static member (/) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let bsq, atimes2, oneoverbcube = b * b, a * 2., 1. / (b * b * b) in DualGH(a / b, (ag * b - a * bg) / bsq, Matrix.CreateSymmetric(ah.Rows, fun i j -> (atimes2 * bg.[j] * bg.[i] + bsq * ah.[i, j] - b * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + a * bh.[i, j])) * oneoverbcube))
    static member Pow (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let apowb, loga, apowbminus2, bsq = a ** b, log a, a ** (b - 2.), b * b in DualGH(apowb, apowb * ((b * ag / a) + (loga * bg)), Matrix.CreateSymmetric(ah.Rows, fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (ag.[j] * (-ag.[i] + loga * a * bg.[i]) + a * (loga * bg.[j] * ag.[i] + ah.[i, j])) + a * (ag.[j] * bg.[i] + bg.[j] * (ag.[i] + loga * loga * a * bg.[i]) + loga * a * bh.[i, j]))))
    // DualGH - float binary operations
    static member (+) (DualGH(a, ag, ah), b) = DualGH(a + b, ag, ah)
    static member (-) (DualGH(a, ag, ah), b) = DualGH(a - b, ag, ah)
    static member (*) (DualGH(a, ag, ah), b) = DualGH(a * b, ag * b, Matrix.CreateSymmetric(ah.Rows, fun i j -> ah.[i, j] * b))
    static member (/) (DualGH(a, ag, ah), b) = DualGH(a / b, ag / b, Matrix.CreateSymmetric(ah.Rows, fun i j -> ah.[i, j] / b))
    static member Pow (DualGH(a, ag, ah), b) = let apowb, bsq, apowbminus2 = a ** b, b * b, a ** (b - 2.) in DualGH(apowb, apowb * (b * ag / a), Matrix.CreateSymmetric(ah.Rows, fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (a * ah.[i, j] - ag.[j] * ag.[i]))))
    // float - DualGH binary operations
    static member (+) (a, DualGH(b, bg, bh)) = DualGH(a + b, bg, bh)
    static member (-) (a, DualGH(b, bg, bh)) = DualGH(a - b, -bg, -bh)
    static member (*) (a, DualGH(b, bg, bh)) = DualGH(a * b, a * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> a * bh.[i, j]))
    static member (/) (a, DualGH(b, bg, bh)) = let aoverbcube = a / (b * b * b) in DualGH(a / b, -aoverbcube * b * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> (2. * bg.[j] * bg.[i] - b * bh.[i, j]) * aoverbcube))
    static member Pow (a, DualGH(b, bg, bh)) = let apowb, loga, term = a ** b, log a, (a ** (b - 2.)) * a * log a in DualGH(apowb, apowb * loga * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> term * (bg.[j] * loga * a * bg.[i] + a * bh.[i, j])))
    // DualGH - int binary operations
    static member (+) (a:DualGH, b:int) = a + float b
    static member (-) (a:DualGH, b:int) = a - float b
    static member (*) (a:DualGH, b:int) = a * float b
    static member (/) (a:DualGH, b:int) = a / float b
    static member Pow (a:DualGH, b:int) = DualGH.Pow(a, float b)
    // int - DualGH binary operations
    static member (+) (a:int, b:DualGH) = (float a) + b
    static member (-) (a:int, b:DualGH) = (float a) - b
    static member (*) (a:int, b:DualGH) = (float a) * b
    static member (/) (a:int, b:DualGH) = (float a) / b
    static member Pow (a:int, b:DualGH) = DualGH.Pow(float a, b)
    // DualGH unary operations
    static member Log (DualGH(a, ag, ah)) = let asq = a * a in DualGH(log a, ag / a, Matrix.CreateSymmetric(ah.Rows, fun i j -> -ag.[i] * ag.[j] / asq + ah.[i, j] / a))
    static member Exp (DualGH(a, ag, ah)) = let expa = exp a in DualGH(expa, expa * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> expa * ag.[i] * ag.[j] + expa * ah.[i, j]))
    static member Sin (DualGH(a, ag, ah)) = let sina, cosa = sin a, cos a in DualGH(sina, cosa * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> -sina * ag.[i] * ag.[j] + cosa * ah.[i, j]))
    static member Cos (DualGH(a, ag, ah)) = let sina, cosa = sin a, cos a in DualGH(cosa, -sina * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> -cosa * ag.[i] * ag.[j] - sina * ah.[i, j]))
    static member Tan (DualGH(a, ag, ah)) = let tana, secsqa = tan a, 1. / ((cos a) * (cos a)) in DualGH(tana, secsqa * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> 2. * secsqa * tana * ag.[i] * ag.[j] + secsqa * ah.[i, j]))
    static member (~-) (DualGH(a, ag, ah)) = DualGH(-a, -ag, -ah)
    static member Sqrt (DualGH(a, ag, ah)) = let term = 1. / (2. * sqrt a) in DualGH(sqrt a, term * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> (term / (-2. * a)) * ag.[i] * ag.[j] + term * ah.[i,j]))
    static member Sinh (DualGH(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in DualGH(sinha, cosha * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> sinha * ag.[i] * ag.[j] + cosha * ah.[i, j]))
    static member Cosh (DualGH(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in DualGH(cosha, sinha * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> cosha * ag.[i] * ag.[j] + sinha * ah.[i, j]))
    static member Tanh (DualGH(a, ag, ah)) = let tanha, sechsqa = tanh a, 1. / ((cosh a) * (cosh a)) in DualGH(tanha, sechsqa * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> -2. * sechsqa * tanha * ag.[i] * ag.[j] + sechsqa * ah.[i, j]))
    static member Asin (DualGH(a, ag, ah)) = let term, term2 = 1. / sqrt (1. - a * a), (a / (1. - a * a)) in DualGH(asin a, term * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Acos (DualGH(a, ag, ah)) = let term, term2 = -1. / sqrt (1. - a * a), (a / (1. - a * a)) in DualGH(acos a, term * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Atan (DualGH(a, ag, ah)) = let term, term2 = 1. / (1. + a * a), (-2. * a / (1. + a * a)) in DualGH(atan a, term * ag, Matrix.CreateSymmetric(ah.Rows, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))


/// DualGH operations module (automatically opened)
[<AutoOpen>]
module DualGHOps =
    /// Make DualGH, with primal value `p`, gradient dimension `m`, and all gradient and Hessian components 0
    let inline dualGH p m = DualGH(p, Vector.Create(m, 0.), Matrix.Create(m, m, 0.))
    /// Make DualGH, with primal value `p`, gradient array `g`, and Hessian 2d array `h`
    let inline dualGHSet (p, g, h:float[,]) = DualGH(p, Vector.Create(g), Matrix.Create(h))
    /// Make active DualGH (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the gradient component with index `i` having value 1, the rest of the gradient components 0, and Hessian components 0
    let inline dualGHAct p m i = DualGH(p, Vector.Create(m, i, 1.), Matrix.Create(m, m, 0.))
    /// Make an array of active DualGH, with primal values given in array `x`. For a DualGH with index _i_, the gradient is the unit vector with 1 in the _i_th place, and the Hessian components are 0.
    let inline dualGHActArray (x:float[]) = Array.init x.Length (fun i -> dualGHAct x.[i] x.Length i)
    /// Get the primal value of a DualGH
    let inline primal (DualGH(p, _, _)) = p
    /// Get the gradient array of a DualGH
    let inline gradient (DualGH(_, g, _)) = g.V
    /// Get the Hessian 2d array of a DualGH
    let inline hessian (DualGH(_, _, h)) = h.M
    /// Get the primal and the first gradient component of a DualGH, as a tuple
    let inline tuple (DualGH(p, g, _)) = (p, g.V.[0])
    /// Get the primal and the gradient array of a DualGH, as a tuple
    let inline tupleG (DualGH(p, g, _)) = (p, g.V)
    /// Get the primal and Hessian 2d array of a DualGH, as a tuple
    let inline tupleH (DualGH(p, _, h)) = (p, h.M)
    /// Get the primal, the gradient array, and the Hessian 2d array of a DualGH, as a tuple
    let inline tupleGH (DualGH(p, g, h)) = (p, g.V, h.M)


/// ForwardGH differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardGHOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        fun x -> dualGHAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        diff' f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        dualGHActArray >> f >> tupleG

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        fun x ->
            let a = dualGHActArray x |> f
            (Array.map primal a, Matrix.Create(a.Length, fun i -> gradient a.[i]).M)

    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f =
        jacobian' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f =
        fun x -> let (v, j) = jacobian' f x in (v, transpose j)
    
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f =
        jacobianT' f >> snd

    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f =
        dualGHActArray >> f >> tupleH

    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f =
        hessian' f >> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f =
        dualGHActArray >> f >> tupleGH
    
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f =
        gradhessian' f >> sndtrd

    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f =
        fun x -> let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f =
        laplacian' f >> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f = ForwardGHOps.diff' f
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f = ForwardGHOps.diff f
    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f = array >> ForwardGHOps.grad' f >> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`
    let inline grad f = array >> ForwardGHOps.grad f >> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f = array >> ForwardGHOps.laplacian' f
    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f = array >> ForwardGHOps.laplacian f
    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f = array >> ForwardGHOps.jacobianT' f >> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f = array >> ForwardGHOps.jacobianT f >> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f = array >> ForwardGHOps.jacobian' f >> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f = array >> ForwardGHOps.jacobian f >> matrix
    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f = array >> ForwardGHOps.hessian' f >> fun (a, b) -> (a, matrix b)
    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f = array >> ForwardGHOps.hessian f >> matrix
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f = array >> ForwardGHOps.gradhessian' f >> fun (a, b, c) -> (a, vector b, matrix c)
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f = array >> ForwardGHOps.gradhessian f >> fun (a, b) -> (vector a, matrix b)