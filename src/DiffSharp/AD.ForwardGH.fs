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
[<CustomEquality; CustomComparison>]
type DualGH =
    // Primal, vector of gradient components, matrix of Hessian components
    | DualGH of float * Vector * Matrix
    override d.ToString() = let (DualGH(p, g, h)) = d in sprintf "DualGH (%f, %A, %A)" p g h
    static member op_Explicit(p) = DualGH(p, ZeroVector(), ZeroMatrix())
    static member op_Explicit(DualGH(p, _, _)) = p
    static member DivideByInt(DualGH(p, g, m), i:int) = DualGH(p / float i, g / float i, m / float i)
    static member Zero = DualGH(0., ZeroVector(), ZeroMatrix())
    static member One = DualGH(1., ZeroVector(), ZeroMatrix())
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? DualGH as d2 -> let DualGH(a, _, _), DualGH(b, _, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this DualGH with another type of object."
    override d.Equals(other) = 
        match other with
        | :? DualGH as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (DualGH(a, b, c)) = d in hash [|a; b; c|]
    // DualGH - DualGH binary operations
    static member (+) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a + b, ag + bg, Matrix.SymmetricOp(ah, bh, fun i j -> ah.[i, j] + bh.[i, j]))
    static member (-) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a - b, ag - bg, Matrix.SymmetricOp(ah, bh, fun i j -> ah.[i, j] - bh.[i, j]))
    static member (*) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a * b, ag * b + a * bg, Matrix.SymmetricOp(ah, bh, fun i j -> ag.[j] * bg.[i] + a * bh.[i, j] + bg.[j] * ag.[i] + b * ah.[i, j]))
    static member (/) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let bsq, atimes2, oneoverbcube = b * b, a * 2., 1. / (b * b * b) in DualGH(a / b, (ag * b - a * bg) / bsq, Matrix.SymmetricOp(ah, bh, fun i j -> (atimes2 * bg.[j] * bg.[i] + bsq * ah.[i, j] - b * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + a * bh.[i, j])) * oneoverbcube))
    static member Pow (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let apowb, loga, apowbminus2, bsq = a ** b, log a, a ** (b - 2.), b * b in DualGH(apowb, apowb * ((b * ag / a) + (loga * bg)), Matrix.SymmetricOp(ah, bh, fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (ag.[j] * (-ag.[i] + loga * a * bg.[i]) + a * (loga * bg.[j] * ag.[i] + ah.[i, j])) + a * (ag.[j] * bg.[i] + bg.[j] * (ag.[i] + loga * loga * a * bg.[i]) + loga * a * bh.[i, j]))))
    // DualGH - float binary operations
    static member (+) (DualGH(a, ag, ah), b) = DualGH(a + b, ag, ah)
    static member (-) (DualGH(a, ag, ah), b) = DualGH(a - b, ag, ah)
    static member (*) (DualGH(a, ag, ah), b) = DualGH(a * b, ag * b, Matrix.SymmetricOp(ah, fun i j -> ah.[i, j] * b))
    static member (/) (DualGH(a, ag, ah), b) = DualGH(a / b, ag / b, Matrix.SymmetricOp(ah, fun i j -> ah.[i, j] / b))
    static member Pow (DualGH(a, ag, ah), b) = let apowb, bsq, apowbminus2 = a ** b, b * b, a ** (b - 2.) in DualGH(apowb, apowb * (b * ag / a), Matrix.SymmetricOp(ah, fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (a * ah.[i, j] - ag.[j] * ag.[i]))))
    // float - DualGH binary operations
    static member (+) (a, DualGH(b, bg, bh)) = DualGH(a + b, bg, bh)
    static member (-) (a, DualGH(b, bg, bh)) = DualGH(a - b, -bg, -bh)
    static member (*) (a, DualGH(b, bg, bh)) = DualGH(a * b, a * bg, Matrix.SymmetricOp(bh, fun i j -> a * bh.[i, j]))
    static member (/) (a, DualGH(b, bg, bh)) = let aoverbcube = a / (b * b * b) in DualGH(a / b, -aoverbcube * b * bg, Matrix.SymmetricOp(bh, fun i j -> (2. * bg.[j] * bg.[i] - b * bh.[i, j]) * aoverbcube))
    static member Pow (a, DualGH(b, bg, bh)) = let apowb, loga, term = a ** b, log a, (a ** (b - 2.)) * a * log a in DualGH(apowb, apowb * loga * bg, Matrix.SymmetricOp(bh, fun i j -> term * (bg.[j] * loga * a * bg.[i] + a * bh.[i, j])))
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
    static member Log (DualGH(a, ag, ah)) = let asq = a * a in DualGH(log a, ag / a, Matrix.SymmetricOp(ah, fun i j -> -ag.[i] * ag.[j] / asq + ah.[i, j] / a))
    static member Exp (DualGH(a, ag, ah)) = let expa = exp a in DualGH(expa, expa * ag, Matrix.SymmetricOp(ah, fun i j -> expa * ag.[i] * ag.[j] + expa * ah.[i, j]))
    static member Sin (DualGH(a, ag, ah)) = let sina, cosa = sin a, cos a in DualGH(sina, cosa * ag, Matrix.SymmetricOp(ah, fun i j -> -sina * ag.[i] * ag.[j] + cosa * ah.[i, j]))
    static member Cos (DualGH(a, ag, ah)) = let sina, cosa = sin a, cos a in DualGH(cosa, -sina * ag, Matrix.SymmetricOp(ah, fun i j -> -cosa * ag.[i] * ag.[j] - sina * ah.[i, j]))
    static member Tan (DualGH(a, ag, ah)) = let tana, secsqa = tan a, 1. / ((cos a) * (cos a)) in DualGH(tana, secsqa * ag, Matrix.SymmetricOp(ah, fun i j -> 2. * secsqa * tana * ag.[i] * ag.[j] + secsqa * ah.[i, j]))
    static member (~-) (DualGH(a, ag, ah)) = DualGH(-a, -ag, -ah)
    static member Sqrt (DualGH(a, ag, ah)) = let term = 1. / (2. * sqrt a) in DualGH(sqrt a, term * ag, Matrix.SymmetricOp(ah, fun i j -> (term / (-2. * a)) * ag.[i] * ag.[j] + term * ah.[i,j]))
    static member Sinh (DualGH(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in DualGH(sinha, cosha * ag, Matrix.SymmetricOp(ah, fun i j -> sinha * ag.[i] * ag.[j] + cosha * ah.[i, j]))
    static member Cosh (DualGH(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in DualGH(cosha, sinha * ag, Matrix.SymmetricOp(ah, fun i j -> cosha * ag.[i] * ag.[j] + sinha * ah.[i, j]))
    static member Tanh (DualGH(a, ag, ah)) = let tanha, sechsqa = tanh a, 1. / ((cosh a) * (cosh a)) in DualGH(tanha, sechsqa * ag, Matrix.SymmetricOp(ah, fun i j -> -2. * sechsqa * tanha * ag.[i] * ag.[j] + sechsqa * ah.[i, j]))
    static member Asin (DualGH(a, ag, ah)) = let term, term2 = 1. / sqrt (1. - a * a), (a / (1. - a * a)) in DualGH(asin a, term * ag, Matrix.SymmetricOp(ah, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Acos (DualGH(a, ag, ah)) = let term, term2 = -1. / sqrt (1. - a * a), (a / (1. - a * a)) in DualGH(acos a, term * ag, Matrix.SymmetricOp(ah, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Atan (DualGH(a, ag, ah)) = let term, term2 = 1. / (1. + a * a), (-2. * a / (1. + a * a)) in DualGH(atan a, term * ag, Matrix.SymmetricOp(ah, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))


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
    let inline tuple (DualGH(p, g, _)) = (p, g.FirstItem)
    /// Get the primal and the gradient array of a DualGH, as a tuple
    let inline tupleG (DualGH(p, g, _)) = (p, g.V)
    /// Get the primal and Hessian 2d array of a DualGH, as a tuple
    let inline tupleH (DualGH(p, _, h)) = (p, h.M)
    /// Get the primal, the gradient array, and the Hessian 2d array of a DualGH, as a tuple
    let inline tupleGH (DualGH(p, g, h)) = (p, g.V, h.M)


/// ForwardGH differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardGHOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        dualGHAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        diff' f x |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        dualGHActArray x |> f |> tupleG

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        let a = dualGHActArray x |> f
        (Array.map primal a, Matrix.Create(a.Length, fun i -> gradient a.[i]).M)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let (v, j) = jacobian' f x in (v, transpose j)
    
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x =
        dualGHActArray x |> f |> tupleH

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        hessian' f x |> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x =
        dualGHActArray x |> f |> tupleGH
    
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x =
        gradhessian' f x |> sndtrd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x =
        let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x = ForwardGHOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x = ForwardGHOps.diff f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x = ForwardGHOps.grad' f (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x = ForwardGHOps.grad f (array x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x = ForwardGHOps.laplacian' f (array x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x = ForwardGHOps.laplacian f (array x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x = ForwardGHOps.jacobianT' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x = ForwardGHOps.jacobianT f (array x) |> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x = ForwardGHOps.jacobian' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x = ForwardGHOps.jacobian f (array x) |> matrix
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x = ForwardGHOps.hessian' f (array x) |> fun (a, b) -> (a, matrix b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x = ForwardGHOps.hessian f (array x) |> matrix
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x = ForwardGHOps.gradhessian' f (array x) |> fun (a, b, c) -> (a, vector b, matrix c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x = ForwardGHOps.gradhessian f (array x) |> fun (a, b) -> (vector a, matrix b)
