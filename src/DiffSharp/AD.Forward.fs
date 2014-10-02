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

#light

/// Forward AD module
module DiffSharp.AD.Forward

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// Dual numeric type, keeping primal and tangent values
// UNOPTIMIZED
type Dual =
    | Dual of float * float
    override d.ToString() = let (Dual(p, t)) = d in sprintf "Dual(%f, %f)" p t
    static member op_Explicit(p) = Dual(p, 0.)
    static member op_Explicit(Dual(p, _)) = p
    static member DivideByInt(Dual(p, t), i:int) = Dual(p / float i, t / float i)
    static member Zero = Dual(0., 0.)
    static member One = Dual(1., 0.)
    static member (+) (Dual(a, at), Dual(b, bt)) = Dual(a + b, at + bt)
    static member (-) (Dual(a, at), Dual(b, bt)) = Dual(a - b, at - bt)
    static member (*) (Dual(a, at), Dual(b, bt)) = Dual(a * b, at * b + a * bt)
    static member (/) (Dual(a, at), Dual(b, bt)) = Dual(a / b, (at * b - a * bt) / (b * b))
    static member Pow (Dual(a, at), Dual(b, bt)) = Dual(a ** b, (a ** b) * ((b * at / a) + ((log a) * bt)))
    static member (+) (Dual(a, at), b) = Dual(a + b, at)
    static member (-) (Dual(a, at), b) = Dual(a - b, at)
    static member (*) (Dual(a, at), b) = Dual(a * b, at * b)
    static member (/) (Dual(a, at), b) = Dual(a / b, at / b)
    static member Pow (Dual(a, at), b) = Dual(a ** b, b * (a ** (b - 1.)) * at)
    static member (+) (a, Dual(b, bt)) = Dual(b + a, bt)
    static member (-) (a, Dual(b, bt)) = Dual(a - b, -bt)
    static member (*) (a, Dual(b, bt)) = Dual(b * a, bt * a)
    static member (/) (a, Dual(b, bt)) = Dual(a / b, -a * bt / (b * b))
    static member Pow (a, Dual(b, bt)) = Dual(a ** b, (a ** b) * (log a) * bt)
    static member Log (Dual(a, at)) = Dual(log a, at / a)
    static member Exp (Dual(a, at)) = Dual(exp a, at * exp a)
    static member Sin (Dual(a, at)) = Dual(sin a, at * cos a)
    static member Cos (Dual(a, at)) = Dual(cos a, -at * sin a)
    static member Tan (Dual(a, at)) = Dual(tan a, at / ((cos a) * (cos a)))
    static member (~-) (Dual(a, at)) = Dual(-a, -at)
    static member Sqrt (Dual(a, at)) = Dual(sqrt a, at / (2. * sqrt a))
    static member Sinh (Dual(a, at)) = Dual(sinh a, at * cosh a)
    static member Cosh (Dual(a, at)) = Dual(cosh a, at * sinh a)
    static member Tanh (Dual(a, at)) = Dual(tanh a, at / ((cosh a) * (cosh a)))
    static member Asin (Dual(a, at)) = Dual(asin a, at / sqrt (1. - a * a))
    static member Acos (Dual(a, at)) = Dual(acos a, -at / sqrt (1. - a * a))
    static member Atan (Dual(a, at)) = Dual(atan a, at / (1. + a * a))


/// Dual operations module (automatically opened)
[<AutoOpen>]
module DualOps =
    /// Make Dual, with primal value `p` and tangent 0
    let inline dual p = Dual(p, 0.)
    /// Make Dual, with primal value `p` and tangent value `t`
    let inline dualSet (p, t) = Dual(p, t)
    /// Make active Dual (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualAct p = Dual(p, 1.)
    /// Make an array of arrays of Dual, with primal values given in array `x`. The tangent values along the diagonal are 1 and the rest are 0.
    let inline dualActArrayArray (x:float[]) = Array.init x.Length (fun i -> (Array.init x.Length (fun j -> if i = j then dualAct x.[j] else dual x.[j])))
    /// Get the primal value of a Dual
    let inline primal (Dual(p, _)) = p
    /// Get the tangent value of a Dual
    let inline tangent (Dual(_, t)) = t
    /// Get the primal and tangent values of a Dual, as a tuple
    let inline tuple (Dual(p, t)) = (p, t)


/// Forward differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        dualAct >> f >> tuple

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        dualAct >> f >> tangent
       
    /// Original value and directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir' r f =
        fun x -> Array.zip x r |> Array.map dualSet |> f |> tuple

    /// Directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir r f =
        diffdir' r f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        fun x ->
            let a = Array.map f (dualActArrayArray x)
            (primal a.[0], Array.map tangent a)

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f =
        fun x ->
            let a = Array.map f (dualActArrayArray x)
            (Array.map primal a.[0], Array2D.map tangent (array2D a))

    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f =
        jacobianT' f >> snd

    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        jacobianT' f >> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f =
        jacobian' f >> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f = ForwardOps.diff' f
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f = ForwardOps.diff f
    /// Original value and directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir' r f = array >> ForwardOps.diffdir' (array r) f
    /// Directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir r f = array >> ForwardOps.diffdir (array r) f
    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f = array >> ForwardOps.grad' f >> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`
    let inline grad f = array >> ForwardOps.grad f >> vector
    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f = array >> ForwardOps.jacobianT' f >> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f = array >> ForwardOps.jacobianT f >> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f = array >> ForwardOps.jacobian' f >> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f = array >> ForwardOps.jacobian f >> matrix


/// Numeric literal for a Dual with tangent 0
module NumericLiteralQ = // (Allowed literals : Q, R, Z, I, N, G)
    let FromZero () = dual 0.
    let FromOne () = dual 1.
    let FromInt32 p = dual (float p)


/// Numeric literal for a Dual with tangent 1 (i.e. the variable of differentiation)
module NumericLiteralR =    
    let FromZero () = dualAct 0.
    let FromOne () = dualAct 1.
    let FromInt32 p = dualAct (float p)
