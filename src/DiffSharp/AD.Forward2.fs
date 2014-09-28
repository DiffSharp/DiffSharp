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

/// Forward AD module, 2nd order
module DiffSharp.AD.Forward2

open DiffSharp.Util.General

/// Dual2 numeric type, keeping primal, tangent, and tangent-of-tangent values
// UNOPTIMIZED
type Dual2 =
    | Dual2 of float * float * float
    override d.ToString() = let (Dual2(p, t, t2)) = d in sprintf "Dual2(%f, %f, %f)" p t t2
    static member op_Explicit(p) = Dual2(p, 0. , 0.)
    static member op_Explicit(Dual2(p, _, _)) = p
    static member DivideByInt(Dual2(p, t, t2), i:int) = Dual2(p / float i, t / float i, t2 / float i)
    static member Zero = Dual2(0., 0., 0.)
    static member One = Dual2(1., 0., 0.)
    static member (+) (Dual2(a, at, at2), Dual2(b, bt, bt2)) = Dual2(a + b, at + bt, at2 + bt2)
    static member (-) (Dual2(a, at, at2), Dual2(b, bt, bt2)) = Dual2(a - b, at - bt, at2 - bt2)
    static member (*) (Dual2(a, at, at2), Dual2(b, bt, bt2)) = Dual2(a * b, at * b + a * bt, 2. * at * bt + b * at2 + a * bt2)
    static member (/) (Dual2(a, at, at2), Dual2(b, bt, bt2)) = Dual2(a / b, (at * b - a * bt) / (b * b), (2. * a * bt * bt + b * b * at2 - b * (2. * at * bt + a * bt2)) / (b * b * b))
    static member Pow (Dual2(a, at, at2), Dual2(b, bt, bt2)) = Dual2(a ** b, (a ** b) * ((b * at / a) + ((log a) * bt)), a ** b * (((b - 1.) * b * at * at) / (a * a) + (b * at2 + 2. * at * bt * (b * (log a) + 1.)) / a + (log a) * ((log a) * bt * bt + bt2)))
    static member (+) (Dual2(a, at, at2), b) = Dual2(a + b, at, at2)
    static member (-) (Dual2(a, at, at2), b) = Dual2(a - b, at, at2)
    static member (*) (Dual2(a, at, at2), b) = Dual2(a * b, at * b, at2 * b)
    static member (/) (Dual2(a, at, at2), b) = Dual2(a / b, at / b, at2 / b)
    static member Pow (Dual2(a, at, at2), b) = Dual2(a ** b, b * (a ** (b - 1.)) * at, a * (a ** (a - 2.)) * ((a - 1.) * at * at + a * at2))
    static member (+) (a, Dual2(b, bt, bt2)) = Dual2(b + a, bt, bt2)
    static member (-) (a, Dual2(b, bt, bt2)) = Dual2(a - b, -bt, -bt2)
    static member (*) (a, Dual2(b, bt, bt2)) = Dual2(b * a, bt * a, bt2 * a)
    static member (/) (a, Dual2(b, bt, bt2)) = Dual2(a / b, -a * bt / (b * b), a * ((2. * bt * bt / (a * a * a)) - (bt2 / (a * a))))
    static member Pow (a, Dual2(b, bt, bt2)) = Dual2(a ** b, (a ** b) * (log a) * bt, (a ** a) * (log a) * ((log a) * bt * bt + bt2))
    static member Log (Dual2(a, at, at2)) = Dual2(log a, at / a, (-at * at + a * at2) / (a * a))
    static member Exp (Dual2(a, at, at2)) = Dual2(exp a, at * exp a, (exp a) * (at * at + at2))
    static member Sin (Dual2(a, at, at2)) = Dual2(sin a, at * cos a, -(sin a) * at * at + (cos a) * at2)
    static member Cos (Dual2(a, at, at2)) = Dual2(cos a, -at * sin a, -(cos a) * at * at - (sin a) * at2)
    static member Tan (Dual2(a, at, at2)) = Dual2(tan a, at / ((cos a) * (cos a)), (2. * (tan a) * at * at + at2) / ((cos a) * (cos a)))
    static member (~-) (Dual2(a, at, at2)) = Dual2(-a, -at, -at2)
    static member Sqrt (Dual2(a, at, at2)) = Dual2(sqrt a, at / (2. * sqrt a), (-at * at + 2. * a * at2) / (4. * a ** 1.5))
    static member Sinh (Dual2(a, at, at2)) = Dual2(sinh a, at * cosh a, (sinh a) * at * at + (cosh a) * at2)
    static member Cosh (Dual2(a, at, at2)) = Dual2(cosh a, at * sinh a, (cosh a) * at * at + (sinh a) * at2)
    static member Tanh (Dual2(a, at, at2)) = Dual2(tanh a, at / ((cosh a) * (cosh a)), (-2. * (tanh a) * at * at + at2) / ((cosh a) * (cosh a)))
    static member Asin (Dual2(a, at, at2)) = Dual2(asin a, at / sqrt (1. - a * a), (a * at * at - (a * a - 1.) * at2) / (1. - a * a) ** 1.5)
    static member Acos (Dual2(a, at, at2)) = Dual2(acos a, -at / sqrt (1. - a * a), -((a * at * at + at2 - a * a * at2) / (1. - a * a) ** 1.5))
    static member Atan (Dual2(a, at, at2)) = Dual2(atan a, at / (1. + a * a), (-2. * a * at * at + (1. + a * a) * at2) / (1. + a * a) ** 2.)


/// Dual2 operations module (automatically opened)
[<AutoOpen>]
module Dual2Ops =
    /// Make Dual2, with primal value `p`, tangent 0, and tangent-of-tangent 0
    let inline dual2 p = Dual2(p, 0., 0.)
    /// Make Dual2, with primal value `p`, tangent value `t`, and tangent-of-tangent value `t2`
    let inline dual2Set (p, t, t2) = Dual2(p, t, t2)
    /// Make active Dual2 (i.e. variable of differentiation), with primal value `p`, tangent 1, and tangent-of-tangent 0
    let inline dual2Act p = Dual2(p, 1., 0.)
    /// Make an array of arrays of Dual2, with primal values given in array `x`. The tangent values along the diagonal are 1, the rest are 0.
    let inline dual2ActArrayArray (x:float[]) = Array.init x.Length (fun i -> (Array.init x.Length (fun j -> if i = j then dual2Act x.[j] else dual2 x.[j])))
    /// Make a list of Dual2, given a list of primal values `p`
    let inline dual2List p = List.map dual2 p
    /// Get the primal value of a Dual2
    let inline primal (Dual2(p, _, _)) = p
    /// Get the tangent value of a Dual2
    let inline tangent (Dual2(_, t, _)) = t
    /// Get the tangent-of-tangent value of a Dual2
    let inline tangent2 (Dual2(_, _, t2)) = t2
    /// Get the primal and tangent values of a Dual2, as a tuple
    let inline tuple (Dual2(p, t, _)) = (p, t)
    /// Get the primal and tangent-of-tangent values of a Dual2, as a tuple
    let inline tuple2 (Dual2(p, _, t2)) = (p, t2)
    /// Get the primal, tangent, and tangent-of-tangent values of a Dual2, as a tuple
    let inline tupleAll (Dual2(p, t, t2)) = (p, t, t2)


/// Forward2 differentiation operations module (automatically opened)
[<AutoOpen>]
module Forward2Ops =
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        dual2Act >> f >> tupleAll

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        dual2Act >> f >> tangent

    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f =
        dual2Act >> f >> tangent2

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f =
        dual2Act >> f >> tuple2

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        fun x ->
            let a = Array.map f (dual2ActArrayArray x)
            (primal a.[0], Array.map tangent a)

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd

    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f =
        fun x ->
            let a = Array.map f (dual2ActArrayArray x)
            (let (Dual2(p, _, _)) = a.[0] in p, Array.sumBy tangent2 a)

    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f =
        laplacian' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f =
        fun x ->
            let a = Array.map f (dual2ActArrayArray x)
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