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

open DiffSharp.Util.LinearAlgebra
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
    static member (/) (Dual2(a, at, at2), Dual2(b, bt, bt2)) = let bsq = b * b in Dual2(a / b, (at * b - a * bt) / bsq, (2. * a * bt * bt + bsq * at2 - b * (2. * at * bt + a * bt2)) / (bsq * b))
    static member Pow (Dual2(a, at, at2), Dual2(b, bt, bt2)) = let apowb, loga, btimesat = a ** b, log a, b * at in Dual2(apowb, apowb * ((btimesat / a) + (loga * bt)), apowb * (((b - 1.) * btimesat * at) / (a * a) + (b * at2 + 2. * at * bt * (b * loga + 1.)) / a + loga * (loga * bt * bt + bt2)))
    static member (+) (Dual2(a, at, at2), b) = Dual2(a + b, at, at2)
    static member (-) (Dual2(a, at, at2), b) = Dual2(a - b, at, at2)
    static member (*) (Dual2(a, at, at2), b) = Dual2(a * b, at * b, at2 * b)
    static member (/) (Dual2(a, at, at2), b) = Dual2(a / b, at / b, at2 / b)
    static member Pow (Dual2(a, at, at2), b) = Dual2(a ** b, b * (a ** (b - 1.)) * at, b * (a ** (b - 2.)) * ((b - 1.) * at * at + a * at2))
    static member (+) (a, Dual2(b, bt, bt2)) = Dual2(b + a, bt, bt2)
    static member (-) (a, Dual2(b, bt, bt2)) = Dual2(a - b, -bt, -bt2)
    static member (*) (a, Dual2(b, bt, bt2)) = Dual2(b * a, bt * a, bt2 * a)
    static member (/) (a, Dual2(b, bt, bt2)) = let atimesa = a * a in Dual2(a / b, -a * bt / (b * b), a * ((2. * bt * bt / (atimesa * a)) - (bt2 / atimesa)))
    static member Pow (a, Dual2(b, bt, bt2)) = let apowb, loga = a ** b, log a in Dual2(apowb, apowb * loga * bt, apowb * loga * (loga * bt * bt + bt2))
    static member Log (Dual2(a, at, at2)) = Dual2(log a, at / a, (-at * at + a * at2) / (a * a))
    static member Exp (Dual2(a, at, at2)) = let expa = exp a in Dual2(expa, at * expa, expa * (at * at + at2))
    static member Sin (Dual2(a, at, at2)) = let sina, cosa = sin a, cos a in Dual2(sina, at * cosa, -sina * at * at + cosa * at2)
    static member Cos (Dual2(a, at, at2)) = let cosa, sina = cos a, sin a in Dual2(cosa, -at * sina, -cosa * at * at - sina * at2)
    static member Tan (Dual2(a, at, at2)) = let tana, secsqa = tan a, 1. / ((cos a) * (cos a)) in Dual2(tana, at * secsqa, (2. * tana * at * at + at2) * secsqa)
    static member (~-) (Dual2(a, at, at2)) = Dual2(-a, -at, -at2)
    static member Sqrt (Dual2(a, at, at2)) = let sqrta = sqrt a in Dual2(sqrta, at / (2. * sqrta), (-at * at + 2. * a * at2) / (4. * a ** 1.5))
    static member Sinh (Dual2(a, at, at2)) = let sinha, cosha = sinh a, cosh a in Dual2(sinha, at * cosha, sinha * at * at + cosha * at2)
    static member Cosh (Dual2(a, at, at2)) = let cosha, sinha = cosh a, sinh a in Dual2(cosha, at * sinha, cosha * at * at + sinha * at2)
    static member Tanh (Dual2(a, at, at2)) = let tanha, sechsqa = tanh a, 1. / ((cosh a) * (cosh a)) in Dual2(tanha, at * sechsqa, (-2. * tanha * at * at + at2) * sechsqa)
    static member Asin (Dual2(a, at, at2)) = let asq = a * a in Dual2(asin a, at / sqrt (1. - asq), (a * at * at - (asq - 1.) * at2) / (1. - asq) ** 1.5)
    static member Acos (Dual2(a, at, at2)) = let asq = a * a in Dual2(acos a, -at / sqrt (1. - asq), -((a * at * at + at2 - asq * at2) / (1. - asq) ** 1.5))
    static member Atan (Dual2(a, at, at2)) = let asq = a * a in Dual2(atan a, at / (1. + asq), (-2. * a * at * at + (1. + asq) * at2) / (1. + asq) ** 2.)


/// Dual2 operations module (automatically opened)
[<AutoOpen>]
module Dual2Ops =
    /// Make Dual2, with primal value `p`, tangent 0, and tangent-of-tangent 0
    let inline dual2 p = Dual2(p, 0., 0.)
    /// Make Dual2, with primal value `p`, tangent value `t`, and tangent-of-tangent 0
    let inline dual2Set (p, t) = Dual2(p, t, 0.)
    /// Make Dual2, with primal value `p`, tangent value `t`, and tangent-of-tangent value `t2`
    let inline dual2Set2 (p, t, t2) = Dual2(p, t, t2)
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

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f =
        dual2Act >> f >> tuple2

    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f =
        dual2Act >> f >> tangent2
        
    /// Original value and directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir' r f =
        fun x -> Array.zip x r |> Array.map dual2Set |> f |> tuple

    /// Directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir r f =
        diffdir' r f >> snd

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


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f = Forward2Ops.diff' f
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f = Forward2Ops.diff f
    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f = Forward2Ops.diff2' f
    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f = Forward2Ops.diff2 f
    /// Original value and directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir' r f = array >> Forward2Ops.diffdir' (array r) f
    /// Directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir r f = array >> Forward2Ops.diffdir (array r) f
    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f = array >> Forward2Ops.grad' f >> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`
    let inline grad f = array >> Forward2Ops.grad f >> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f = array >> Forward2Ops.laplacian' f
    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f = array >> Forward2Ops.laplacian f
    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f = array >> Forward2Ops.jacobianT' f >> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f = array >> Forward2Ops.jacobianT f >> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f = array >> Forward2Ops.jacobian' f >> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f = array >> Forward2Ops.jacobian f >> matrix