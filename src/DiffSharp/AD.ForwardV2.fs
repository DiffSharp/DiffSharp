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

/// Forward AD module, using a vector of gradient components and a matrix of Hessian components
module DiffSharp.AD.ForwardV2

open DiffSharp.Util
open DiffSharp.Util.General

/// DualV2 numeric type, keeping a triplet of primal value, a vector of gradient components, and a matrix of Hessian components
// NOT FULLY OPTIMIZED
type DualV2 =
    | DualV2 of float * Vector * Matrix
    override d.ToString() = let (DualV2(p, g, h)) = d in sprintf "DualV2 (%f, %A, %A)" p g h
    static member op_Explicit(p) = DualV2(p, Vector.Zero, Matrix.Zero)
    static member op_Explicit(DualV2(p, _, _)) = p
    static member DivideByInt(DualV2(p, g, m), i:int) = DualV2(p / float i, g / float i, m / float i)
    static member Zero = DualV2(0., Vector.Zero, Matrix.Zero)
    static member One = DualV2(1., Vector.Zero, Matrix.Zero)
    static member (+) (DualV2(a, ag, ah), DualV2(b, bg, bh)) = DualV2(a + b, ag + bg, ah + bh)
    static member (-) (DualV2(a, ag, ah), DualV2(b, bg, bh)) = DualV2(a - b, ag - bg, ah - bh)
    static member (*) (DualV2(a, ag, ah), DualV2(b, bg, bh)) = DualV2(a * b, ag * b + a * bg, Matrix.Create(ah.Rows, ah.Cols, fun i j -> ag.[j] * bg.[i] + a * bh.[i, j] + bg.[j] * ag.[i] + b * ah.[i, j]))
    static member (/) (DualV2(a, ag, ah), DualV2(b, bg, bh)) = DualV2(a / b, (ag * b - a * bg) / (b * b), Matrix.Create(ah.Rows, ah.Cols, fun i j -> (2. * a * bg.[j] * bg.[i] + b * b * ah.[i, j] - b * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + a * bh.[i, j])) / (b * b * b)))
    static member Pow (DualV2(a, ag, ah), DualV2(b, bg, bh)) = let loga = log a in DualV2(a ** b, (a ** b) * ((b * ag / a) + (loga * bg)), Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a ** (b - 2.)) * (b * b * ag.[j] * ag.[i] + b * (ag.[j] * (-ag.[i] + loga * a * bg.[i]) + a * (loga * bg.[j] * ag.[i] + ah.[i, j])) + a * (ag.[j] * bg.[i] + bg.[j] * (ag.[i] + loga * loga * a * bg.[i]) + loga * a * bh.[i, j]))))
    static member (+) (DualV2(a, ag, ah), b) = DualV2(a + b, ag, ah)
    static member (-) (DualV2(a, ag, ah), b) = DualV2(a - b, ag, ah)
    static member (*) (DualV2(a, ag, ah), b) = DualV2(a * b, ag * b, Matrix.Create(ah.Rows, ah.Cols, fun i j -> ah.[i, j] * b))
    static member (/) (DualV2(a, ag, ah), b) = DualV2(a / b, ag / b, Matrix.Create(ah.Rows, ah.Cols, fun i j -> ah.[i, j] / b))
    static member Pow (DualV2(a, ag, ah), b) = DualV2(a ** b, (a ** b) * (b * ag / a), Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a ** (b - 2.)) * (b * b * ag.[j] * ag.[i] + b * (a * ah.[i, j] - ag.[j] * ag.[i]))))
    static member (+) (a, DualV2(b, bg, bh)) = DualV2(a + b, bg, bh)
    static member (-) (a, DualV2(b, bg, bh)) = DualV2(a - b, -bg, -bh)
    static member (*) (a, DualV2(b, bg, bh)) = DualV2(a * b, a * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> a * bh.[i, j]))
    static member (/) (a, DualV2(b, bg, bh)) = DualV2(a / b, (-a * bg) / (b * b), Matrix.Create(bh.Rows, bh.Cols, fun i j -> a * (2. * bg.[j] * bg.[i] - b * bh.[i, j]) / (b * b * b)))
    static member Pow (a, DualV2(b, bg, bh)) = let loga = log a in DualV2(a ** b, (a ** b) * loga * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> (a ** (b - 2.)) * a * loga * (bg.[j] * loga * a * bg.[i] + a * bh.[i, j])))
    static member Log (DualV2(a, ag, ah)) = DualV2(log a, ag / a, Matrix.Create(ah.Rows, ah.Cols, fun i j -> - ag.[i] * ag.[j] / (a * a) + ah.[i, j] / a))
    static member Exp (DualV2(a, ag, ah)) = let expa = exp a in DualV2(expa, expa * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> expa * ag.[i] * ag.[j] + expa * ah.[i, j]))
    static member Sin (DualV2(a, ag, ah)) = let (sina, cosa) = (sin a, cos a) in DualV2(sina, cosa * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> -sina * ag.[i] * ag.[j] + cosa * ah.[i, j]))
    static member Cos (DualV2(a, ag, ah)) = let (sina, cosa) = (sin a, cos a) in DualV2(cosa, -sina * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> -cosa * ag.[i] * ag.[j] - sina * ah.[i, j]))
    static member Tan (DualV2(a, ag, ah)) = let (tana, seca) = (tan a, 1. / cos a) in DualV2(tana, (seca * seca) * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> 2. * seca * seca * tana * ag.[i] * ag.[j] + seca * seca * ah.[i, j]))
    static member (~-) (DualV2(a, ag, ah)) = DualV2(-a, -ag, -ah)
    static member Sqrt (DualV2(a, ag, ah)) = let s = 1. / (2. * sqrt a) in DualV2(sqrt a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (s / (-2. * a)) * ag.[i] * ag.[j] + s * ah.[i,j]))
    static member Sinh (DualV2(a, ag, ah)) = let (sinha, cosha) = (sinh a, cosh a) in DualV2(sinha, cosha * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> sinha * ag.[i] * ag.[j] + cosha * ah.[i, j]))
    static member Cosh (DualV2(a, ag, ah)) = let (sinha, cosha) = (sinh a, cosh a) in DualV2(cosha, sinha * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> cosha * ag.[i] * ag.[j] + sinha * ah.[i, j]))
    static member Tanh (DualV2(a, ag, ah)) = let (tanha, secha) = (tanh a, 1. / cosh a) in DualV2(tanha, secha * secha * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> -2. * secha * secha * tanha * ag.[i] * ag.[j] + secha * secha * ah.[i, j]))
    static member Asin (DualV2(a, ag, ah)) = let s = 1. / sqrt (1. - a * a) in DualV2(asin a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a / (1. - a * a)) * s * ag.[i] * ag.[j] + s * ah.[i, j]))
    static member Acos (DualV2(a, ag, ah)) = let s = -1. / sqrt (1. - a * a) in DualV2(acos a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a / (1. - a * a)) * s * ag.[i] * ag.[j] + s * ah.[i, j]))
    static member Atan (DualV2(a, ag, ah)) = let s = 1. / (1. + a * a) in DualV2(atan a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (-2. * a / (1. + a * a)) * s * ag.[i] * ag.[j] + s * ah.[i, j]))


/// DualV2 operations module (automatically opened)
[<AutoOpen>]
module DualV2Ops =
    /// Make DualV2, with primal value `p`, gradient dimension `m`, and all gradient and Hessian components 0
    let inline dualV2 p m = DualV2(p, Vector.Create(m, 0.), Matrix.Create(m, m, 0.))
    /// Make DualV2, with primal value `p`, gradient array `g`, and Hessian 2d array `h`
    let inline dualV2Set (p, g, h) = DualV2(p, Vector.Create(g), Matrix.Create(h))
    /// Make active DualV2 (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the gradient component with index `i` having value 1, the rest of the gradient components 0, and Hessian components 0
    let inline dualV2Act p m i = DualV2(p, Vector.Create(m, i, 1.), Matrix.Create(m, m, 0.))
    /// Make an array of active DualV2, with primal values given in array `x`. For a DualV2 with index _i_, the gradient is the unit vector with 1 in the _i_th place, and the Hessian components are 0.
    let inline dualV2ActArray (x:float[]) = Array.init x.Length (fun i -> dualV2Act x.[i] x.Length i)
    /// Get the primal value of a DualV2
    let inline primal (DualV2(p, _, _)) = p
    /// Get the gradient array of a DualV2
    let inline gradient (DualV2(_, g, _)) = g.V
    /// Get the Hessian 2d array of a DualV2
    let inline hessian (DualV2(_, _, h)) = h.M
    /// Get the primal and the first gradient component of a DualV2, as a tuple
    let inline tuple (DualV2(p, g, _)) = (p, g.V.[0])
    /// Get the primal and the gradient array of a DualV2, as a tuple
    let inline tupleG (DualV2(p, g, _)) = (p, g.V)
    /// Get the primal and Hessian 2d array of a DualV2, as a tuple
    let inline tupleH (DualV2(p, _, h)) = (p, h.M)
    /// Get the primal, the gradient array, and the Hessian 2d array of a DualV2, as a tuple
    let inline tupleGH (DualV2(p, g, h)) = (p, g.V, h.M)


/// ForwardV2 differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardV2Ops =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        fun x -> dualV2Act x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        diff' f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        dualV2ActArray >> f >> tupleG

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        fun x ->
            let a = dualV2ActArray x |> f
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
        dualV2ActArray >> f >> tupleH

    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f =
        hessian' f >> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f =
        dualV2ActArray >> f >> tupleGH
    
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f =
        gradhessian' f >> sndtrd

    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f =
        fun x -> let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f =
        laplacian' f >> snd

