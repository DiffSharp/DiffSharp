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

/// Numerical differentiation module
module DiffSharp.Numerical

open DiffSharp.Util
open DiffSharp.Util.General

/// Numerical differentiation operations module (automatically opened)
[<AutoOpen>]
module NumericalOps =
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff (f:float->float) =
        fun x -> ((f (x + eps)) - (f (x - eps))) / deps
    
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        fun x -> (f x, diff f x)

    /// Directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir r (f:float[]->float) =
        let reps = eps * Vector.Create(r)
        fun x ->
            let xv = Vector.Create(x)
            ((f ((xv + reps).V)) - (f ((xv - reps).V))) / deps

    /// Original value and directional derivative of a vector-to-scalar function `f`, with direction `r`
    let inline diffdir' r f =
        fun x -> (f x, diffdir r f x)

    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f =
        fun x -> ((f (x + eps)) - 2. * (f x) + (f (x - eps))) / epssq

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f =
        fun x -> (f x, diff2 f x)

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' (f:float[]->float) =
        fun x ->
            let xv = Vector.Create(x)
            let fx = f x
            let g = Vector.Create(x.Length, fx)
            let gg = Vector.Create(x.Length, fun i -> f (xv + Vector.Create(x.Length, i, eps)).V)
            (fx, ((gg - g) / eps).V)
    
    /// Gradient of a vector-to-scalar function `f`
    let grad f =
        grad' f >> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f =
        fun x ->
            let xv = Vector(x)
            let (fx, g) = grad' f x
            let h = Matrix.Create(x.Length, g)
            let hh = Matrix.Create(x.Length, fun i -> grad f (xv + Vector.Create(x.Length, i, eps)).V)
            (fx, g, ((hh - h) / eps).M)

    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f =
        gradhessian' f >> sndtrd

    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f =
        gradhessian' f >> fsttrd
                
    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f =
        gradhessian' f >> trd

    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f =
        fun x -> let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f =
        laplacian' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' (f:float[]->float[]) =
        fun x ->
            let xv = Vector(x)
            let fx = f x
            let j = Matrix.Create(x.Length, fx)
            let jj = Matrix.Create(x.Length, fun i -> f (xv + Vector.Create(x.Length, i, eps)).V)
            (fx, ((jj - j) / eps).M)

    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f =
        jacobianT' f >> snd

    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        jacobianT' f >> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f =
        jacobian' f >> snd