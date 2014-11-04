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

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// Numerical differentiation operations module (automatically opened)
[<AutoOpen>]
module NumericalOps =
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:float->float) x =
        ((f (x + eps)) - (f (x - eps))) / deps
    
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        (f x, diff f x)

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, with direction `v`
    let inline gradv (f:float[]->float) x v =
        let veps = eps * Vector.Create(v)
        let xv = Vector.Create(x)
        ((f ((xv + veps).V)) - (f ((xv - veps).V))) / deps

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, with direction `v`
    let inline gradv' f x v =
        (f x, gradv f x v)

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f x =
        ((f (x + eps)) - 2. * (f x) + (f (x - eps))) / epssq

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f x =
        (f x, diff2 f x)

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f x =
        (f x, diff f x, diff2 f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:float[]->float) x =
        let xv = Vector.Create(x)
        let fx = f x
        let g = Vector.Create(x.Length, fx)
        let gg = Vector.Create(x.Length, fun i -> f (xv + Vector.Create(x.Length, i, eps)).V)
        (fx, ((gg - g) / eps).V)
    
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let grad f x =
        grad' f x |> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x =
        let xv = Vector(x)
        let (fx, g) = grad' f x
        let h = Matrix.Create(x.Length, g)
        let hh = Matrix.Create(x.Length, fun i -> grad f (xv + Vector.Create(x.Length, i, eps)).V)
        (fx, g, ((hh - h) / eps).M)

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x =
        gradhessian' f x |> sndtrd

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x =
        gradhessian' f x |> fsttrd
                
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        gradhessian' f x |> trd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x =
        let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:float[]->float[]) x =
        let xv = Vector(x)
        let fx = f x
        let j = Matrix.Create(x.Length, fx)
        let jj = Matrix.Create(x.Length, fun i -> f (xv + Vector.Create(x.Length, i, eps)).V)
        (fx, ((jj - j) / eps).M)

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x = NumericalOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x = NumericalOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f x = NumericalOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f x = NumericalOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f x = NumericalOps.diff2'' f x
    /// Original value and directional derivative of a vector-to-scalar function `f`, at point `x`, with direction `v`
    let inline gradv' f x v = NumericalOps.gradv' f (array x) (array v)
    /// Directional derivative of a vector-to-scalar function `f`, at point `x`, with direction `v`
    let inline gradv f x v = NumericalOps.gradv f (array x) (array v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x = NumericalOps.grad' f (array x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x = NumericalOps.grad f (array x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x = NumericalOps.laplacian' f (array x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x = NumericalOps.laplacian f (array x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x = NumericalOps.jacobianT' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x = NumericalOps.jacobianT f (array x) |> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x = NumericalOps.jacobian' f (array x) |> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x = NumericalOps.jacobian f (array x) |> matrix
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x = NumericalOps.hessian' f (array x) |> fun (a, b) -> (a, matrix b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x = NumericalOps.hessian f (array x) |> matrix
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x = NumericalOps.gradhessian' f (array x) |> fun (a, b, c) -> (a, vector b, matrix c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x = NumericalOps.gradhessian f (array x) |> fun (a, b) -> (vector a, matrix b)