//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
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
//   Brain and Computation Lab
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
// HIGHLY UNOPTIMIZED PLACEHOLDER CODE
[<AutoOpen>]
module NumericalOps =
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:float->float) x =
        ((f (x + eps)) - (f (x - eps))) / deps
    
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        (f x, diff f x)

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:float[]->float) (x:float[]) (v:float[]) =
        let veps = eps * (vector v)
        let xv = vector x
        ((f (Vector.toArray (xv + veps))) - (f (Vector.toArray (xv - veps)))) / deps

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
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
        let xv = vector x
        let fx = f x
        let g = Vector.create x.Length fx
        let gg = Vector.init x.Length (fun i -> f (Vector.toArray (xv + Vector.Create(x.Length, i, eps))))
        (fx, Vector.toArray ((gg - g) / eps))
    
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let grad f x =
        grad' f x |> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x =
        let xv = vector x
        let (fx, g) = grad' f x
        let h = Matrix.Create(x.Length, g)
        let hh = Matrix.Create(x.Length, fun i -> grad f (Vector.toArray (xv + Vector.Create(x.Length, i, eps))))
        (fx, g, Matrix.toArray2D ((hh - h) / eps))

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x =
        gradhessian' f x |> sndtrd

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x =
        gradhessian' f x |> fsttrd
                
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        gradhessian' f x |> trd

    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline hessianv' (f:float[]->float) (x:float[]) (v:float[]) =
        let xv = vector x
        let veps = eps * (vector v)
        let fx, gg1 = grad' f (Vector.toArray (xv + veps))
        let gg2 = grad f (Vector.toArray (xv - veps))
        (fx, Vector.toArray (((vector gg1) - (vector gg2)) / deps))

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline hessianv (f:float[]->float) x v =
        hessianv' f x v |> snd

    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar funtion `f`, at point `x`, along vector `v`
    let inline gradhessianv' (f:float[]->float) x v =
        let fx, gv = gradv' f x v
        let hv = hessianv f x v
        (fx, gv, hv)

    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradhessianv (f:float[]->float) x v =
        gradhessianv' f x v |> sndtrd

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
        let jj = Matrix.Create(x.Length, fun i -> f (Vector.toArray (xv + Vector.Create(x.Length, i, eps))))
        (fx, Matrix.toArray2D ((jj - j) / eps))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:float[]->float[]) x v =
        let veps = eps * Vector.Create(v)
        let xv = Vector.Create(x)
        Vector.toArray ((vector (f (Vector.toArray (xv + veps))) - vector (f (Vector.toArray (xv - veps)))) / deps)

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v =
        (f x, jacobianv f x v)

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurl()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x =
        curl' f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x =
        let v, j = jacobianT' f x
        if Array2D.length1 j <> Array2D.length2 j then invalidArgDiv()
        v, trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x =
        div' f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurlDiv()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x =
        curldiv' f x |> sndtrd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:float->float) x = NumericalOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:float->float) x = NumericalOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' (f:float->float) x = NumericalOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 (f:float->float) x = NumericalOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' (f:float->float) x = NumericalOps.diff2'' f x
    /// Original value and directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' (f:Vector<float>->float) x v = NumericalOps.gradv' (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:Vector<float>->float) x v = NumericalOps.gradv (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<float>->float) x = NumericalOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<float>->float) x = NumericalOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<float>->float) x = NumericalOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian (f:Vector<float>->float) x = NumericalOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<float>->Vector<float>) x = NumericalOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<float>->Vector<float>) x = NumericalOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<float>->Vector<float>) x = NumericalOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<float>->Vector<float>) x = NumericalOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<float>->float) x = NumericalOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2d b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<float>->float) x = NumericalOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<float>->float) x = NumericalOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2d c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<float>->float) x = NumericalOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' (f:Vector<float>->Vector<float>) x v = NumericalOps.jacobianv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:Vector<float>->Vector<float>) x v = NumericalOps.jacobianv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<float>->Vector<float>) x = NumericalOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<float>->Vector<float>) x = NumericalOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<float>->Vector<float>) x = NumericalOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<float>->Vector<float>) x = NumericalOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<float>->Vector<float>) x = NumericalOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<float>->Vector<float>) x = NumericalOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
