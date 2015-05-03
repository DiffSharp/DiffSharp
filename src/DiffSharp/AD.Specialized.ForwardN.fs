//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
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

#light

/// Non-nested forward mode AD, lazy higher-order
namespace DiffSharp.AD.Specialized.ForwardN

open DiffSharp.Util
open FsAlg.Generic

/// Numeric type where the tangent value is another D, forming a lazy chain of higher-order derivatives
[<CustomEquality; CustomComparison>]
type D =
    // Primal, tangent
    | D of float * Lazy<D>
    static member Create(p) = D(p, lazy (D.Zero))
    static member Create(p, t) = D(p, lazy (D.Create(t)))
    static member op_Explicit(D(p, _)):float = p
    static member op_Explicit(D(p, _)):int = int p
    override d.ToString() = let (D(p, t)) = d in sprintf "D (%A, %A)" p (float t.Value)
    member d.P = let (D(p, _)) = d in p
    member d.T = let (D(_, t)) = d in t.Value
    static member Zero = D(0., lazy (D.Zero))
    static member One = D(1., lazy (D.Zero))
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> let D(a, _), D(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this D with another type of object."
    override d.Equals(other) = 
        match other with
        | :? D as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (D(a, b)) = d in hash [|a; b|]
    // D - D binary operations
    static member (+) (a:D, b:D) = D(a.P + b.P, lazy (a.T + b.T))
    static member (-) (a:D, b:D) = D(a.P - b.P, lazy (a.T - b.T))
    static member (*) (a:D, b:D) = D(a.P * b.P, lazy (a.T * b + a * b.T))
    static member (/) (a:D, b:D) = D(a.P / b.P, lazy ((a.T * b - a * b.T) / (b * b)))
    static member Pow (a:D, b:D) = D(a.P ** b.P, lazy ((a ** b) * ((b * a.T / a) + ((log a) * b.T))))
    static member Atan2 (a:D, b:D) = D(atan2 a.P b.P, lazy ((a.T * b - a * b.T) / (a * a + b * b)))
    // D - float binary operations
    static member (+) (a:D, b) = D(a.P + b, lazy (a.T))
    static member (-) (a:D, b) = D(a.P - b, lazy (a.T))
    static member (*) (a:D, b) = D(a.P * b, lazy (a.T * b))
    static member (/) (a:D, b) = D(a.P / b, lazy (a.T / b))
    static member Pow (a:D, b) = D(a.P ** b, lazy (b * (a ** (b - 1.)) * a.T))
    static member Atan2 (a:D, b) = D(atan2 a.P b, lazy ((b * a.T) / (b * b + a * a)))
    // float - D binary operations
    static member (+) (a, b:D) = D(b.P + a, lazy (b.T))
    static member (-) (a, b:D) = D(a - b.P, lazy (-b.T))
    static member (*) (a, b:D) = D(b.P * a, lazy (b.T * a))
    static member (/) (a, b:D) = D(a / b.P, lazy (-a * b.T / (b * b)))
    static member Pow (a, b:D) = D(a ** b.P, lazy ((D.Create(a) ** b) * (log a) * b.T))
    static member Atan2 (a, b:D) = D(atan2 a b.P, lazy (-(a * b.T) / (a * a + b * b)))
    // D - int binary operations
    static member (+) (a:D, b:int) = a + float b
    static member (-) (a:D, b:int) = a - float b
    static member (*) (a:D, b:int) = a * float b
    static member (/) (a:D, b:int) = a / float b
    static member Pow (a:D, b:int) = D.Pow(a, float b)
    static member Atan2 (a:D, b:int) = D.Atan2(a, float b)
    // int - D binary operations
    static member (+) (a:int, b:D) = (float a) + b
    static member (-) (a:int, b:D) = (float a) - b
    static member (*) (a:int, b:D) = (float a) * b
    static member (/) (a:int, b:D) = (float a) / b
    static member Pow (a:int, b:D) = D.Pow(float a, b)
    static member Atan2 (a:int, b:D) = D.Atan2(float a, b)
    // D unary operations
    static member Log (a:D) = 
        if a.P <= 0. then invalidArgLog()
        D(log a.P, lazy (a.T / a))
    static member Log10 (a:D) = 
        if a.P <= 0. then invalidArgLog10()
        D(log10 a.P, lazy (a.T / (a * log10val)))
    static member Exp (a:D) = D(exp a.P, lazy (a.T * exp a))
    static member Sin (a:D) = D(sin a.P, lazy (a.T * cos a))
    static member Cos (a:D) = D(cos a.P, lazy (-a.T * sin a))
    static member Tan (a:D) = 
        if cos a.P = 0. then invalidArgTan()
        D(tan a.P, lazy (a.T / ((cos a) * (cos a))))
    static member (~-) (a:D) = D(-a.P, lazy (-a.T))
    static member Sqrt (a:D) = 
        if a.P <= 0. then invalidArgSqrt()
        D(sqrt a.P, lazy (a.T / (2. * sqrt a)))
    static member Sinh (a:D) = D(sinh a.P, lazy (a.T * cosh a))
    static member Cosh (a:D) = D(cosh a.P, lazy (a.T * sinh a))
    static member Tanh (a:D) = D(tanh a.P, lazy (a.T / ((cosh a) * (cosh a))))
    static member Asin (a:D) = 
        if (abs a.P) >= 1. then invalidArgAsin()
        D(asin a.P, lazy (a.T / sqrt (1. - a * a)))
    static member Acos (a:D) = 
        if (abs a.P) >= 1. then invalidArgAcos()
        D(acos a.P, lazy (-a.T / sqrt (1. - a * a)))
    static member Atan (a:D) = D(atan a.P, lazy (a.T / (1. + a * a)))
    static member Abs (a:D) = 
        if a.P = 0. then invalidArgAbs()
        D(abs a.P, lazy (a.T * float (sign a.P)))
    static member Floor (a:D) =
        if isInteger a.P then invalidArgFloor()
        D(floor a.P, lazy (D.Zero))
    static member Ceiling (a:D) =
        if isInteger a.P then invalidArgCeil()
        D(ceil a.P, lazy (D.Zero))
    static member Round (a:D) =
        if isHalfway a.P then invalidArgRound()
        D(round a.P, lazy (D.Zero))

/// D operations module (automatically opened)
[<AutoOpen>]
module DOps =
    /// Make D, with primal value `p` and tangent 0
    let inline makeD p = D.Create(float p, 0.)
    /// Make D, with primal value `p` and tangent value `t`
    let inline makeDPT p t = D.Create(float p, float t)
    /// Make active D (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline makeDP1 p = D.Create(float p, 1.)
    /// Get the primal value of a D
    let inline primal (D(p, _)) = p
    /// Get the tangent value of a D
    let inline tangent (D(_, t)) = float t.Value
    /// Get the tangent-of-tangent value of a D
    let inline tangent2 (d:D) = tangent d.T
    /// Get the primal and tangent value of a D, as a tuple
    let inline tuple (D(p, t)) = (p, float t.Value)

    /// Compute the `n`-th derivative of a D
    let rec diffLazy n =
        match n with
        | a when a < 0 -> invalidArgDiffn()
        | 0 -> fun (x:D) -> x
        | 1 -> fun x -> x.T
        | _ -> fun x -> diffLazy (n - 1) x.T

    /// Custom operator (/^) for differentiation. Usage: `x` /^ `n`, value of the `n`-th order derivative of `x`.
    let ( /^ ) x n =
        diffLazy n x |> primal


/// ForwardN differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f (x:float) =
        x |> makeDP1 |> f |> tuple
    
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f (x:float) =
        x |> makeDP1 |> f |> tangent

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f (x:float) =
        x |> makeDP1 |> f |> fun a -> (primal a, tangent2 a)
        
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f (x:float) =
        x |> makeDP1 |> f |> tangent2

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f (x:float) =
        x |> makeDP1 |> f |> fun a -> (primal a, tangent a, tangent2 a)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn n f (x:float) =
        x |> makeDP1 |> f |> diffLazy n |> primal

    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' n f (x:float) =
        let orig = x |> makeDP1 |> f
        let d = orig |> diffLazy n
        (primal orig, primal d)

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f (x:float[]) (v:float[]) =
        Array.map2 makeDPT x v |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f (x:float[]) =
        let a = Array.init x.Length (fun i -> gradv' f x (standardBasis x.Length i))
        (fst a.[0], Array.map snd a)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
            
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f (x:float[]) =
        let a = Array.init x.Length (fun i ->
                                        standardBasis x.Length i
                                        |> Array.map2 makeDPT x
                                        |> f)
        (primal a.[0], Array.sumBy tangent2 a)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f (x:float[]) (v:float[]) = 
        Array.map2 makeDPT x v |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f x v = 
        jacobianv' f x v |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f (x:float[]) =
        let a = Array.init x.Length (fun i -> jacobianv' f x (standardBasis x.Length i))
        (fst a.[0], array2D (Array.map snd a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (v, j) -> (v, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

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
    let inline diff' (f:D->D) x = DiffOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:D->D) x = DiffOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' (f:D->D) x = DiffOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 (f:D->D) x = DiffOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' (f:D->D) x = DiffOps.diff2'' f x
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' (n:int) (f:D->D) x = DiffOps.diffn' n f x 
    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn (n:int) (f:D->D) x = DiffOps.diffn n f x
    /// Original value and directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' (f:Vector<D>->D) x v = DiffOps.gradv' (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:Vector<D>->D) x v = DiffOps.gradv (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<D>->D) x = DiffOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<D>->D) x = DiffOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<D>->D) x = DiffOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point x
    let inline laplacian (f:Vector<D>->D) x = DiffOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<D>->Vector<D>) x = DiffOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<D>->Vector<D>) x = DiffOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' (f:Vector<D>->Vector<D>) x v = DiffOps.jacobianv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:Vector<D>->Vector<D>) x v = DiffOps.jacobianv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<D>->Vector<D>) x = DiffOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<D>->Vector<D>) x = DiffOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<D>->Vector<D>) x = DiffOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<D>->Vector<D>) x = DiffOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<D>->Vector<D>) x = DiffOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<D>->Vector<D>) x = DiffOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
