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

/// Nested forward mode AD
namespace DiffSharp.AD.Forward

open DiffSharp.Util
open FsAlg.Generic

/// Dual numeric type with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]
type D =
    | D of float // Primal
    | DF of D * D * uint32 // Primal, tangent, tag
    static member op_Explicit(d:D):float =
        match d with
        | D(a) -> a
        | DF(ap, _, _) -> float ap
    static member op_Explicit(d:D):int =
        match d with
        | D(a) -> int a
        | DF(ap, _, _) -> int ap
    static member DivideByInt(d:D, i:int) =
        match d with
        | D(a) -> D(a / float i)
        | DF(ap, at, ai) -> DF(ap/ float i, at / float i, ai)
    static member Zero = D 0.
    static member One = D 1.
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> compare ((float) d) ((float) d2)
            | _ -> invalidArg "" "Cannot compare this D with another type of object."
    override d.Equals(other) =
        match other with
        | :? D as d2 -> compare ((float) d) ((float) d2) = 0
        | _ -> false
    override d.GetHashCode() =
        match d with
        | D(a) -> hash [| a |]
        | DF(ap, at, ai) -> hash [|ap; at; ai|]
    // D - D binary operations
    static member (+) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a + b)
        | D(a),           DF(bp, bt, bi) -> DF(a + bp, bt, bi)
        | DF(ap, at, ai), D(b)           -> DF(ap + b, at, ai)
        | DF(_ , _ , ai), DF(bp, bt, bi) when ai < bi -> DF(a + bp, bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap + bp, at + bt, ai)
        | DF(ap, at, ai), DF(_ , _ , bi) when ai > bi -> DF(ap + b, at, ai)
    static member (-) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a - b)
        | D(a),           DF(bp, bt, bi) -> DF(a - bp, -bt, bi)
        | DF(ap, at, ai), D(b)           -> DF(ap - b, at, ai)
        | DF(_ , _ , ai), DF(bp, bt, bi) when ai < bi -> DF(a - bp, -bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap - bp, at - bt, ai)
        | DF(ap, at, ai), DF(_ , _ , bi) when ai > bi -> DF(ap - b, at, ai)
    static member (*) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a * b)
        | D(a),           DF(bp, bt, bi) -> DF(a * bp, a * bt, bi)
        | DF(ap, at, ai), D(b)           -> DF(ap * b, at * b, ai)
        | DF(_ , _ , ai), DF(bp, bt, bi) when ai < bi -> DF(a * bp, bt * a, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap * bp, at * bp + bt * ap, ai)
        | DF(ap, at, ai), DF(_ , _ , bi) when ai > bi -> DF(ap * b, at * b, ai)
    static member (/) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a / b)
        | D(a),           DF(bp, bt, bi) -> DF(a / bp, -(bt * a) / (bp * bp), bi)
        | DF(ap, at, ai), D(b)           -> DF(ap / b, at / b, ai)
        | DF(_ , _ , ai), DF(bp, bt, bi) when ai < bi -> DF(a / bp, -(bt * a) / (bp * bp), bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap / bp, (at * bp - bt * ap) / (bp * bp), ai)
        | DF(ap, at, ai), DF(_ , _ , bi) when ai > bi -> DF(ap / b, at / b, ai)
    static member Pow (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a ** b)
        | D(a),           DF(bp, bt, bi) -> let apb = D.Pow(a, bp) in DF(apb, apb * (log a) * bt, bi)
        | DF(ap, at, ai), D(b)           -> DF(ap ** b, b * (ap ** (b - 1.)) * at, ai)
        | DF(_ , _ , ai), DF(bp, bt, bi) when ai < bi -> let apb = a ** bp in DF(apb, apb * (log a) * bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> let apb = ap ** bp in DF(apb, apb * ((bp * at / ap) + ((log ap) * bt)), ai)
        | DF(ap, at, ai), DF(_ , _ , bi) when ai > bi -> let apb = ap ** b in DF(apb, apb * (b * at / ap), ai)
    static member Atan2 (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(atan2 a b)
        | D(a),           DF(bp, bt, bi) -> DF(D.Atan2(a, bp), -(a * bt) / (a * a + bp * bp), bi)
        | DF(ap, at, ai), D(b)           -> DF(D.Atan2(ap, b), (b * at) / (ap * ap + b * b), ai)
        | DF(_ , _ , ai), DF(bp, bt, bi) when ai < bi -> DF(atan2 a bp, -(a * bt) / (a * a + bp * bp), bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(atan2 ap bp, (at * bp - ap * bt) / (ap * ap + bp * bp), ai)
        | DF(ap, at, ai), DF(_ , _ , bi) when ai > bi -> DF(atan2 ap b, (at * b) / (ap * ap + b * b), ai)
    // D - float binary operations
    static member (+) (a:D, b:float) = a + (D b)
    static member (-) (a:D, b:float) = a - (D b)
    static member (*) (a:D, b:float) = a * (D b)
    static member (/) (a:D, b:float) = a / (D b)
    static member Pow (a:D, b:float) = a ** (D b)
    static member Atan2 (a:D, b:float) = atan2 a (D b)
    // float - D binary operations
    static member (+) (a:float, b:D) = (D a) + b
    static member (-) (a:float, b:D) = (D a) - b
    static member (*) (a:float, b:D) = (D a) * b
    static member (/) (a:float, b:D) = (D a) / b
    static member Pow (a:float, b:D) = (D a) ** b
    static member Atan2 (a:float, b:D) = atan2 (D a) b
    // D - int binary operations
    static member (+) (a:D, b:int) = a + (D (float b))
    static member (-) (a:D, b:int) = a - (D (float b))
    static member (*) (a:D, b:int) = a * (D (float b))
    static member (/) (a:D, b:int) = a / (D (float b))
    static member Pow (a:D, b:int) = D.Pow(a, (D (float b)))
    static member Atan2 (a:D, b:int) = D.Atan2(a, (D (float b)))
    // int - D binary operations
    static member (+) (a:int, b:D) = (D (float a)) + b
    static member (-) (a:int, b:D) = (D (float a)) - b
    static member (*) (a:int, b:D) = (D (float a)) * b
    static member (/) (a:int, b:D) = (D (float a)) / b
    static member Pow (a:int, b:D) = D.Pow((D (float a)), b)
    static member Atan2 (a:int, b:D) = D.Atan2((D (float a)), b)
    // D unary operations
    static member Log (a:D) =
        if (float a) <= 0. then invalidArgLog()
        match a with
        | D(ap) -> D(log ap)
        | DF(ap, at, ai) -> DF(log ap, at / ap, ai)
    static member Log10 (a:D) =
        if (float a) <= 0. then invalidArgLog10()
        match a with
        | D(ap) -> D(log10 ap)
        | DF(ap, at, ai) -> DF(log10 ap, at / (ap * log10val), ai)
    static member Exp (a:D) =
        match a with
        | D(ap) -> D(exp ap)
        | DF(ap, at, ai) -> let expa = exp ap in DF(expa, at * expa, ai)
    static member Sin (a:D) =
        match a with
        | D(ap) -> D(sin ap)
        | DF(ap, at, ai) -> DF(sin ap, at * cos ap, ai)
    static member Cos (a:D) =
        match a with
        | D(ap) -> D(cos ap)
        | DF(ap, at, ai) -> DF(cos ap, -at * sin ap, ai)
    static member Tan (a:D) =
        if (float (cos a)) = 0. then invalidArgTan()
        match a with
        | D(ap) -> D(tan ap)
        | DF(ap, at, ai) -> let cosa = cos ap in DF(tan ap, at / (cosa * cosa), ai)
    static member (~-) (a:D) =
        match a with
        | D(ap) -> D(-ap)
        | DF(ap, at, ai) -> DF(-ap, -at, ai)
    static member Sqrt (a:D) =
        if (float a) <= 0. then invalidArgSqrt()
        match a with
        | D(ap) -> D(sqrt ap)
        | DF(ap, at, ai) -> let sqrta = sqrt ap in DF(sqrta, at / (2. * sqrta), ai)
    static member Sinh (a:D) =
        match a with
        | D(ap) -> D(sinh ap)
        | DF(ap, at, ai) -> DF(sinh ap, at * cosh ap, ai)
    static member Cosh (a:D) =
        match a with
        | D(ap) -> D(cosh ap)
        | DF(ap, at, ai) -> DF(cosh ap, at * sinh ap, ai)
    static member Tanh (a:D) =
        match a with
        | D(ap) -> D(tanh ap)
        | DF(ap, at, ai) -> let cosha = cosh ap in DF(tanh ap, at / (cosha * cosha), ai)
    static member Asin (a:D) =
        if abs (float a) >= 1. then invalidArgAsin()
        match a with
        | D(ap) -> D(asin ap)
        | DF(ap, at, ai) -> DF(asin ap, at / sqrt (1. - ap * ap), ai)
    static member Acos (a:D) =
        if abs (float a) >= 1. then invalidArgAcos()
        match a with
        | D(ap) -> D(acos ap)
        | DF(ap, at, ai) -> DF(acos ap, -at / sqrt (1. - ap * ap), ai)
    static member Atan (a:D) =
        match a with
        | D(ap) -> D(atan ap)
        | DF(ap, at, ai) -> DF(atan ap, at / (1. + ap * ap), ai)
    static member Abs (a:D) =
        if (float a) = 0. then invalidArgAbs()
        match a with
        | D(ap) -> D(abs ap)
        | DF(ap, at, ai) -> DF(abs ap, at * float (sign (float ap)), ai)
    static member Floor (a:D) =
        if isInteger (float a) then invalidArgFloor()
        match a with
        | D(ap) -> D(floor ap)
        | DF(ap, _, ai) -> DF(floor ap, D 0., ai)
    static member Ceiling (a:D) =
        if isInteger (float a) then invalidArgCeil()
        match a with
        | D(ap) -> D(ceil ap)
        | DF(ap, _, ai) -> DF(ceil ap, D 0., ai)
    static member Round (a:D) =
        if isHalfway (float a) then invalidArgRound()
        match a with
        | D(ap) -> D(round ap)
        | DF(ap, _, ai) -> DF(round ap, D 0., ai)


/// D operations module (automatically opened)
[<AutoOpen>]
module DOps =
    /// Make DF, with tag `i`, primal value `p`, and tangent value `t`
    let inline makeDF i t p = DF(p, t, i)
    /// Get the primal value of `d`
    let inline primal (d:D) =
        match d with
        | D(_) -> d
        | DF(p,_,_) -> p
    /// Get the tangent value of `d`
    let inline tangent (d:D) =
        match d with
        | D(_) -> D(0.)
        | DF(_,t,_) -> t
    /// Get the primal and tangent values of  `d`, as a tuple
    let inline tuple (d:D) =
        match d with
        | D(_) -> (d, D 0.)
        | DF(p,t,_) -> (p, t)


/// Forward differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        x |> makeDF GlobalTagger.Next (D 1.) |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        x |> makeDF GlobalTagger.Next (D 1.) |> f |> tangent

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 f x =
        diff (diff f) x

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' f x =
        let v, d = diff' f x
        let d2 = diff2 f x
        (v, d, d2)

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' f x =
        diff2'' f x |> fsttrd

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn n f x =
        if n < 0 then invalidArgDiffn()
        elif n = 0 then x |> f
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f
    
    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' n f x =
        (diffn 0 f x, diffn n f x)

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (makeDF i) v x |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f (x:_[]) =
        let a = Array.init x.Length (fun i -> gradv' f x (standardBasis x.Length i))
        (fst a.[0], Array.map snd a)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f (x:_[]) =
        (x |> f, Array.init x.Length (fun i -> x |> fVVtoSS i i (grad f) |> diff <| x.[i]) |> Array.sum)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (makeDF i) v x |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f x v =
        jacobianv' f x v |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f (x:_[]) =
        let a = Array.init x.Length (fun i -> jacobianv' f x (standardBasis x.Length i))
        (fst a.[0], array2D (Array.map snd a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x =
        jacobian' (grad f) x

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x =
        let g, h = gradhessian f x
        (x |> f, g, h)

    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradhessianv' f x v =
        let gv, hv = grad' (fun xx -> gradv f xx v) x
        (x |> f, gv, hv)

    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradhessianv f x v =
        gradhessianv' f x v |> sndtrd

    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline hessianv' f x v =
        gradhessianv' f x v |> fsttrd

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline hessianv f x v =
        hessianv' f x v |> snd

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        jacobian (grad f) x

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x =
        (x |> f, hessian f x)

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
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<D>->D) x = DiffOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<D>->D) x = DiffOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2D b)
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<D>->D) x = DiffOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2D c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<D>->D) x = DiffOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using reverse-on-forward mode AD.
    let inline gradhessianv' (f:Vector<D>->D) x v = DiffOps.gradhessianv' (vector >> f) (Vector.toArray x) (Vector.toArray v) |> fun (a, b, c) -> (a, b, vector c)
    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Computed using reverse-on-forward mode AD.
    let inline gradhessianv (f:Vector<D>->D) x v = DiffOps.gradhessianv (vector >> f) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (a, vector b)
    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessianv' (f:Vector<D>->D) x v = DiffOps.hessianv' (vector >> f) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (a, vector b)
    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`. Computed using reverse-on-forward mode AD.
    let inline hessianv (f:Vector<D>->D) x v = DiffOps.hessianv (vector >> f) (Vector.toArray x) (Vector.toArray v) |> vector
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