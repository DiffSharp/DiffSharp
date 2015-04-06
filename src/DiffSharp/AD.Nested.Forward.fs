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

#light

/// Nested forward mode AD module
module DiffSharp.AD.Nested.Forward

open DiffSharp.Util.General
open FsAlg.Generic

/// Dual numeric type with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]
type D =
    | D of float
    | DD of D * D * uint64
    static member op_Explicit(d:D) =
        match d with
        | D(a) -> a
        | DD(ap, _, _) -> (float) ap
    static member DivideByInt(d:D, i:int) =
        match d with
        | D(a) -> D(a / float i)
        | DD(ap, at, ai) -> DD(ap/ float i, at / float i, ai)
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
        | DD(ap, at, ai) -> hash [|ap; at; ai|]
    // D - D binary operations
    static member (+) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a + b)
        | D(a),           DD(bp, bt, bi) -> DD(a + bp, bt, bi)
        | DD(ap, at, ai), D(b)           -> DD(ap + b, at, ai)
        | DD(_ , _ , ai), DD(bp, bt, bi) when ai < bi -> DD(a + bp, bt, bi)
        | DD(ap, at, ai), DD(bp, bt, bi) when ai = bi -> DD(ap + bp, at + bt, ai)
        | DD(ap, at, ai), DD(_ , _ , bi) when ai > bi -> DD(ap + b, at, ai)
    static member (-) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a - b)
        | D(a),           DD(bp, bt, bi) -> DD(a - bp, -bt, bi)
        | DD(ap, at, ai), D(b)           -> DD(ap - b, at, ai)
        | DD(_ , _ , ai), DD(bp, bt, bi) when ai < bi -> DD(a - bp, -bt, bi)
        | DD(ap, at, ai), DD(bp, bt, bi) when ai = bi -> DD(ap - bp, at - bt, ai)
        | DD(ap, at, ai), DD(_ , _ , bi) when ai > bi -> DD(ap - b, at, ai)
    static member (*) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a * b)
        | D(a),           DD(bp, bt, bi) -> DD(a * bp, a * bt, bi)
        | DD(ap, at, ai), D(b)           -> DD(ap * b, at * b, ai)
        | DD(_ , _ , ai), DD(bp, bt, bi) when ai < bi -> DD(a * bp, bt * a, bi)
        | DD(ap, at, ai), DD(bp, bt, bi) when ai = bi -> DD(ap * bp, at * bp + bt * ap, ai)
        | DD(ap, at, ai), DD(_ , _ , bi) when ai > bi -> DD(ap * b, at * b, ai)
    static member (/) (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a / b)
        | D(a),           DD(bp, bt, bi) -> DD(a / bp, -(bt * a) / (bp * bp), bi)
        | DD(ap, at, ai), D(b)           -> DD(ap / b, at / b, ai)
        | DD(_ , _ , ai), DD(bp, bt, bi) when ai < bi -> DD(a / bp, -(bt * a) / (bp * bp), bi)
        | DD(ap, at, ai), DD(bp, bt, bi) when ai = bi -> DD(ap / bp, (at * bp - bt * ap) / (bp * bp), ai)
        | DD(ap, at, ai), DD(_ , _ , bi) when ai > bi -> DD(ap / b, at / b, ai)
    static member Pow (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(a ** b)
        | D(a),           DD(bp, bt, bi) -> let apb = D.Pow(a, bp) in DD(apb, apb * (log a) * bt, bi)
        | DD(ap, at, ai), D(b)           -> DD(ap ** b, b * (ap ** (b - 1.)) * at, ai)
        | DD(_ , _ , ai), DD(bp, bt, bi) when ai < bi -> let apb = a ** bp in DD(apb, apb * (log a) * bt, bi)
        | DD(ap, at, ai), DD(bp, bt, bi) when ai = bi -> let apb = ap ** bp in DD(apb, apb * ((bp * at / ap) + ((log ap) * bt)), ai)
        | DD(ap, at, ai), DD(_ , _ , bi) when ai > bi -> let apb = ap ** b in DD(apb, apb * (b * at / ap), ai)
    static member Atan2 (a:D, b:D) =
        match a, b with
        | D(a),           D(b)           -> D(atan2 a b)
        | D(a),           DD(bp, bt, bi) -> DD(D.Atan2(a, bp), -(a * bt) / (a * a + bp * bp), bi)
        | DD(ap, at, ai), D(b)           -> DD(D.Atan2(ap, b), (b * at) / (ap * ap + b * b), ai)
        | DD(_ , _ , ai), DD(bp, bt, bi) when ai < bi -> DD(atan2 a bp, -(a * bt) / (a * a + bp * bp), bi)
        | DD(ap, at, ai), DD(bp, bt, bi) when ai = bi -> DD(atan2 ap bp, (at * bp - ap * bt) / (ap * ap + bp * bp), ai)
        | DD(ap, at, ai), DD(_ , _ , bi) when ai > bi -> DD(atan2 ap b, (at * b) / (ap * ap + b * b), ai)
    // D - float binary operations
    static member (+) (a:D, b:float) =
        match a with
        | D(a) -> D(a + b)
        | DD(ap, at, ai) -> DD(ap + b, at, ai)
    static member (-) (a:D, b:float) =
        match a with
        | D(a) -> D(a - b)
        | DD(ap, at, ai) -> DD(ap - b, at, ai)
    static member (*) (a:D, b:float) =
        match a with
        | D(a) -> D(a * b)
        | DD(ap, at, ai) -> DD(ap * b, at * b, ai)
    static member (/) (a:D, b:float) =
        match a with
        | D(a) -> D(a / b)
        | DD(ap, at, ai) -> DD(ap / b, at / b, ai)
    static member Pow (a:D, b:float) =
        match a with
        | D(a) -> D(a ** b)
        | DD(ap, at, ai) -> DD(ap ** b, b * (ap ** (b - 1.)) * at, ai)
    static member Atan2 (a:D, b:float) =
        match a with
        | D(a) -> D(atan2 a b)
        | DD(ap, at, ai) -> DD(D.Atan2(ap, b), (b * at) / (b * b + ap * ap), ai)
    // float - D binary operations
    static member (+) (a:float, b:D) =
        match b with
        | D(b) -> D(a + b)
        | DD(bp, bt, bi) -> DD(a + bp, bt, bi)
    static member (-) (a:float, b:D) =
        match b with
        | D(b) -> D(a - b)
        | DD(bp, bt, bi) -> DD(a - bp, -bt, bi)
    static member (*) (a:float, b:D) =
        match b with
        | D(b) -> D(a * b)
        | DD(bp, bt, bi) -> DD(a * bp, a * bt, bi)
    static member (/) (a:float, b:D) =
        match b with
        | D(b) -> D(a / b)
        | DD(bp, bt, bi) -> DD(a / bp, -a * bt / (bp * bp), bi)
    static member Pow (a:float, b:D) =
        match b with
        | D(b) -> D(a ** b)
        | DD(bp, bt, bi) -> let apb = D.Pow(a, bp) in DD(apb, apb * (log a) * bt , bi)
    static member Atan2 (a:float, b:D) =
        match b with
        | D(b) -> D(atan2 a b)
        | DD(bp, bt, bi) -> DD(D.Atan2(a, bp), -(a * bt) / (a * a + bp * bp), bi)
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
        if (float a) <= 0. then invalidArgLog()
        match a with
        | D(a) -> D(log a)
        | DD(ap, at, ai) -> DD(log ap, at / ap, ai)
    static member Log10 (a:D) =
        if (float a) <= 0. then invalidArgLog10()
        match a with
        | D(a) -> D(log10 a)
        | DD(ap, at, ai) -> DD(log10 ap, at / (ap * log10val), ai)
    static member Exp (a:D) =
        match a with
        | D(a) -> D(exp a)
        | DD(ap, at, ai) -> let expa = exp ap in DD(expa, at * expa, ai)
    static member Sin (a:D) =
        match a with
        | D(a) -> D(sin a)
        | DD(ap, at, ai) -> DD(sin ap, at * cos ap, ai)
    static member Cos (a:D) =
        match a with
        | D(a) -> D(cos a)
        | DD(ap, at, ai) -> DD(cos ap, -at * sin ap, ai)
    static member Tan (a:D) =
        match a with
        | D(a) -> 
            if cos a = 0. then invalidArgTan()
            D(tan a)
        | DD(ap, at, ai) ->
            let cosa = cos ap
            if (float cosa) = 0. then invalidArgTan()
            DD(tan ap, at / (cosa * cosa), ai)
    static member (~-) (a:D) =
        match a with
        | D(a) -> D(-a)
        | DD(ap, at, ai) -> DD(-ap, -at, ai)
    static member Sqrt (a:D) =
        if (float a) <= 0. then invalidArgSqrt()
        match a with
        | D(a) -> D(sqrt a)
        | DD(ap, at, ai) -> let sqrta = sqrt ap in DD(sqrta, at / (2. * sqrta), ai)
    static member Sinh (a:D) =
        match a with
        | D(a) -> D(sinh a)
        | DD(ap, at, ai) -> DD(sinh ap, at * cosh ap, ai)
    static member Cosh (a:D) =
        match a with
        | D(a) -> D(cosh a)
        | DD(ap, at, ai) -> DD(cosh ap, at * sinh ap, ai)
    static member Tanh (a:D) =
        match a with
        | D(a) -> D(tanh a)
        | DD(ap, at, ai) -> let cosha = cosh ap in DD(tanh ap, at / (cosha * cosha), ai)
    static member Asin (a:D) =
        if abs (float a) >= 1. then invalidArgAsin()
        match a with
        | D(a) -> D(asin a)
        | DD(ap, at, ai) -> DD(asin ap, at / sqrt (1. - ap * ap), ai)
    static member Acos (a:D) =
        if abs (float a) >= 1. then invalidArgAcos()
        match a with
        | D(a) -> D(acos a)
        | DD(ap, at, ai) -> DD(acos ap, -at / sqrt (1. - ap * ap), ai)
    static member Atan (a:D) =
        match a with
        | D(a) -> D(atan a)
        | DD(ap, at, ai) -> DD(atan ap, at / (1. + ap * ap), ai)
    static member Abs (a:D) =
        if (float a) = 0. then invalidArgAbs()
        match a with
        | D(a) -> D(abs a)
        | DD(ap, at, ai) -> DD(abs ap, at * float (sign (float ap)), ai)
    static member Floor (a:D) =
        if isInteger (float a) then invalidArgFloor()
        match a with
        | D(a) -> D(floor a)
        | DD(ap, at, ai) -> DD(floor ap, D 0., ai)
    static member Ceiling (a:D) =
        if isInteger (float a) then invalidArgCeil()
        match a with
        | D(a) -> D(ceil a)
        | DD(ap, at, ai) -> DD(ceil ap, D 0., ai)
    static member Round (a:D) =
        if isHalfway (float a) then invalidArgRound()
        match a with
        | D(a) -> D(round a)
        | DD(ap, at, ai) -> DD(round ap, D 0., ai)


/// Tagger for generating incremental integers
type Tagger =
    val mutable LastTag : uint64
    new(t) = {LastTag = t}
    member t.Next() = t.LastTag <- t.LastTag + 1UL; t.LastTag

/// Global tagger for D operations
type GlobalTagger() =
    static let T = new Tagger(0UL)
    static member Next = T.Next()
    static member Reset = T.LastTag <- 0UL


/// D operations module (automatically opened)
[<AutoOpen>]
module DOps =
    /// Make D, with primal value `p`
    let inline dual p =
        match box p with
        | :? float as p -> D(p)
        | :? int as p -> D(float p)
        | :? D as p -> p
    /// Make D, with tag `i`, primal value `p`, and tangent value `t`
    let inline dualIPT i p t =
        match box p with
        | :? float as p ->
            match box t with
            | :? float as t -> DD(D p, D t, i)
            | :? D as t -> DD(D p, t, i)
        | :? D as p ->
            match box t with
            | :? float as t -> DD(p, D t, i)
            | :? D as t ->
                match p, t with
                | D(_), D(_) -> DD(p, t, i)
                | D(_), DD(_,_,_) -> DD(p, t, i)
                | DD(_,_,_), D(_) -> DD(p, t, i)
                | DD(pp,_ ,pi), DD(tp,tt,ti) when pi < ti -> DD(DD(pp,D 0.,ti), DD(tp,tt,ti), i)
                | DD(pp,pt,pi), DD(tp,tt,ti) when pi = ti -> DD(DD(pp,pt,pi), DD(tp,tt,pi), i)
                | DD(pp,pt,pi), DD(tp,_ ,ti) when pi > ti -> DD(DD(pp,pt,pi), DD(tp,D 0.,pi), i)
    /// Make D, with primal value `p` and tangent value `t`. A new tag will be attached using the global tagger.
    let inline dualPT p t = dualIPT GlobalTagger.Next p t
    /// Make D, with primal tag `i` and primal value `p`, and tangent 1.
    let inline dualIP1 i p =
        match box p with
        | :? float as p -> DD(D p, D 1., i)
        | :? D as p ->
            match p with
            | D(_) -> DD(p, D 1., i)
            | DD(_, _, pi) -> DD(p, DD(D 1., D 0., pi), i)
    /// Make D, with primal value `p` and tangent 1. A new tag will be attached using the global tagger.
    let inline dualP1 p = dualIP1 GlobalTagger.Next p
    /// Get the primal value of `d`
    let inline primal (d:D) =
        match d with
        | D(_) -> d
        | DD(p,_,_) -> p
    /// Get the primal value of `d` with the given tag `i`
    let inline primalI i (d:D) =
        match d with
        | D(_) -> d
        | DD(dp,_,di) when i = di -> dp
        | DD(_,_,di) when i <> di -> d
    /// Get the tangent value of `d`
    let inline tangent (d:D) =
        match d with
        | D(_) -> D(0.)
        | DD(_,t,_) -> t
    /// Get the tangent value of `d` with the given tag `i`
    let inline tangentI i (d:D) =
        match d with
        | D(_) -> D(0.)
        | DD(_,dt,di) when i = di -> dt
        | DD(_,_,di) when i <> di -> D(0.)
    /// Get the primal and the first gradient component of `d`, as a tuple
    let inline tuple (d:D) =
        match d with
        | D(_) -> (d, D 0.)
        | DD(p,t,_) -> (p, t)


[<AutoOpen>]
module ForwardOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        x |> dualP1 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        x |> dualP1 |> f |> tangent

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
    let diffn n f x =
        if n < 0 then invalidArg "" "Order of differentiation cannot be negative."
        elif n = 0 then x |> dual |> f
        else
            let rec d n f =
                match n with
                | 1 -> f
                | _ -> d (n - 1) (diff f)
            x |> dualP1 |> (d n f) |> tangent
    
    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let diffn' n f x =
        (diffn 0 f x, diffn n f x)

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (dualIPT i) x v |> f |> tuple

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
        let i = GlobalTagger.Next
        let a = Array.init x.Length (fun j ->
                                        let xd = standardBasis x.Length j |> Array.map2 (dualIPT i) x
                                        fVStoSS j f xd
                                        |> diff
                                        <| xd.[j])
        (x |> Array.map dual |> f, Array.sumBy tangent a)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (dualIPT i) x v |> f |> Array.map tuple |> Array.unzip

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
        (x |> Array.map dual |> f, g, h)

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        jacobian (grad f) x

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f x =
        (x |> Array.map dual |> f, hessian f x)

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


module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:D->D) x = ForwardOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:D->D) x = ForwardOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' (f:D->D) x = ForwardOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 (f:D->D) x = ForwardOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' (f:D->D) x = ForwardOps.diff2'' f x
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' (n:int) (f:D->D) x = ForwardOps.diffn' n f x
    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn (n:int) (f:D->D) x = ForwardOps.diffn n f x
    /// Original value and directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' (f:Vector<D>->D) x v = ForwardOps.gradv' (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Directional derivative of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:Vector<D>->D) x v = ForwardOps.gradv (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<D>->D) x = ForwardOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<D>->D) x = ForwardOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<D>->D) x = ForwardOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point x
    let inline laplacian (f:Vector<D>->D) x = ForwardOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<D>->Vector<_>) x = ForwardOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<D>->Vector<_>) x = ForwardOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<D>->Vector<_>) x = ForwardOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<D>->Vector<_>) x = ForwardOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' (f:Vector<D>->Vector<D>) x v = ForwardOps.jacobianv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:Vector<D>->Vector<D>) x v = ForwardOps.jacobianv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<D>->D) x = ForwardOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<D>->D) x = ForwardOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2D b)
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<D>->D) x = ForwardOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2D c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<D>->D) x = ForwardOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<D>->Vector<D>) x = ForwardOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<D>->Vector<D>) x = ForwardOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<D>->Vector<D>) x = ForwardOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<D>->Vector<D>) x = ForwardOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<D>->Vector<D>) x = ForwardOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<D>->Vector<D>) x = ForwardOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)