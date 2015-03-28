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

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// Dual numeric type with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]
type D =
    | Df of float
    | D of D * D * uint64
    static member op_Explicit(d:D) =
        match d with
        | Df(a) -> a
        | D(ap, _, _) -> (float) ap
    static member DivideByInt(d:D, i:int) =
        match d with
        | Df(a) -> Df(a / float i)
        | D(ap, at, ai) -> D(ap/ float i, at / float i, ai)
    static member Zero = Df 0.
    static member One = Df 1.
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
        | Df(a) -> hash [| a |]
        | D(ap, at, ai) -> hash [|ap; at; ai|]
        // D - D binary operations
    static member (+) (a:D, b:D) =
        match a, b with
        | Df(a),         Df(b)         -> Df(a + b)
        | Df(a),         D(bp, bt, bi) -> D(a + bp, bt, bi)
        | D(ap, at, ai), Df(b)         -> D(ap + b, at, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai < bi -> D(a + bp, bt, bi)
        | D(ap, at, ai), D(bp, bt, bi) when ai = bi -> D(ap + bp, at + bt, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai > bi -> D(ap + b, at, ai)
    static member (-) (a:D, b:D) =
        match a, b with
        | Df(a),         Df(b)         -> Df(a - b)
        | Df(a),         D(bp, bt, bi) -> D(a - bp, -bt, bi)
        | D(ap, at, ai), Df(b)         -> D(ap - b, at, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai < bi -> D(a - bp, -bt, bi)
        | D(ap, at, ai), D(bp, bt, bi) when ai = bi -> D(ap - bp, at - bt, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai > bi -> D(ap - b, at, ai)
    static member (*) (a:D, b:D) =
        match a, b with
        | Df(a),         Df(b)         -> Df(a * b)
        | Df(a),         D(bp, bt, bi) -> D(a * bp, a * bt, bi)
        | D(ap, at, ai), Df(b)         -> D(ap * b, at * b, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai < bi -> D(a * bp, bt * a, bi)
        | D(ap, at, ai), D(bp, bt, bi) when ai = bi -> D(ap * bp, at * bp + bt * ap, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai > bi -> D(ap * b, at * b, ai)
    static member (/) (a:D, b:D) =
        match a, b with
        | Df(a),         Df(b)         -> Df(a / b)
        | Df(a),         D(bp, bt, bi) -> D(a / bp, -(bt * a) / (bp * bp), bi)
        | D(ap, at, ai), Df(b)         -> D(ap / b, at / b, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai < bi -> D(a / bp, -(bt * a) / (bp * bp), bi)
        | D(ap, at, ai), D(bp, bt, bi) when ai = bi -> D(ap / bp, (at * bp - bt * ap) / (bp * bp), ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai > bi -> D(ap / b, at / b, ai)
    static member Pow (a:D, b:D) =
        match a, b with
        | Df(a),         Df(b)         -> Df(a ** b)
        | Df(a),         D(bp, bt, bi) -> let apb = D.Pow(a, bp) in D(apb, apb * (log a) * bt, bi)
        | D(ap, at, ai), Df(b)         -> D(ap ** b, b * (ap ** (b - 1.)) * at, ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai < bi -> let apb = a ** bp in D(apb, apb * (log a) * bt, bi)
        | D(ap, at, ai), D(bp, bt, bi) when ai = bi -> let apb = ap ** bp in D(apb, apb * ((bp * at / ap) + ((log ap) * bt)), ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai > bi -> let apb = ap ** b in D(apb, apb * (b * at / ap), ai)
    static member Atan2 (a:D, b:D) =
        match a, b with
        | Df(a),         Df(b)         -> Df(atan2 a b)
        | Df(a),         D(bp, bt, bi) -> D(D.Atan2(a, bp), -(a * bt) / (a * a + bp * bp), bi)
        | D(ap, at, ai), Df(b)         -> D(D.Atan2(ap, b), (b * at) / (ap * ap + b * b), ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai < bi -> D(atan2 a bp, -(a * bt) / (a * a + bp * bp), bi)
        | D(ap, at, ai), D(bp, bt, bi) when ai = bi -> D(atan2 ap bp, (at * bp - ap * bt) / (ap * ap + bp * bp), ai)
        | D(ap, at, ai), D(bp, bt, bi) when ai > bi -> D(atan2 ap b, (at * b) / (ap * ap + b * b), ai)
    // D - float binary operations
    static member (+) (a:D, b:float) =
        match a with
        | Df(a) -> Df(a + b)
        | D(ap, at, ai) -> D(ap + b, at, ai)
    static member (-) (a:D, b:float) =
        match a with
        | Df(a) -> Df(a - b)
        | D(ap, at, ai) -> D(ap - b, at, ai)
    static member (*) (a:D, b:float) =
        match a with
        | Df(a) -> Df(a * b)
        | D(ap, at, ai) -> D(ap * b, at * b, ai)
    static member (/) (a:D, b:float) =
        match a with
        | Df(a) -> Df(a / b)
        | D(ap, at, ai) -> D(ap / b, at / b, ai)
    static member Pow (a:D, b:float) =
        match a with
        | Df(a) -> Df(a ** b)
        | D(ap, at, ai) -> D(ap ** b, b * (ap ** (b - 1.)) * at, ai)
    static member Atan2 (a:D, b:float) =
        match a with
        | Df(a) -> Df(atan2 a b)
        | D(ap, at, ai) -> D(D.Atan2(ap, b), (b * at) / (b * b + ap * ap), ai)
    // float - D binary operations
    static member (+) (a:float, b:D) =
        match b with
        | Df(b) -> Df(a + b)
        | D(bp, bt, bi) -> D(a + bp, bt, bi)
    static member (-) (a:float, b:D) =
        match b with
        | Df(b) -> Df(a - b)
        | D(bp, bt, bi) -> D(a - bp, -bt, bi)
    static member (*) (a:float, b:D) =
        match b with
        | Df(b) -> Df(a * b)
        | D(bp, bt, bi) -> D(a * bp, a * bt, bi)
    static member (/) (a:float, b:D) =
        match b with
        | Df(b) -> Df(a / b)
        | D(bp, bt, bi) -> D(a / bp, -a * bt / (bp * bp), bi)
    static member Pow (a:float, b:D) =
        match b with
        | Df(b) -> Df(a ** b)
        | D(bp, bt, bi) -> let apb = D.Pow(a, bp) in D(apb, apb * (log a) * bt , bi)
    static member Atan2 (a:float, b:D) =
        match b with
        | Df(b) -> Df(atan2 a b)
        | D(bp, bt, bi) -> D(D.Atan2(a, bp), -(a * bt) / (a * a + bp * bp), bi)
    // D unary operations
    static member Log (a:D) =
        if (float a) <= 0. then invalidArgLog()
        match a with
        | Df(a) -> Df(log a)
        | D(ap, at, ai) -> D(log ap, at / ap, ai)
    static member Log10 (a:D) =
        if (float a) <= 0. then invalidArgLog10()
        match a with
        | Df(a) -> Df(log10 a)
        | D(ap, at, ai) -> D(log10 ap, at / (ap * log10val), ai)
    static member Exp (a:D) =
        match a with
        | Df(a) -> Df(exp a)
        | D(ap, at, ai) -> let expa = exp ap in D(expa, at * expa, ai)
    static member Sin (a:D) =
        match a with
        | Df(a) -> Df(sin a)
        | D(ap, at, ai) -> D(sin ap, at * cos ap, ai)
    static member Cos (a:D) =
        match a with
        | Df(a) -> Df(cos a)
        | D(ap, at, ai) -> D(cos ap, -at * sin ap, ai)
    static member Tan (a:D) =
        match a with
        | Df(a) -> 
            if cos a = 0. then invalidArgTan()
            Df(tan a)
        | D(ap, at, ai) ->
            let cosa = cos ap
            if (float cosa) = 0. then invalidArgTan()
            D(tan ap, at / (cosa * cosa), ai)
    static member (~-) (a:D) =
        match a with
        | Df(a) -> Df(-a)
        | D(ap, at, ai) -> D(-ap, -at, ai)
    static member Sqrt (a:D) =
        if (float a) <= 0. then invalidArgSqrt()
        match a with
        | Df(a) -> Df(sqrt a)
        | D(ap, at, ai) -> let sqrta = sqrt ap in D(sqrta, at / (2. * sqrta), ai)
    static member Sinh (a:D) =
        match a with
        | Df(a) -> Df(sinh a)
        | D(ap, at, ai) -> D(sinh ap, at * cosh ap, ai)
    static member Cosh (a:D) =
        match a with
        | Df(a) -> Df(cosh a)
        | D(ap, at, ai) -> D(cosh ap, at * sinh ap, ai)
    static member Tanh (a:D) =
        match a with
        | Df(a) -> Df(tanh a)
        | D(ap, at, ai) -> let cosha = cosh ap in D(tanh ap, at / (cosha * cosha), ai)
    static member Asin (a:D) =
        if abs (float a) >= 1. then invalidArgAsin()
        match a with
        | Df(a) -> Df(asin a)
        | D(ap, at, ai) -> D(asin ap, at / sqrt (1. - ap * ap), ai)
    static member Acos (a:D) =
        if abs (float a) >= 1. then invalidArgAcos()
        match a with
        | Df(a) -> Df(acos a)
        | D(ap, at, ai) -> D(acos ap, -at / sqrt (1. - ap * ap), ai)
    static member Atan (a:D) =
        match a with
        | Df(a) -> Df(atan a)
        | D(ap, at, ai) -> D(atan ap, at / (1. + ap * ap), ai)
    static member Abs (a:D) =
        if (float a) = 0. then invalidArgAbs()
        match a with
        | Df(a) -> Df(abs a)
        | D(ap, at, ai) -> D(abs ap, at * float (sign (float ap)), ai)
    static member Floor (a:D) =
        if isInteger (float a) then invalidArgFloor()
        match a with
        | Df(a) -> Df(floor a)
        | D(ap, at, ai) -> D(floor ap, Df 0., ai)
    static member Ceiling (a:D) =
        if isInteger (float a) then invalidArgCeil()
        match a with
        | Df(a) -> Df(ceil a)
        | D(ap, at, ai) -> D(ceil ap, Df 0., ai)
    static member Round (a:D) =
        if isHalfway (float a) then invalidArgRound()
        match a with
        | Df(a) -> Df(round a)
        | D(ap, at, ai) -> D(round ap, Df 0., ai)


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
        | :? float as p -> Df(p)
        | :? int as p -> Df(float p)
        | :? D as p -> p
    /// Make D, with tag `i`, primal value `p`, and tangent value `t`
    let inline dualIPT i p t =
        match box p with
        | :? float as p ->
            match box t with
            | :? float as t -> D(Df p, Df t, i)
            | :? D as t -> D(Df p, t, i)
        | :? D as p ->
            match box t with
            | :? float as t -> D(p, Df t, i)
            | :? D as t ->
                match p, t with
                | Df(_), Df(_) -> D(p, t, i)
                | Df(_), D(_,_,_) -> D(p, t, i)
                | D(_,_,_), Df(_) -> D(p, t, i)
                | D(pp,pt,pi), D(tp,tt,ti) when pi < ti -> D(D(pp,Df 0.,ti), D(tp,tt,ti), i)
                | D(pp,pt,pi), D(tp,tt,ti) when pi = ti -> D(D(pp,pt,pi), D(tp,tt,pi), i)
                | D(pp,pt,pi), D(tp,tt,ti) when pi > ti -> D(D(pp,pt,pi), D(tp,Df 0.,pi), i)
    /// Make D, with primal value `p` and tangent value `t`. A new tag will be attached using the global tagger.
    let inline dualPT p t = dualIPT GlobalTagger.Next p t
    /// Make D, with primal tag `i` and primal value `p`, and tangent 1.
    let inline dualIP1 i p =
        match box p with
        | :? float as p -> D(Df p, Df 1., i)
        | :? D as p ->
            match p with
            | Df(_) -> D(p, Df 1., i)
            | D(_, _, i) -> D(p, D(Df 1., Df 0., i), i)
    /// Make D, with primal value `p` and tangent 1. A new tag will be attached using the global tagger.
    let inline dualP1 p = dualIP1 GlobalTagger.Next p
    /// Get the primal value of a D
    let inline primal (d:D) =
        match d with
        | Df(_) -> d
        | D(p,_,_) -> p
    /// Get the tangent value of a D
    let inline tangent (d:D) =
        match d with
        | Df(_) -> Df(0.)
        | D(_,t,_) -> t
    /// Get the primal and the first gradient component of a D, as a tuple
    let inline tuple (d:D) =
        match d with
        | Df(_) -> (d, Df 0.)
        | D(p,t,_) -> (p, t)


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