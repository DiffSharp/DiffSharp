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

module DiffSharp.AD.Nested.Forward

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

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

type GlobalTagger() =
    static let T = new Tagger(0UL)
    static member Next = T.Next()
    static member Reset = T.LastTag <- 0UL


[<AutoOpen>]
module DOps =
    let inline dual p = Df(float p)
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
    let inline dualPT p t = dualIPT GlobalTagger.Next p t
    let inline dualP1 p =
        match box p with
        | :? float as p -> D(Df p, Df 1., GlobalTagger.Next)
        | :? D as p ->
            match p with
            | Df(_) -> D(p, Df 1., GlobalTagger.Next)
            | D(_, _, i) -> D(p, D(Df 1., Df 0., i), GlobalTagger.Next)
    let inline primal (d:D) =
        match d with
        | Df(_) -> d
        | D(p,_,_) -> p
    let inline tangent (d:D) =
        match d with
        | Df(_) -> Df(0.)
        | D(_,t,_) -> t
    let inline tuple (d:D) =
        match d with
        | Df(_) -> (d, Df 0.)
        | D(p,t,_) -> (p, t)

[<AutoOpen>]
module ForwardOps =
    
    let inline diff' f x =
        x |> dualP1 |> f |> tuple

    let inline diff f x =
        x |> dualP1 |> f |> tangent

    let inline diff2 f x =
        diff (diff f) x

    let inline diff2'' f x =
        let v, d = diff' f x
        let d2 = diff2 f x
        (v, d, d2)

    let inline diff2' f x =
        diff2'' f x |> fsttrd

    let inline gradv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (dualIPT i) x v |> f |> tuple

    let inline gradv f x v=
        gradv' f x v |> snd

    let inline grad' f (x:_[]) =
        let a = Array.init x.Length (fun i -> gradv' f x (standardBasis x.Length i))
        (fst a.[0], Array.map snd a)

    let inline grad f x =
        grad' f x |> snd

    let inline jacobianv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (dualIPT i) x v |> f |> Array.map tuple |> Array.unzip

    let inline jacobianv f x v =
        jacobianv' f x v |> snd

    let inline jacobianT' f (x:_[]) =
        let a = Array.init x.Length (fun i -> jacobianv' f x (standardBasis x.Length i))
        (fst a.[0], array2D (Array.map snd a))

    let inline jacobianT f x =
        jacobianT' f x |> snd

    let inline jacobian' f x =
        jacobianT' f x |> fun (r, j) -> (r, transpose j)

    let inline jacobian f x =
        jacobian' f x |> snd

    let inline curl' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurl()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    let inline curl f x =
        curl' f x |> snd

    let inline div' f x =
        let v, j = jacobianT' f x
        if Array2D.length1 j <> Array2D.length2 j then invalidArgDiv()
        v, trace j

    let inline div f x =
        div' f x |> snd

    let inline curldiv' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurlDiv()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    let inline curldiv f x =
        curldiv' f x |> sndtrd


module Vector =
    let inline diff' (f:D->D) x = ForwardOps.diff' f x
    let inline diff (f:D->D) x = ForwardOps.diff f x
    let inline diff2' (f:D->D) x = ForwardOps.diff2' f x
    let inline diff2 (f:D->D) x = ForwardOps.diff2 f x
    let inline diff2'' (f:D->D) x = ForwardOps.diff2'' f x
    let inline gradv' (f:Vector<D>->D) x v = ForwardOps.gradv' (vector >> f) (Vector.toArray x) (Vector.toArray v)
    let inline gradv (f:Vector<D>->D) x v = ForwardOps.gradv (vector >> f) (Vector.toArray x) (Vector.toArray v)
    let inline grad' (f:Vector<D>->D) x = ForwardOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    let inline grad (f:Vector<D>->D) x = ForwardOps.grad (vector >> f) (Vector.toArray x) |> vector
    let inline jacobianT' (f:Vector<D>->Vector<_>) x = ForwardOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    let inline jacobianT (f:Vector<D>->Vector<_>) x = ForwardOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    let inline jacobian' (f:Vector<D>->Vector<_>) x = ForwardOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    let inline jacobian (f:Vector<D>->Vector<_>) x = ForwardOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    let inline jacobianv' (f:Vector<D>->Vector<D>) x v = ForwardOps.jacobianv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    let inline jacobianv (f:Vector<D>->Vector<D>) x v = ForwardOps.jacobianv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    let inline curl' (f:Vector<D>->Vector<D>) x = ForwardOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    let inline curl (f:Vector<D>->Vector<D>) x = ForwardOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    let inline div' (f:Vector<D>->Vector<D>) x = ForwardOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    let inline div (f:Vector<D>->Vector<D>) x = ForwardOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    let inline curldiv' (f:Vector<D>->Vector<D>) x = ForwardOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    let inline curldiv (f:Vector<D>->Vector<D>) x = ForwardOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)