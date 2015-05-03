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

/// Non-nested 1st-order reverse mode AD
namespace DiffSharp.AD.Specialized.Reverse1

open DiffSharp.Util
open FsAlg.Generic


/// Numeric type keeping adjoint values and traces, with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]    
type D =
    | D of float * (float ref) * TraceOp * (uint32 ref) // Primal, adjoint, parent operation, fan-out counter
    member d.P = let (D(p,_,_,_)) = d in p
    member d.O = let (D(_,_,o,_)) = d in o
    member d.A
        with get() = let (D(_,a,_,_)) = d in !a
        and set(v) = let (D(_,a,_,_)) = d in a := v
    member d.F
        with get() = let (D(_,_,_,f)) = d in !f
        and set(v) = let (D(_,_,_,f)) = d in f := v
    static member op_Explicit(D(p,_,_,_)):float = p
    static member op_Explicit(D(p,_,_,_)):int = int p
    static member DivideByInt(d:D, i:int) = d / float i
    static member Zero = D(0., ref 0., Noop, ref 0u)
    static member One = D(1., ref 0., Noop, ref 0u)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> compare ((float) d) ((float) d2)
            | _ -> invalidArg "" "Cannot compare this D with another type of object."
    override d.Equals(other) =
        match other with
        | :? D as d2 -> compare ((float) d) ((float) d2) = 0
        | _ -> false
    override d.GetHashCode() = let (D(p,a,_,_)) = d in hash [|p; a|]
    // D - D binary operations
    static member (+) (a:D, b:D) = D(a.P + b.P, ref 0., Add(a, b), ref 0u)
    static member (-) (a:D, b:D) = D(a.P - b.P, ref 0., Sub(a, b), ref 0u)
    static member (*) (a:D, b:D) = D(a.P * b.P, ref 0., Mul(a, b), ref 0u)
    static member (/) (a:D, b:D) = D(a.P / b.P, ref 0., Div(a, b), ref 0u)
    static member Pow (a:D, b:D) = D(a.P ** b.P, ref 0., Pow(a, b), ref 0u)
    static member Atan2 (a:D, b:D) = D(atan2 a.P b.P, ref 0., Atan2(a, b), ref 0u)
    // D - float binary operations
    static member (+) (a:D, b:float) = D(a.P + b, ref 0., AddCons(a), ref 0u)
    static member (-) (a:D, b:float) = D(a.P - b, ref 0., SubDCons(a), ref 0u)
    static member (*) (a:D, b:float) = D(a.P * b, ref 0., MulCons(a, b), ref 0u)
    static member (/) (a:D, b:float) = D(a.P / b, ref 0., DivDCons(a, b), ref 0u)
    static member Pow (a:D, b:float) = D(a.P ** b, ref 0., PowDCons(a, b), ref 0u)
    static member Atan2 (a:D, b:float) = D(atan2 a.P b, ref 0., Atan2DCons(a, b), ref 0u)
    // float - D binary operations
    static member (+) (a:float, b:D) = D(a + b.P, ref 0., AddCons(b), ref 0u)
    static member (-) (a:float, b:D) = D(a - b.P, ref 0., SubConsD(b), ref 0u)
    static member (*) (a:float, b:D) = D(a * b.P, ref 0., MulCons(b, a), ref 0u)
    static member (/) (a:float, b:D) = D(a / b.P, ref 0., DivConsD(b, a), ref 0u)
    static member Pow (a:float, b:D) = D(a ** b.P, ref 0., PowConsD(b, a), ref 0u)
    static member Atan2 (a:float, b:D) = D(atan2 a b.P, ref 0., Atan2ConsD(b, a), ref 0u)
    // D - int binary operations
    static member (+) (a:D, b:int) = a + (float b)
    static member (-) (a:D, b:int) = a - (float b)
    static member (*) (a:D, b:int) = a * (float b)
    static member (/) (a:D, b:int) = a / (float b)
    static member Pow (a:D, b:int) = D.Pow(a, (float b))
    static member Atan2 (a:D, b:int) = D.Atan2(a, (float b))
    // int - D binary operations
    static member (+) (a:int, b:D) = (float a) + b
    static member (-) (a:int, b:D) = (float a) - b
    static member (*) (a:int, b:D) = (float a) * b
    static member (/) (a:int, b:D) = (float a) / b
    static member Pow (a:int, b:D) = D.Pow((float a), b)
    static member Atan2 (a:int, b:D) = D.Atan2((float a), b)
    // D unary operations
    static member Log (a:D) =
        if a.P <= 0. then invalidArgLog()
        D(log a.P, ref 0., Log(a), ref 0u)
    static member Log10 (a:D) =
        if a.P <= 0. then invalidArgLog10()
        D(log10 a.P, ref 0., Log10(a), ref 0u)
    static member Exp (a:D) = D(exp a.P, ref 0., Exp(a), ref 0u)
    static member Sin (a:D) = D(sin a.P, ref 0., Sin(a), ref 0u)
    static member Cos (a:D) = D(cos a.P, ref 0., Cos(a), ref 0u)
    static member Tan (a:D) =
        if cos a.P = 0. then invalidArgTan()
        D(tan a.P, ref 0., Tan(a), ref 0u)
    static member (~-) (a:D) = D(-a.P, ref 0., Neg(a), ref 0u)
    static member Sqrt (a:D) =
        if a.P <= 0. then invalidArgSqrt()
        D(sqrt a.P, ref 0., Sqrt(a), ref 0u)
    static member Sinh (a:D) = D(sinh a.P, ref 0., Sinh(a), ref 0u)
    static member Cosh (a:D) = D(cosh a.P, ref 0., Cosh(a), ref 0u)
    static member Tanh (a:D) = D(tanh a.P, ref 0., Tanh(a), ref 0u)
    static member Asin (a:D) =
        if abs a.P >= 1. then invalidArgAsin()
        D(asin a.P, ref 0., Asin(a), ref 0u)
    static member Acos (a:D) =
        if abs a.P >= 1. then invalidArgAcos()
        D(acos a.P, ref 0., Acos(a), ref 0u)
    static member Atan (a:D) = D(atan a.P, ref 0., Atan(a), ref 0u)
    static member Abs (a:D) =
        if a.P = 0. then invalidArgAbs()
        D(abs a.P, ref 0., Abs(a), ref 0u)
    static member Floor (a:D) =
        if isInteger a.P then invalidArgFloor()
        D(floor a.P, ref 0., Floor(a), ref 0u)
    static member Ceiling (a:D) =
        if isInteger a.P then invalidArgCeil()
        D(ceil a.P, ref 0., Ceil(a), ref 0u)
    static member Round (a:D) =
        if isHalfway a.P then invalidArgRound()
        D(round a.P, ref 0., Round(a), ref 0u)

/// Operation types recorded in the evaluation trace
and TraceOp =
    | Add        of D * D
    | AddCons    of D
    | Sub        of D * D
    | SubDCons   of D
    | SubConsD   of D
    | Mul        of D * D
    | MulCons    of D * float
    | Div        of D * D
    | DivDCons   of D * float
    | DivConsD   of D * float
    | Pow        of D * D
    | PowDCons   of D * float
    | PowConsD   of D * float
    | Atan2      of D * D
    | Atan2DCons of D * float
    | Atan2ConsD of D * float
    | Log        of D
    | Log10      of D
    | Exp        of D
    | Sin        of D
    | Cos        of D
    | Tan        of D
    | Neg        of D
    | Sqrt       of D
    | Sinh       of D
    | Cosh       of D
    | Tanh       of D
    | Asin       of D
    | Acos       of D
    | Atan       of D
    | Abs        of D
    | Floor      of D
    | Ceil       of D
    | Round      of D
    | Noop


/// D operations module (automatically opened)
[<AutoOpen>]
module DOps =
    /// Make D with primal value `p`
    let inline makeD p = D(p, ref 0., Noop, ref 0u) 
    /// Get the adjoint value of `d`
    let inline adjoint (d:D) = d.A
    /// Get the primal value of `d`
    let inline primal (d:D) = d.P
    /// Pushes the adjoint `v` backwards through the evaluation trace of `d`
    let rec reversePush (v:float) (d:D) =
        d.A <- d.A + v
        d.F <- d.F - 1u
        if d.F = 0u then
            match d.O with
            | Add(a, b)           -> reversePush d.A a; reversePush d.A b
            | AddCons(a)          -> reversePush d.A a
            | Sub(a, b)           -> reversePush d.A a; reversePush -d.A b
            | SubDCons(a)         -> reversePush d.A a
            | SubConsD(a)         -> reversePush -d.A a
            | Mul(a, b)           -> reversePush (d.A * b.P) a; reversePush (d.A * a.P) b
            | MulCons(a, cons)    -> reversePush (d.A * cons) a
            | Div(a, b)           -> reversePush (d.A / b.P) a; reversePush (d.A * (-a.P / (b.P * b.P))) b
            | DivDCons(a, cons)   -> reversePush (d.A / cons) a
            | DivConsD(a, cons)   -> reversePush (d.A * (-cons / (a.P * a.P))) a
            | Pow(a, b)           -> reversePush (d.A * (a.P ** (b.P - 1.)) * b.P) a; reversePush (d.A * (a.P ** b.P) * log a.P) b
            | PowDCons(a, cons)   -> reversePush (d.A * (a.P ** (cons - 1.)) * cons) a
            | PowConsD(a, cons)   -> reversePush (d.A * (cons ** a.P) * log cons) a
            | Atan2(a, b)         -> let denom = a.P * a.P + b.P * b.P in reversePush (d.A * b.P / denom) a; reversePush (d.A * (-a.P) / denom) b
            | Atan2DCons(a, cons) -> reversePush (d.A * cons / (a.P * a.P + cons * cons)) a
            | Atan2ConsD(a, cons) -> reversePush (d.A * (-cons) / (cons * cons + a.P * a.P)) a
            | Log(a)              -> reversePush (d.A / a.P) a
            | Log10(a)            -> reversePush (d.A / (a.P * log10val)) a
            | Exp(a)              -> reversePush (d.A * d.P) a // d.P = exp a.P
            | Sin(a)              -> reversePush (d.A * cos a.P) a
            | Cos(a)              -> reversePush (d.A * (-sin a.P)) a
            | Tan(a)              -> let seca = 1. / cos a.P in reversePush (d.A * seca * seca) a
            | Neg(a)              -> reversePush -d.A a
            | Sqrt(a)             -> reversePush (d.A / (2. * d.P)) a // d.P = sqrt a.P
            | Sinh(a)             -> reversePush (d.A * cosh a.P) a
            | Cosh(a)             -> reversePush (d.A * sinh a.P) a
            | Tanh(a)             -> let secha = 1. / cosh a.P in reversePush (d.A * secha * secha) a
            | Asin(a)             -> reversePush (d.A / sqrt (1. - a.P * a.P)) a
            | Acos(a)             -> reversePush (-d.A / sqrt (1. - a.P * a.P)) a
            | Atan(a)             -> reversePush (d.A / (1. + a.P * a.P)) a
            | Abs(a)              -> reversePush (d.A * float (sign (float a.P))) a
            | Floor(_)            -> ()
            | Ceil(_)             -> ()
            | Round(_)            -> ()
            | Noop                -> ()
    /// Resets the adjoints of all the values in the evaluation trace of `d`
    let rec reverseReset (d:D) =
        d.A <- 0.
        d.F <- d.F + 1u
        if d.F = 1u then
            match d.O with
            | Add(a, b)           -> reverseReset a; reverseReset b
            | AddCons(a)          -> reverseReset a
            | Sub(a, b)           -> reverseReset a; reverseReset b
            | SubDCons(a)         -> reverseReset a
            | SubConsD(a)         -> reverseReset a
            | Mul(a, b)           -> reverseReset a; reverseReset b
            | MulCons(a, _)       -> reverseReset a
            | Div(a, b)           -> reverseReset a; reverseReset b
            | DivDCons(a, _)      -> reverseReset a
            | DivConsD(a, _)      -> reverseReset a
            | Pow(a, b)           -> reverseReset a; reverseReset b
            | PowDCons(a, _)      -> reverseReset a
            | PowConsD(a, _)      -> reverseReset a
            | Atan2(a, b)         -> reverseReset a; reverseReset b
            | Atan2DCons(a, _)    -> reverseReset a
            | Atan2ConsD(a, _)    -> reverseReset a
            | Log(a)              -> reverseReset a
            | Log10(a)            -> reverseReset a
            | Exp(a)              -> reverseReset a
            | Sin(a)              -> reverseReset a
            | Cos(a)              -> reverseReset a
            | Tan(a)              -> reverseReset a
            | Neg(a)              -> reverseReset a
            | Sqrt(a)             -> reverseReset a
            | Sinh(a)             -> reverseReset a
            | Cosh(a)             -> reverseReset a
            | Tanh(a)             -> reverseReset a
            | Asin(a)             -> reverseReset a
            | Acos(a)             -> reverseReset a
            | Atan(a)             -> reverseReset a
            | Abs(a)              -> reverseReset a
            | Floor(a)            -> reverseReset a
            | Ceil(a)             -> reverseReset a
            | Round(a)            -> reverseReset a
            | Noop                -> ()

    /// Propagates the adjoint `v` backwards through the evaluation trace of `d`. The adjoints in the trace are reset before the push.
    let reverseProp (v:float) (d:D) =
        d |> reverseReset
        d |> reversePush v


/// Reverse differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        let xa = x |> makeD
        let z:D = f xa
        z |> reverseReset
        z |> reversePush 1.
        (primal z, adjoint xa)

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        diff' f x |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`    
    let inline grad' f x =
        let xa = x |> Array.map makeD
        let z:D = f xa
        z |> reverseReset
        z |> reversePush 1.
        (primal z, Array.map adjoint xa)

    /// Gradient of a vector-to-scalar function `f`, at point `x`    
    let inline grad f x =
        grad' f x |> snd

    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' f x =
        let xa = x |> Array.map makeD
        let z:D[] = f xa
        let r1 = Array.map primal z
        let r2 =
            fun v ->
                Array.iter reverseReset z
                Array.iter2 reversePush v z
                Array.map adjoint xa
        (r1, r2)

    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv' f x v =
        let r1, r2 = jacobianTv'' f x
        (r1, r2 v)

    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv f x v =
        jacobianTv' f x v |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        let r1, r2 = jacobianTv'' f x
        let a = Array.init r1.Length (fun j -> r2 (standardBasis r1.Length j))
        (r1, array2D a)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        jacobian' f x |> fun (r, j) -> (r, transpose j)

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

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
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<D>->D) x = DiffOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<D>->D) x = DiffOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<D>->Vector<D>) x = DiffOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<D>->Vector<D>) x = DiffOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv (f:Vector<D>->Vector<D>) x v = DiffOps.jacobianTv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv' (f:Vector<D>->Vector<D>) x v = DiffOps.jacobianTv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of the reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' (f:Vector<D>->Vector<D>) x = DiffOps.jacobianTv'' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Vector.toArray >> b >> vector)
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