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

/// Nested reverse mode AD module
module DiffSharp.AD.Nested.Reverse

open DiffSharp.Util.General
open FsAlg.Generic
open System.Collections.Generic


/// Numeric type keeping adjoint values and traces, with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]    
type D =
    | D of float // Primal
    | DR of D * (D ref) * Stack<Op> * uint64 // Primal, adjoint, trace, tag
    member d.P =
        match d with
        | D(_) -> d
        | DR(p,_,_,_) -> p
    member d.A
        with get() =
            match d with
            | D(_) -> D 0.
            | DR(_,a,_,_) -> !a
        and set(v) =
            match d with
            | D(_) -> ()
            | DR(_,a,_,_) -> a := v
    member d.AddA a =
        d.A <- d.A + a
    static member op_Explicit(d:D) =
        match d with
        | D(a) -> a
        | DR(ap,_,_,_) -> float ap
    static member DivideByInt(d:D, i:int) =
        match d with
        | D(a) -> D(a / float i)
        | DR(_,_,_,_) -> d / float i
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
        | DR(ap,_,atr,ai) -> hash [|ap; atr; ai|]
    // D - D binary operations
    static member (+) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp) -> D(ap + bp)
        | D(ap), DR(bp, _, btr, bi) -> let c = DR(ap + bp, ref (D 0.), btr, bi) in btr.Push(AddCons(b, c)); c
        | DR(ap, _, atr, ai), D(bp) -> let c = DR(ap + bp, ref (D 0.), atr, ai) in atr.Push(AddCons(a, c)); c
        | DR(_ , _, _ , ai), DR(bp, _, btr, bi) when ai < bi -> let c = DR(a + bp, ref (D 0.), btr, bi) in btr.Push(AddCons(b, c)); c
        | DR(ap, _, atr, ai), DR(bp, _, _ , bi) when ai = bi -> let c = DR(ap + bp, ref (D 0.), atr, ai) in atr.Push(Add(a, b, c)); c
        | DR(ap, _, atr, ai), DR(_ , _, _ , bi) when ai > bi -> let c = DR(ap + b, ref (D 0.), atr, ai) in atr.Push(AddCons(a, c)); c
    static member (-) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp) -> D(ap - bp)
        | D(ap), DR(bp, _, btr, bi) -> let c = DR(ap - bp, ref (D 0.), btr, bi) in btr.Push(SubConsD(b, c)); c
        | DR(ap, _, atr, ai), D(bp) -> let c = DR(ap - bp, ref (D 0.), atr, ai) in atr.Push(SubDCons(a, c)); c
        | DR(_ , _, _ , ai), DR(bp, _, btr, bi) when ai < bi -> let c = DR(a - bp, ref (D 0.), btr, bi) in btr.Push(SubConsD(b, c)); c
        | DR(ap, _, atr, ai), DR(bp, _, _ , bi) when ai = bi -> let c = DR(ap - bp, ref (D 0.), atr, ai) in atr.Push(Sub(a, b, c)); c
        | DR(ap, _, atr, ai), DR(_ , _, _ , bi) when ai > bi -> let c = DR(ap - b, ref (D 0.), atr, ai) in atr.Push(SubDCons(a, c)); c
    static member (*) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp) -> D(ap * bp)
        | D(ap), DR(bp, _, btr, bi) -> let c = DR(ap * bp, ref (D 0.), btr, bi) in btr.Push(MulCons(b, a, c)); c
        | DR(ap, _, atr, ai), D(bp) -> let c = DR(ap * bp, ref (D 0.), atr, ai) in atr.Push(MulCons(a, b, c)); c
        | DR(_ , _, _ , ai), DR(bp, _, btr, bi) when ai < bi -> let c = DR(a * bp, ref (D 0.), btr, bi) in btr.Push(MulCons(b, a, c)); c
        | DR(ap, _, atr, ai), DR(bp, _, _ , bi) when ai = bi -> let c = DR(ap * bp, ref (D 0.), atr, ai) in atr.Push(Mul(a, b, c)); c
        | DR(ap, _, atr, ai), DR(_ , _, _ , bi) when ai > bi -> let c = DR(ap * b, ref (D 0.), atr, ai) in atr.Push(MulCons(a, b, c)); c
    static member (/) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp) -> D(ap / bp)
        | D(ap), DR(bp, _, btr, bi) -> let c = DR(ap / bp, ref (D 0.), btr, bi) in btr.Push(DivConsD(a, b, c)); c
        | DR(ap, _, atr, ai), D(bp) -> let c = DR(ap / bp, ref (D 0.), atr, ai) in atr.Push(DivDCons(a, b, c)); c
        | DR(_ , _, _ , ai), DR(bp, _, btr, bi) when ai < bi -> let c = DR(a / bp, ref (D 0.), btr, bi) in btr.Push(DivConsD(a, b, c)); c
        | DR(ap, _, atr, ai), DR(bp, _, _ , bi) when ai = bi -> let c = DR(ap / bp, ref (D 0.), atr, ai) in atr.Push(Div(a, b, c)); c
        | DR(ap, _, atr, ai), DR(_ , _, _ , bi) when ai > bi -> let c = DR(ap / b, ref (D 0.), atr, ai) in atr.Push(DivDCons(a, b, c)); c
    static member Pow (a:D, b:D) =
        match a, b with
        | D(ap), D(bp) -> D(ap ** bp)
        | D(ap), DR(bp, _, btr, bi) -> let c = DR(D.Pow(ap, bp), ref (D 0.), btr, bi) in btr.Push(PowConsD(a, b, c)); c
        | DR(ap, _, atr, ai), D(bp) -> let c = DR(ap ** bp, ref (D 0.), atr, ai) in atr.Push(PowDCons(a, b, c)); c
        | DR(_ , _, _ , ai), DR(bp, _, btr, bi) when ai < bi -> let c = DR(a ** bp, ref (D 0.), btr, bi) in btr.Push(PowConsD(a, b, c)); c
        | DR(ap, _, atr, ai), DR(bp, _, _ , bi) when ai = bi -> let c = DR(ap ** bp, ref (D 0.), atr, ai) in atr.Push(Pow(a, b, c)); c
        | DR(ap, _, atr, ai), DR(_ , _, _ , bi) when ai > bi -> let c = DR(ap ** b, ref (D 0.), atr, ai) in atr.Push(PowDCons(a, b, c)); c
    static member Atan2 (a:D, b:D) =
        match a, b with
        | D(ap), D(bp) -> D(atan2 ap bp)
        | D(ap), DR(bp, _, btr, bi) -> let c = DR(D.Atan2(ap, bp), ref (D 0.), btr, bi) in btr.Push(Atan2ConsD(a, b, c)); c
        | DR(ap, _, atr, ai), D(bp) -> let c = DR(D.Atan2(ap, bp), ref (D 0.), atr, ai) in atr.Push(Atan2DCons(a, b, c)); c
        | DR(_ , _, _ , ai), DR(bp, _, btr, bi) when ai < bi -> let c = DR(atan2 a bp, ref (D 0.), btr, bi) in btr.Push(Atan2ConsD(a, b, c)); c
        | DR(ap, _, atr, ai), DR(bp, _, _ , bi) when ai = bi -> let c = DR(atan2 ap bp, ref (D 0.), atr, ai) in atr.Push(Atan2(a, b, c)); c
        | DR(ap, _, atr, ai), DR(_ , _, _ , bi) when ai > bi -> let c = DR(atan2 ap b, ref (D 0.), atr, ai) in atr.Push(Atan2DCons(a, b, c)); c
    // D - float binary operations
    static member (+) (a:D, b:float) =
        match a with
        | D(a) -> D(a + b)
        | DR(ap, _, atr, ai) -> let c = DR(ap + b, ref (D 0.), atr, ai) in atr.Push(AddCons(a, c)); c
    static member (-) (a:D, b:float) =
        match a with
        | D(a) -> D(a - b)
        | DR(ap, _, atr, ai) -> let c = DR(ap - b, ref (D 0.), atr, ai) in atr.Push(SubDCons(a, c)); c
    static member (*) (a:D, b:float) =
        match a with
        | D(a) -> D(a * b)
        | DR(ap, _, atr, ai) -> let c = DR(ap * b, ref (D 0.), atr, ai) in atr.Push(MulCons(a, D b, c)); c
    static member (/) (a:D, b:float) =
        match a with
        | D(a) -> D(a / b)
        | DR(ap, _, atr, ai) -> let c = DR(ap / b, ref (D 0.), atr, ai) in atr.Push(DivDCons(a, D b, c)); c
    static member Pow (a:D, b:float) =
        match a with
        | D(a) -> D(a ** b)
        | DR(ap, _, atr, ai) -> let c = DR(ap ** b, ref (D 0.), atr, ai) in atr.Push(PowDCons(a, D b, c)); c
    static member Atan2 (a:D, b:float) =
        match a with
        | D(a) -> D(atan2 a b)
        | DR(ap, _, atr, ai) -> let c = DR(D.Atan2(ap, b), ref (D 0.), atr, ai) in atr.Push(Atan2DCons(a, D b, c)); c
    // float - D binary operations
    static member (+) (a:float, b:D) =
        match b with
        | D(b) -> D(a + b)
        | DR(bp, _, btr, bi) -> let c = DR(a + bp, ref (D 0.), btr, bi) in btr.Push(AddCons(b, c)); c
    static member (-) (a:float, b:D) =
        match b with
        | D(b) -> D(a - b)
        | DR(bp, _, btr, bi) -> let c = DR(a - bp, ref (D 0.), btr, bi) in btr.Push(SubConsD(b, c)); c
    static member (*) (a:float, b:D) =
        match b with
        | D(b) -> D(a * b)
        | DR(bp, _, btr, bi) -> let c = DR(a * bp, ref (D 0.), btr, bi) in btr.Push(MulCons(b, D a, c)); c
    static member (/) (a:float, b:D) =
        match b with
        | D(b) -> D(a / b)
        | DR(bp, _, btr, bi) -> let c = DR(a * bp, ref (D 0.), btr, bi) in btr.Push(DivConsD(D a, b, c)); c
    static member Pow (a:float, b:D) =
        match b with
        | D(b) -> D(a ** b)
        | DR(bp, _, btr, bi) -> let c = DR(D.Pow(a, bp), ref (D 0.), btr, bi) in btr.Push(PowConsD(D a, b, c)); c
    static member Atan2 (a:float, b:D) =
        match b with
        | D(b) -> D(atan2 a b)
        | DR(bp, _, btr, bi) -> let c = DR(D.Atan2(a, bp), ref (D 0.), btr, bi) in btr.Push(Atan2ConsD(D a, b, c)); c
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
        | DR(ap, _, atr, ai) -> let c = DR(log ap, ref (D 0.), atr, ai) in atr.Push(Log(a, c)); c
    static member Log10 (a:D) =
        if (float a) <= 0. then invalidArgLog10()
        match a with
        | D(a) -> D(log10 a)
        | DR(ap, _, atr, ai) -> let c = DR(log10 ap, ref (D 0.), atr, ai) in atr.Push(Log10(a, c)); c
    static member Exp (a:D) =
        match a with
        | D(a) -> D(exp a)
        | DR(ap, _, atr, ai) -> let c = DR(exp ap, ref (D 0.), atr, ai) in atr.Push(Exp(a, c)); c
    static member Sin (a:D) =
        match a with
        | D(a) -> D(sin a)
        | DR(ap, _, atr, ai) -> let c = DR(sin ap, ref (D 0.), atr, ai) in atr.Push(Sin(a, c)); c
    static member Cos (a:D) =
        match a with
        | D(a) -> D(cos a)
        | DR(ap, _, atr, ai) -> let c = DR(cos ap, ref (D 0.), atr, ai) in atr.Push(Cos(a, c)); c
    static member Tan (a:D) =
        if (float (cos a)) = 0. then invalidArgTan()
        match a with
        | D(a) -> D(tan a)
        | DR(ap, _, atr, ai) -> let c = DR(tan ap, ref (D 0.), atr, ai) in atr.Push(Tan(a, c)); c
    static member (~-) (a:D) =
        match a with
        | D(a) -> D(-a)
        | DR(ap, _, atr, ai) -> let c = DR(-ap, ref (D 0.), atr, ai) in atr.Push(Neg(a, c)); c
    static member Sqrt (a:D) =
        if (float a) <= 0. then invalidArgSqrt()
        match a with
        | D(a) -> D(sqrt a)
        | DR(ap, _, atr, ai) -> let c = DR(sqrt ap, ref (D 0.), atr, ai) in atr.Push(Sqrt(a, c)); c
    static member Sinh (a:D) =
        match a with
        | D(a) -> D(sinh a)
        | DR(ap, _, atr, ai) -> let c = DR(sinh ap, ref (D 0.), atr, ai) in atr.Push(Sinh(a, c)); c
    static member Cosh (a:D) =
        match a with
        | D(a) -> D(cosh a)
        | DR(ap, _, atr, ai) -> let c = DR(cosh ap, ref (D 0.), atr, ai) in atr.Push(Cosh(a, c)); c
    static member Tanh (a:D) =
        match a with
        | D(a) -> D(tanh a)
        | DR(ap, _, atr, ai) -> let c = DR(tanh ap, ref (D 0.), atr, ai) in atr.Push(Cosh(a, c)); c
    static member Asin (a:D) =
        if abs (float a) >= 1. then invalidArgAsin()
        match a with
        | D(a) -> D(asin a)
        | DR(ap, _, atr, ai) -> let c = DR(asin ap, ref (D 0.), atr, ai) in atr.Push(Asin(a, c)); c
    static member Acos (a:D) =
        if abs (float a) >= 1. then invalidArgAcos()
        match a with
        | D(a) -> D(acos a)
        | DR(ap, _, atr, ai) -> let c = DR(acos ap, ref (D 0.), atr, ai) in atr.Push(Acos(a, c)); c
    static member Atan (a:D) =
        match a with
        | D(a) -> D(atan a)
        | DR(ap, _, atr, ai) -> let c = DR(atan ap, ref (D 0.), atr, ai) in atr.Push(Atan(a, c)); c
    static member Abs (a:D) =
        if float a = 0. then invalidArgAbs()
        match a with
        | D(a) -> D(abs a)
        | DR(ap, _, atr, ai) -> let c = DR(abs ap, ref (D 0.), atr, ai) in atr.Push(Abs(a, c)); c
    static member Floor (a:D) =
        if isInteger (float a) then invalidArgFloor()
        match a with
        | D(a) -> D(floor a)
        | DR(ap, _, atr, ai) -> let c = DR(floor ap, ref (D 0.), atr, ai) in atr.Push(Floor(a, c)); c
    static member Ceiling (a:D) =
        if isInteger (float a) then invalidArgCeil()
        match a with
        | D(a) -> D(ceil a)
        | DR(ap, _, atr, ai) -> let c = DR(ceil ap, ref (D 0.), atr, ai) in atr.Push(Ceil(a, c)); c
    static member Round (a:D) =
        if isHalfway (float a) then invalidArgRound()
        match a with
        | D(a) -> D(round a)
        | DR(ap, _, atr, ai) -> let c = DR(round ap, ref (D 0.), atr, ai) in atr.Push(Round(a, c)); c

/// Operation types for the trace
and Op =
    | Add        of D * D * D
    | AddCons    of D * D
    | Sub        of D * D * D
    | SubDCons   of D * D
    | SubConsD   of D * D
    | Mul        of D * D * D
    | MulCons    of D * D * D
    | Div        of D * D * D
    | DivDCons   of D * D * D
    | DivConsD   of D * D * D
    | Pow        of D * D * D
    | PowDCons   of D * D * D
    | PowConsD   of D * D * D
    | Atan2      of D * D * D
    | Atan2DCons of D * D * D
    | Atan2ConsD of D * D * D
    | Log        of D * D
    | Log10      of D * D
    | Exp        of D * D
    | Sin        of D * D
    | Cos        of D * D
    | Tan        of D * D
    | Neg        of D * D
    | Sqrt       of D * D
    | Sinh       of D * D
    | Cosh       of D * D
    | Tanh       of D * D
    | Asin       of D * D
    | Acos       of D * D
    | Atan       of D * D
    | Abs        of D * D
    | Floor      of D * D
    | Ceil       of D * D
    | Round      of D * D

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

[<AutoOpen>]
module DOps =
    let inline reverseTr (d:D) =
        match d with
        | D(_) -> ()
        | DR(_,_,o,_) ->
            for op in o do
                match op with
                | Add(a, b, c)           -> a.AddA c.A; b.AddA c.A
                | AddCons(a, c)          -> a.AddA c.A
                | Sub(a, b, c)           -> a.AddA c.A; b.AddA -c.A
                | SubDCons(a, c)         -> a.AddA c.A
                | SubConsD(a, c)         -> a.AddA -c.A
                | Mul(a, b, c)           -> a.AddA (c.A * b.P); b.AddA (c.A * a.P)
                | MulCons(a, cons, c)    -> a.AddA (c.A * cons)
                | Div(a, b, c)           -> a.AddA (c.A / b.P); b.AddA (c.A * (-a.P / (b.P * b.P)))
                | DivDCons(a, cons, c)   -> a.AddA (c.A / cons)
                | DivConsD(cons, b, c)   -> b.AddA (c.A * (-cons / (b.P * b.P)))
                | Pow(a, b, c)           -> a.AddA (c.A * (a.P ** (b.P - D 1.)) * b.P); b.AddA (c.A * (a.P ** b.P) * log a.P)
                | PowDCons(a, cons, c)   -> a.AddA (c.A * (a.P ** (cons - D 1.)) * cons)
                | PowConsD(cons, b, c)   -> b.AddA (c.A * (cons ** b.P) * log cons)
                | Atan2(a, b, c)         -> let denom = a.P * a.P + b.P * b.P in a.AddA (c.A * b.P / denom); b.AddA (c.A * (-a.P) / denom)
                | Atan2DCons(a, cons, c) -> a.AddA (c.A * cons / (a.P * a.P + cons * cons))
                | Atan2ConsD(cons, b, c) -> b.AddA (c.A * (-cons) / (cons * cons + b.P * b.P))
                | Log(a, c)              -> a.AddA (c.A / a.P)
                | Log10(a, c)            -> a.AddA (c.A / (a.P * log10val))
                | Exp(a, c)              -> a.AddA (c.A * c.P) // c.P = exp a.P
                | Sin(a, c)              -> a.AddA (c.A * cos a.P)
                | Cos(a, c)              -> a.AddA (c.A * (-sin a.P))
                | Tan(a, c)              -> let seca = D 1. / cos a.P in a.AddA (c.A * seca * seca)
                | Neg(a, c)              -> a.AddA -c.A
                | Sqrt(a, c)             -> a.AddA (c.A / (D 2. * c.P)) // c.P = sqrt a.P
                | Sinh(a, c)             -> a.AddA (c.A * cosh a.P)
                | Cosh(a, c)             -> a.AddA (c.A * sinh a.P)
                | Tanh(a, c)             -> let secha = D 1. / cosh a.P in a.AddA (c.A * secha * secha)
                | Asin(a, c)             -> a.AddA (c.A / sqrt (D 1. - a.P * a.P))
                | Acos(a, c)             -> a.AddA (-c.A / sqrt (D 1. - a.P * a.P))
                | Atan(a, c)             -> a.AddA (c.A / (D 1. + a.P * a.P))
                | Abs(a, c)              -> a.AddA (c.A * float (sign (float a.P)))
                | Floor(_, _)            -> ()
                | Ceil(_, _)             -> ()
                | Round(_, _)            -> ()

    let inline cleanTr (d:D) =
        match d with
        | D(_) -> ()
        | DR(_,_,o,_) ->
            for op in o do
                match op with
                | Add(a,b,c) | Sub(a,b,c) | Mul(a,b,c) | Div(a,b,c) | Pow(a,b,c) | Atan2(a,b,c) -> a.A <- D 0.; b.A <- D 0.; c.A <- D 0.
                | MulCons(a,_,b) | DivDCons(a,_,b) | DivConsD(_,a,b) | PowDCons(a,_,b) | PowConsD(_,a,b) | Atan2DCons(a,_,b) | Atan2ConsD(_,a,b) | Log(a,b) | Log10(a,b) | Exp(a,b) | Sin(a,b) | Cos(a,b) | Tan(a,b) | Neg(a,b) | Sqrt(a,b) | Sinh(a,b) | Cosh(a,b) | Tanh(a,b) | Asin(a,b) | Acos(a,b) | Atan(a,b) | Abs(a,b) | Floor(a,b) | Ceil(a,b) | Round(a,b) -> a.A <- D 0.; b.A <- D 0.
                | AddCons(a,_) | SubDCons(a,_) | SubConsD(_,a) -> a.A <- D 0.

    let inline makeDR i p = 
        DR(p, ref (D 0.), Stack<Op>(), i) 

    let inline adjoint (d:D) = d.A

    let inline primal (d:D) = d.P

[<AutoOpen>]
module ReverseOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f x =
        let xa = x |> makeDR GlobalTagger.Next
        let (z:D) = xa |> f
        z.A <- D 1.
        z |> reverseTr
        (primal z, adjoint xa)

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        diff' f x |> snd

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
        if n < 0 then invalidArg "" "Order of differentiation cannot be negative."
        elif n = 0 then x |> f
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> makeDR GlobalTagger.Next |> (d n f)

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' n f x =
        (diffn 0 f x, diffn n f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`    
    let inline grad' f (x:_[]) =
        let i = GlobalTagger.Next
        let xa = x |> Array.map (makeDR i)
        let z:D = f xa
        z.A <- D 1.
        z |> reverseTr
        (primal z, Array.map adjoint xa)

    /// Gradient of a vector-to-scalar function `f`, at point `x`    
    let inline grad f x =
        grad' f x |> snd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f (x:_[]) =
        let a = Array.init x.Length (fun i -> x |> fVVtoSS i i (grad f) |> diff <| x.[i])
        (x |> f, a |> Array.sum)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' f x =
        let i = GlobalTagger.Next
        let xa = x |> Array.map (makeDR i)
        let z:D[] = f xa
        let r1 = Array.map primal z
        let r2 =
            fun v ->
                Array.iter cleanTr z
                Array.iter2 (fun (a:D) b -> a.A <- b) z v
                Array.iter reverseTr z
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

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x =
        jacobian' (grad f) x

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f x =
        let g, h = gradhessian f x
        (x |> f, g, h)

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
    let inline diff' (f:D->D) x = ReverseOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:D->D) x = ReverseOps.diff f x
    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2' (f:D->D) x = ReverseOps.diff2' f x
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2 (f:D->D) x = ReverseOps.diff2 f x
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff2'' (f:D->D) x = ReverseOps.diff2'' f x
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn' (n:int) (f:D->D) x = ReverseOps.diffn' n f x
    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diffn (n:int) (f:D->D) x = ReverseOps.diffn n f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<D>->D) x = ReverseOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<D>->D) x = ReverseOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<D>->D) x = ReverseOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point x
    let inline laplacian (f:Vector<D>->D) x = ReverseOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<D>->Vector<D>) x = ReverseOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<D>->Vector<D>) x = ReverseOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<D>->Vector<D>) x = ReverseOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<D>->Vector<D>) x = ReverseOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv (f:Vector<D>->Vector<D>) x v = ReverseOps.jacobianTv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv' (f:Vector<D>->Vector<D>) x v = ReverseOps.jacobianTv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of the reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' (f:Vector<D>->Vector<D>) x = ReverseOps.jacobianTv'' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Vector.toArray >> b >> vector)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<D>->D) x = ReverseOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<D>->D) x = ReverseOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2D b)
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<D>->D) x = ReverseOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2D c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<D>->D) x = ReverseOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<D>->Vector<D>) x = ReverseOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<D>->Vector<D>) x = ReverseOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<D>->Vector<D>) x = ReverseOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<D>->Vector<D>) x = ReverseOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<D>->Vector<D>) x = ReverseOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<D>->Vector<D>) x = ReverseOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)