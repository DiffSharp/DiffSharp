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

/// Nested mixed (forward and reverse combined) mode AD
namespace DiffSharp.AD

open DiffSharp.Util
open FsAlg.Generic

/// Numeric type keeping dual numbers for forward mode and adjoints and tapes for reverse mode AD, with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]
type D =
    | D of float // Primal
    | DF of D * D * uint32 // Primal, tangent, tag
    | DR of D * (D ref) * TraceOp * (uint32 ref) * uint32 // Primal, adjoint, parent operation, fan-out counter, tag
    /// Primal value of this D
    member d.P =
        match d with
        | D(_) -> d
        | DF(p,_,_) -> p
        | DR(p,_,_,_,_) -> p
    /// Tangent value of this D
    member d.T =
        match d with
        | D(_) -> D 0.
        | DF(_,t,_) -> t
        | DR(_,_,_,_,_) -> failwith "DR does not have a tangent value."
    /// Adjoint value of this D
    member d.A
        with get() =
            match d with
            | D(_) -> D 0.
            | DF(_,_,_) -> failwith "DF does not have an adjoint value."
            | DR(_,a,_,_,_) -> !a
        and set(v) =
            match d with
            | D(_) -> ()
            | DF(_,_,_) -> failwith "Cannot set adjoint value for DF."
            | DR(_,a,_,_,_) -> a := v
    /// Fan-out counter of this D
    member d.F
        with get() =
            match d with
            | D(_) -> 0u
            | DF(_,_,_) -> failwith "DF does not have a fan-out value."
            | DR(_,_,_,f,_) -> !f
        and set(v) =
            match d with
            | D(_) -> ()
            | DF(_,_,_) -> failwith "Cannot set fan-out value for DF."
            | DR(_,_,_,f,_) -> f := v
    static member op_Explicit(d:D):float =
        match d with
        | D(a) -> a
        | DF(ap,_,_) -> float ap
        | DR(ap,_,_,_,_) -> float ap
    static member op_Explicit(d:D):int =
        match d with
        | D(a) -> int a
        | DF(ap,_,_) -> int ap
        | DR(ap,_,_,_,_) -> int ap
    static member DivideByInt(d:D, i:int) = d / float i
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
        | DR(ap,_,ao,_,ai) -> hash [|ap; ao; ai|]
    // D - D binary operations
    static member (+) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp)               -> D(ap + bp)
        | D(ap), DF(bp, bt, bi)      -> DF(ap + bp, bt, bi)
        | D(ap), DR(bp, _, _, _, bi) -> DR(ap + bp, ref (D 0.), AddCons(b), ref 0u, bi)
        | DF(ap, at, ai), D(bp)      -> DF(ap + bp, at, ai)
        | DR(ap, _, _, _, ai), D(bp) -> DR(ap + bp, ref (D 0.), AddCons(a), ref 0u, ai)
        | DF( _,  _, ai), DF(bp, bt, bi) when ai < bi -> DF(a + bp, bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap + bp, at + bt, ai)
        | DF(ap, at, ai), DF( _,  _, bi) when ai > bi -> DF(ap + b, at, ai)
        | DR( _, _, _, _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a + bp, ref (D 0.), AddCons(b), ref 0u, bi)
        | DR(ap, _, _, _, ai), DR(bp, _, _, _, bi) when ai = bi -> DR(ap + bp, ref (D 0.), Add(a, b), ref 0u, ai)
        | DR(ap, _, _, _, ai), DR( _, _, _, _, bi) when ai > bi -> DR(ap + b, ref (D 0.), AddCons(a), ref 0u, ai)
        | DF( _,  _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a + bp, ref (D 0.), AddCons(b), ref 0u, bi)
        | DF(ap, at, ai), DR( _, _, _, _, bi) when ai > bi -> DF(ap + b, at, ai)
        | DR( _, _, _, _, ai), DF(bp, bt, bi) when ai < bi -> DF(a + bp, bt, bi)
        | DR(ap, _, _, _, ai), DF( _,  _, bi) when ai > bi -> DR(ap + b, ref (D 0.), AddCons(a), ref 0u, ai)
    static member (-) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp)               -> D(ap - bp)
        | D(ap), DF(bp, bt, bi)      -> DF(ap - bp, -bt, bi)
        | D(ap), DR(bp, _, _, _, bi) -> DR(ap - bp, ref (D 0.), SubConsD(b), ref 0u, bi)
        | DF(ap, at, ai), D(bp)      -> DF(ap - bp, at, ai)
        | DR(ap, _, _, _, ai), D(bp) -> DR(ap - bp, ref (D 0.), SubDCons(a), ref 0u, ai)
        | DF( _,  _, ai), DF(bp, bt, bi) when ai < bi -> DF(a - bp, -bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap - bp, at - bt, ai)
        | DF(ap, at, ai), DF( _,  _, bi) when ai > bi -> DF(ap - b, at, ai)
        | DR( _, _, _, _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a - bp, ref (D 0.), SubConsD(b), ref 0u, bi)
        | DR(ap, _, _, _, ai), DR(bp, _, _, _, bi) when ai = bi -> DR(ap - bp, ref (D 0.), Sub(a, b), ref 0u, ai)
        | DR(ap, _, _, _, ai), DR( _, _, _, _, bi) when ai > bi -> DR(ap - b, ref (D 0.), SubDCons(a), ref 0u, ai)
        | DF( _,  _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a - bp, ref (D 0.), SubConsD(b), ref 0u, bi)
        | DF(ap, at, ai), DR( _, _, _, _, bi) when ai > bi -> DF(ap - b, at, ai)
        | DR( _, _, _, _, ai), DF(bp, bt, bi) when ai < bi -> DF(a - bp, -bt, bi)
        | DR(ap, _, _, _, ai), DF( _,  _, bi) when ai > bi -> DR(ap - b, ref (D 0.), SubDCons(a), ref 0u, ai)
    static member (*) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp)               -> D(ap * bp)
        | D(ap), DF(bp, bt, bi)      -> DF(ap * bp, ap * bt, bi)
        | D(ap), DR(bp, _, _, _, bi) -> DR(ap * bp, ref (D 0.), MulCons(b, a), ref 0u, bi)
        | DF(ap, at, ai), D(bp)      -> DF(ap * bp, at * bp, ai)
        | DR(ap, _, _, _, ai), D(bp) -> DR(ap * bp, ref (D 0.), MulCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DF(bp, bt, bi) when ai < bi -> DF(a * bp, a * bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap * bp, at * bp + bt * ap, ai)
        | DF(ap, at, ai), DF( _,  _, bi) when ai > bi -> DF(ap * b, at * b, ai)
        | DR( _, _, _, _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a * bp, ref (D 0.), MulCons(b, a), ref 0u, bi)
        | DR(ap, _, _, _, ai), DR(bp, _, _, _, bi) when ai = bi -> DR(ap * bp, ref (D 0.), Mul(a, b), ref 0u, ai)
        | DR(ap, _, _, _, ai), DR( _, _, _, _, bi) when ai > bi -> DR(ap * b, ref (D 0.), MulCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a * bp, ref (D 0.), MulCons(b, a), ref 0u, bi)
        | DF(ap, at, ai), DR( _, _, _, _, bi) when ai > bi -> DF(ap * b, at * b, ai)
        | DR( _, _, _, _, ai), DF(bp, bt, bi) when ai < bi -> DF(a * bp, a * bt, bi)
        | DR(ap, _, _, _, ai), DF( _,  _, bi) when ai > bi -> DR(ap * b, ref (D 0.), MulCons(a, b), ref 0u, ai)
    static member (/) (a:D, b:D) =
        match a, b with
        | D(ap), D(bp)               -> D(ap / bp)
        | D(ap), DF(bp, bt, bi)      -> DF(ap / bp, -(bt * ap) / (bp * bp), bi)
        | D(ap), DR(bp, _, _, _, bi) -> DR(ap / bp, ref (D 0.), DivConsD(b, a), ref 0u, bi)
        | DF(ap, at, ai), D(bp)      -> DF(ap / bp, at / bp, ai)
        | DR(ap, _, _, _, ai), D(bp) -> DR(ap / bp, ref (D 0.), DivDCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DF(bp, bt, bi) when ai < bi -> DF(a / bp, -(bt * a) / (bp * bp), bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(ap / bp, (at * bp - bt * ap) / (bp * bp), ai)
        | DF(ap, at, ai), DF( _,  _, bi) when ai > bi -> DF(ap / b, at / b, ai)
        | DR( _, _, _, _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a / bp, ref (D 0.), DivConsD(b, a), ref 0u, bi)
        | DR(ap, _, _, _, ai), DR(bp, _, _, _, bi) when ai = bi -> DR(ap / bp, ref (D 0.), Div(a, b), ref 0u, ai)
        | DR(ap, _, _, _, ai), DR( _, _, _, _, bi) when ai > bi -> DR(ap / b, ref (D 0.), DivDCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a / bp, ref (D 0.), DivConsD(b, a), ref 0u, bi)
        | DF(ap, at, ai), DR( _, _, _, _, bi) when ai > bi -> DF(ap / b, at / b, ai)
        | DR( _, _, _, _, ai), DF(bp, bt, bi) when ai < bi -> DF(a / bp, -(bt * a) / (bp * bp), bi)
        | DR(ap, _, _, _, ai), DF( _,  _, bi) when ai > bi -> DR(ap / b, ref (D 0.), DivDCons(a, b), ref 0u, ai)
    static member Pow (a:D, b:D) =
        match a, b with
        | D(ap), D(bp)               -> D(ap ** bp)
        | D(ap), DF(bp, bt, bi)      -> let apb = D.Pow(ap, bp) in DF(apb, apb * (log ap) * bt, bi)
        | D(ap), DR(bp, _, _, _, bi) -> DR(D.Pow(ap, bp), ref (D 0.), PowConsD(b, a), ref 0u, bi)
        | DF(ap, at, ai), D(bp)      -> DF(ap ** bp, bp * (ap ** (bp - D 1.)) * at, ai)
        | DR(ap, _, _, _, ai), D(bp) -> DR(ap ** bp, ref (D 0.), PowDCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DF(bp, bt, bi) when ai < bi -> let apb = a ** bp in DF(apb, apb * (log a) * bt, bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> let apb = ap ** bp in DF(apb, apb * ((bp * at / ap) + ((log ap) * bt)), ai)
        | DF(ap, at, ai), DF( _,  _, bi) when ai > bi -> let apb = ap ** b in DF(apb, apb * (b * at / ap), ai)
        | DR( _, _, _, _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a ** bp, ref (D 0.), PowConsD(b, a), ref 0u, bi)
        | DR(ap, _, _, _, ai), DR(bp, _, _, _, bi) when ai = bi -> DR(ap ** bp, ref (D 0.), Pow(a, b), ref 0u, ai)
        | DR(ap, _, _, _, ai), DR( _, _, _, _, bi) when ai > bi -> DR(ap ** b, ref (D 0.), PowDCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(a ** bp, ref (D 0.), PowConsD(b, a), ref 0u, bi)
        | DF(ap, at, ai), DR( _, _, _, _, bi) when ai > bi -> let apb = ap ** b in DF(apb, apb * (b * at / ap), ai)
        | DR( _, _, _, _, ai), DF(bp, bt, bi) when ai < bi -> let apb = a ** bp in DF(apb, apb * (log a) * bt, bi)
        | DR(ap, _, _, _, ai), DF( _,  _, bi) when ai > bi -> DR(ap ** b, ref (D 0.), PowDCons(a, b), ref 0u, ai)
    static member Atan2 (a:D, b:D) =
        match a, b with
        | D(ap), D(bp)               -> D(atan2 ap bp)
        | D(ap), DF(bp, bt, bi)      -> DF(D.Atan2(ap, bp), -(ap * bt) / (ap * ap + bp * bp), bi)
        | D(ap), DR(bp, _, _, _, bi) -> DR(D.Atan2(ap, bp), ref (D 0.), Atan2ConsD(b, a), ref 0u, bi)
        | DF(ap, at, ai), D(bp)      -> DF(D.Atan2(ap, bp), (bp * at) / (ap * ap + bp * bp), ai)
        | DR(ap, _, _, _, ai), D(bp) -> DR(D.Atan2(ap, bp), ref (D 0.), Atan2DCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DF(bp, bt, bi) when ai < bi -> DF(atan2 a bp, -(a * bt) / (a * a + bp * bp), bi)
        | DF(ap, at, ai), DF(bp, bt, bi) when ai = bi -> DF(atan2 ap bp, (at * bp - ap * bt) / (ap * ap + bp * bp), ai)
        | DF(ap, at, ai), DF( _,  _, bi) when ai > bi -> DF(atan2 ap b, (at * b) / (ap * ap + b * b), ai)
        | DR( _, _, _, _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(atan2 a bp, ref (D 0.), Atan2ConsD(b, a), ref 0u, bi)
        | DR(ap, _, _, _, ai), DR(bp, _, _, _, bi) when ai = bi -> DR(atan2 ap bp, ref (D 0.), Atan2(a, b), ref 0u, ai)
        | DR(ap, _, _, _, ai), DR( _, _, _, _, bi) when ai > bi -> DR(atan2 ap b, ref (D 0.), Atan2DCons(a, b), ref 0u, ai)
        | DF( _,  _, ai), DR(bp, _, _, _, bi) when ai < bi -> DR(atan2 a bp, ref (D 0.), Atan2ConsD(b, a), ref 0u, bi)
        | DF(ap, at, ai), DR( _, _, _, _, bi) when ai > bi -> DF(atan2 ap b, (at * b) / (ap * ap + b * b), ai)
        | DR( _, _, _, _, ai), DF(bp, bt, bi) when ai < bi -> DF(atan2 a bp, -(a * bt) / (a * a + bp * bp), bi)
        | DR(ap, _, _, _, ai), DF( _,  _, bi) when ai > bi -> DR(atan2 ap b, ref (D 0.), Atan2DCons(a, b), ref 0u, ai)
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
        | DR(ap,_,_,_,ai) -> DR(log ap, ref (D 0.), Log(a), ref 0u, ai)
    static member Log10 (a:D) =
        if (float a) <= 0. then invalidArgLog10()
        match a with
        | D(ap) -> D(log10 ap)
        | DF(ap, at, ai) -> DF(log10 ap, at / (ap * log10val), ai)
        | DR(ap,_,_,_,ai) -> DR(log10 ap, ref (D 0.), Log10(a), ref 0u, ai)
    static member Exp (a:D) =
        match a with
        | D(ap) -> D(exp ap)
        | DF(ap, at, ai) -> let expa = exp ap in DF(expa, at * expa, ai)
        | DR(ap,_,_,_,ai) -> DR(exp ap, ref (D 0.), Exp(a), ref 0u, ai)
    static member Sin (a:D) =
        match a with
        | D(ap) -> D(sin ap)
        | DF(ap, at, ai) -> DF(sin ap, at * cos ap, ai)
        | DR(ap,_,_,_,ai) -> DR(sin ap, ref (D 0.), Sin(a), ref 0u, ai)
    static member Cos (a:D) =
        match a with
        | D(ap) -> D(cos ap)
        | DF(ap, at, ai) -> DF(cos ap, -at * sin ap, ai)
        | DR(ap,_,_,_,ai) -> DR(cos ap, ref (D 0.), Cos(a), ref 0u, ai)
    static member Tan (a:D) =
        if (float (cos a)) = 0. then invalidArgTan()
        match a with
        | D(ap) -> D(tan ap)
        | DF(ap, at, ai) -> let cosa = cos ap in DF(tan ap, at / (cosa * cosa) , ai)
        | DR(ap,_,_,_,ai) -> DR(tan ap, ref (D 0.), Tan(a), ref 0u, ai)
    static member (~-) (a:D) =
        match a with
        | D(ap) -> D(-ap)
        | DF(ap, at, ai) -> DF(-ap, -at, ai)
        | DR(ap,_,_,_,ai) -> DR(-ap, ref (D 0.), Neg(a), ref 0u, ai)
    static member Sqrt (a:D) =
        if (float a) <= 0. then invalidArgSqrt()
        match a with
        | D(ap) -> D(sqrt ap)
        | DF(ap, at, ai) -> let sqrta = sqrt ap in DF(sqrta, at / (D 2. * sqrta), ai)
        | DR(ap,_,_,_,ai) -> DR(sqrt ap, ref (D 0.), Sqrt(a), ref 0u, ai)
    static member Sinh (a:D) =
        match a with
        | D(ap) -> D(sinh ap)
        | DF(ap, at, ai) -> DF(sinh ap, at * cosh ap, ai)
        | DR(ap,_,_,_,ai) -> DR(sinh ap, ref (D 0.), Sinh(a), ref 0u, ai)
    static member Cosh (a:D) =
        match a with
        | D(ap) -> D(cosh ap)
        | DF(ap, at, ai) -> DF(cosh ap, at * sinh ap, ai)
        | DR(ap,_,_,_,ai) -> DR(cosh ap, ref (D 0.), Cosh(a), ref 0u, ai)
    static member Tanh (a:D) =
        match a with
        | D(ap) -> D(tanh ap)
        | DF(ap, at, ai) -> let cosha = cosh ap in DF(tanh ap, at / (cosha * cosha), ai)
        | DR(ap,_,_,_,ai) -> DR(tanh ap, ref (D 0.), Tanh(a), ref 0u, ai)
    static member Asin (a:D) =
        if abs (float a) >= 1. then invalidArgAsin()
        match a with
        | D(ap) -> D(asin ap)
        | DF(ap, at, ai) -> DF(asin ap, at / sqrt (D 1. - ap * ap), ai)
        | DR(ap,_,_,_,ai) -> DR(asin ap, ref (D 0.), Asin(a), ref 0u, ai)
    static member Acos (a:D) =
        if abs (float a) >= 1. then invalidArgAcos()
        match a with
        | D(ap) -> D(acos ap)
        | DF(ap, at, ai) -> DF(acos ap, -at / sqrt (D 1. - ap * ap), ai)
        | DR(ap,_,_,_,ai) -> DR(acos ap, ref (D 0.), Acos(a), ref 0u, ai)
    static member Atan (a:D) =
        match a with
        | D(ap) -> D(atan ap)
        | DF(ap, at, ai) -> DF(atan ap, at / (D 1. + ap * ap), ai)
        | DR(ap,_,_,_,ai) -> DR(atan ap, ref (D 0.), Atan(a), ref 0u, ai)
    static member Abs (a:D) =
        if float a = 0. then invalidArgAbs()
        match a with
        | D(ap) -> D(abs ap)
        | DF(ap, at, ai) -> DF(abs ap, at * float (sign (float ap)), ai)
        | DR(ap,_,_,_,ai) -> DR(abs ap, ref (D 0.), Abs(a), ref 0u, ai)
    static member Floor (a:D) =
        if isInteger (float a) then invalidArgFloor()
        match a with
        | D(ap) -> D(floor ap)
        | DF(ap, _, ai) -> DF(floor ap, D 0., ai)
        | DR(ap,_,_,_,ai) -> DR(floor ap, ref (D 0.), Floor(a), ref 0u, ai)
    static member Ceiling (a:D) =
        if isInteger (float a) then invalidArgCeil()
        match a with
        | D(ap) -> D(ceil ap)
        | DF(ap, _, ai) -> DF(ceil ap, D 0., ai)
        | DR(ap,_,_,_,ai) -> DR(ceil ap, ref (D 0.), Ceil(a), ref 0u, ai)
    static member Round (a:D) =
        if isHalfway (float a) then invalidArgRound()
        match a with
        | D(ap) -> D(round ap)
        | DF(ap, _, ai) -> DF(round ap, D 0., ai)
        | DR(ap,_,_,_,ai) -> DR(round ap, ref (D 0.), Round(a), ref 0u, ai)

/// Operation types recorded in the evaluation trace
and TraceOp =
    | Add        of D * D
    | AddCons    of D
    | Sub        of D * D
    | SubDCons   of D
    | SubConsD   of D
    | Mul        of D * D
    | MulCons    of D * D
    | Div        of D * D
    | DivDCons   of D * D
    | DivConsD   of D * D
    | Pow        of D * D
    | PowDCons   of D * D
    | PowConsD   of D * D
    | Atan2      of D * D
    | Atan2DCons of D * D
    | Atan2ConsD of D * D
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
    /// Make DF, with tag `i`, primal value `p`, and tangent value `t`
    let inline makeDF i t p = DF(p, t, i)
    /// Make DR, with tag `i` and primal value `p`
    let inline makeDR i p = DR(p, ref (D 0.), Noop, ref 0u, i)
    /// Get the primal value of `d`
    let inline primal (d:D) = d.P
    /// Get the tangent value of `d`
    let inline tangent (d:D) = d.T
    /// Get the adjoint value of `d`
    let inline adjoint (d:D) = d.A
    /// Get the primal and tangent values of  `d`, as a tuple
    let inline tuple (d:D) = (d.P, d.T)
    /// Pushes the adjoint `v` backwards through the evaluation trace of `d`
    let rec reversePush (v:D) (d:D) =
        match d with
        | DR(_,_,o,_,_) ->
            d.A <- d.A + v
            d.F <- d.F - 1u
            if d.F = 0u then
                match o with
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
                | Pow(a, b)           -> reversePush (d.A * (a.P ** (b.P - D 1.)) * b.P) a; reversePush (d.A * (a.P ** b.P) * log a.P) b
                | PowDCons(a, cons)   -> reversePush (d.A * (a.P ** (cons - D 1.)) * cons) a
                | PowConsD(a, cons)   -> reversePush (d.A * (cons ** a.P) * log cons) a
                | Atan2(a, b)         -> let denom = a.P * a.P + b.P * b.P in reversePush (d.A * b.P / denom) a; reversePush (d.A * (-a.P) / denom) b
                | Atan2DCons(a, cons) -> reversePush (d.A * cons / (a.P * a.P + cons * cons)) a
                | Atan2ConsD(a, cons) -> reversePush (d.A * (-cons) / (cons * cons + a.P * a.P)) a
                | Log(a)              -> reversePush (d.A / a.P) a
                | Log10(a)            -> reversePush (d.A / (a.P * log10val)) a
                | Exp(a)              -> reversePush (d.A * d.P) a // d.P = exp a.P
                | Sin(a)              -> reversePush (d.A * cos a.P) a
                | Cos(a)              -> reversePush (d.A * (-sin a.P)) a
                | Tan(a)              -> let seca = D 1. / cos a.P in reversePush (d.A * seca * seca) a
                | Neg(a)              -> reversePush -d.A a
                | Sqrt(a)             -> reversePush (d.A / (D 2. * d.P)) a // d.P = sqrt a.P
                | Sinh(a)             -> reversePush (d.A * cosh a.P) a
                | Cosh(a)             -> reversePush (d.A * sinh a.P) a
                | Tanh(a)             -> let secha = D 1. / cosh a.P in reversePush (d.A * secha * secha) a
                | Asin(a)             -> reversePush (d.A / sqrt (D 1. - a.P * a.P)) a
                | Acos(a)             -> reversePush (-d.A / sqrt (D 1. - a.P * a.P)) a
                | Atan(a)             -> reversePush (d.A / (D 1. + a.P * a.P)) a
                | Abs(a)              -> reversePush (d.A * float (sign (float a.P))) a
                | Floor(_)            -> ()
                | Ceil(_)             -> ()
                | Round(_)            -> ()
                | Noop                -> ()
        | _ -> ()
    /// Resets the adjoints of all the values in the evaluation trace of `d`
    let rec reverseReset (d:D) =
        match d with
        | DR(_,_,o,_,_) ->
            d.A <- D 0.
            d.F <- d.F + 1u
            if d.F = 1u then
                match o with
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
        | _ -> ()
    /// Propagates the adjoint `v` backwards through the evaluation trace of `d`. The adjoints in the trace are reset before the push.
    let reverseProp (v:D) (d:D) =
        d |> reverseReset
        d |> reversePush v

/// Forward and reverse differentiation operations module (automatically opened)
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
        (x |> f, diffn n f x)

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f x v =
        let i = GlobalTagger.Next
        Array.map2 (makeDF i) v x |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f x =
        let i = GlobalTagger.Next
        let xa = x |> Array.map (makeDR i)
        let z:D = f xa
        z |> reverseReset
        z |> reversePush (D 1.)
        (primal z, Array.map adjoint xa)

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

    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' f x =
        let i = GlobalTagger.Next
        let xa = x |> Array.map (makeDR i)
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
    let inline jacobian' f (x:_[]) =
        let o = x |> f |> Array.map primal
        if x.Length > o.Length then // f:R^n -> R^m and n > m, use reverse mode
            let r = jacobianTv f x
            (o, Array.init o.Length (fun j -> r (standardBasis o.Length j)) |> array2D)
        else                        // f:R^n -> R^m and n <= m, use forward mode
            (o, Array.init x.Length (fun i -> jacobianv f x (standardBasis x.Length i)) |> array2D |> transpose)

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
    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv (f:Vector<D>->Vector<D>) x v = DiffOps.jacobianTv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianTv' (f:Vector<D>->Vector<D>) x v = DiffOps.jacobianTv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of the reverse mode AD, with the given vector, without repeating the forward pass).
    let inline jacobianTv'' (f:Vector<D>->Vector<D>) x = DiffOps.jacobianTv'' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Vector.toArray >> b >> vector)
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