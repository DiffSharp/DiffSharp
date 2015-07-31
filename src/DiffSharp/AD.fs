//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under the LGPL license.
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


namespace DiffSharp.AD


open DiffSharp.Util
open DiffSharp.Engine
open System.Threading.Tasks

[<CustomEquality; CustomComparison>]
type D =
    | D of float
    | DF of D * D * uint32
    | DR of D * (D ref) * TraceOp * (uint32 ref) * uint32

    member d.P =
        match d with
        | D(_) -> d
        | DF(ap,_,_) -> ap
        | DR(ap,_,_,_,_) -> ap
    member d.T =
        match d with
        | D(_) -> D 0.
        | DF(_,at,_) -> at
        | DR(_,_,_,_,_) -> failwith "Cannot get tangent value of DR."
    member d.A
        with get() =
            match d with
            | D(_) -> D 0.
            | DF(_,_,_) -> failwith "Cannot get adjoint value of DF."
            | DR(_,a,_,_,_) -> !a
        and set(v) =
            match d with
            | D(_) -> ()
            | DF(_,_,_) -> failwith "Cannot set adjoint value of DF."
            | DR(_,a,_,_,_) -> a := v
    member d.F
        with get() =
            match d with
            | D(_) -> failwith "Cannot get fan-out value of D."
            | DF(_,_,_) -> failwith "Cannot get fan-out value of DF."
            | DR(_,_,_,f,_) -> !f
        and set(v) =
            match d with
            | D(_) -> failwith "Cannot set fan-out value of D."
            | DF(_,_,_) -> failwith "Cannot set fan-out value of DF."
            | DR(_,_,_,f,_) -> f := v
    member d.Copy() =
        match d with
        | D(ap) -> D(ap)
        | DF(ap,at,ai) -> DF(ap.Copy(), at.Copy(), ai)
        | DR(ap,aa,at,af,ai) -> DR(ap.Copy(), ref ((!aa).Copy()), at, ref (!af), ai)

    static member Zero = D 0.
    static member One = D 1.
    static member op_Explicit(d:D):float =
        match d with
        | D(ap) -> ap
        | DF(ap,_,_) -> float ap
        | DR(ap,_,_,_,_) -> float ap
    static member op_Explicit(d) = D(d)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> compare ((float) d) ((float) d2)
            | _ -> invalidArg "" "Cannot compare thid D with another type."
    override d.Equals(other) =
        match other with
        | :? D as d2 -> compare ((float) d) ((float) d2) = 0
        | _ -> false
    override d.GetHashCode() =
        match d with
        | D(ap) -> hash [|ap|]
        | DF(ap,at,ai) -> hash [|ap; at; ai|]
        | DR(ap,_,ao,_,ai) -> hash [|ap; ao; ai|]

    static member inline Op_D_D (a, ff, fd, df, r) =
        match a with
        | D(ap)                      -> D(ff(ap))
        | DF(ap, at, ai)             -> let cp = fd(ap) in DF(cp, df(cp, ap, at), ai)
        | DR(ap,_,_,_,ai)            -> DR(fd(ap), ref (D 0.), r(a), ref 0u, ai)

    static member inline Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | D(ap) ->
            match b with
            | D(bp)                  -> D(ff(ap, bp))
            | DF(bp, bt, bi)         -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi)
            | DR(bp,  _,  _,  _, bi) -> DR(fd(a, bp), ref (D 0.), r_c_d(a, b), ref 0u, bi)
        | DF(ap, at, ai) ->
            match b with
            | D(_)                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | 0                  -> let cp = fd(ap, bp) in DF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                 -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                  -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                 -> DR(fd(a, bp), ref (D 0.), r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                  -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                  -> failwith "Forward and reverse AD cannot run on the same level."
        | DR(ap,  _,  _,  _, ai) ->
            match b with
            | D(_)                   -> DR(fd(ap, b), ref (D 0.), r_d_c(a, b), ref 0u, ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | -1                 -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                  -> DR(fd(ap, b), ref (D 0.), r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                  -> failwith "Forward and reverse AD cannot run on the same level."
            | DR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                  -> DR(fd(ap, bp), ref (D 0.), r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                 -> DR(fd(a, bp), ref (D 0.), r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                  -> DR(fd(ap, b), ref (D 0.), r_d_c(a, b), ref 0u, ai) // ai > bi

    static member (+) (a:D, b:D) =
        let inline ff(a, b) = a + b
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_D_D(a, b)
        let inline r_d_c(a, b) = Add_D_DCons(a)
        let inline r_c_d(a, b) = Add_D_DCons(b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:D, b:D) =
        let inline ff(a, b) = a - b
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_D_D(a, b)
        let inline r_d_c(a, b) = Sub_D_DCons(a)
        let inline r_c_d(a, b) = Sub_DCons_D(b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:D, b:D) =
        let inline ff(a, b) = a * b
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = at * bp + ap * bt
        let inline r_d_d(a, b) = Mul_D_D(a, b)
        let inline r_d_c(a, b) = Mul_D_DCons(a, b)
        let inline r_c_d(a, b) = Mul_D_DCons(b, a)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (/) (a:D, b:D) =
        let inline ff(a, b) = a / b
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt * cp / bp // cp = a / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt * cp) / bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_D_D(a, b)
        let inline r_d_c(a, b) = Div_D_DCons(a, b)
        let inline r_c_d(a, b) = Div_D_DCons(a, b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Pow (a:D, b:D) =
        let inline ff(a, b) = a ** b
        let inline fd(a, b) = a ** b
        let inline df_da(cp, ap, at) = at * cp * b / ap // cp = ap ** b
        let inline df_db(cp, bp, bt) = bt * cp * log a // cp = a ** bp
        let inline df_dab(cp, ap, at, bp, bt) = cp * (at * bp / ap + bt * log ap) // cp = ap ** bp
        let inline r_d_d(a, b) = Pow_D_D(a, b)
        let inline r_d_c(a, b) = Pow_D_DCons(a, b)
        let inline r_c_d(a, b) = Pow_D_DCons(a, b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Atan2 (a:D, b:D) =
        let inline ff(a, b) = atan2 a b
        let inline fd(a, b) = atan2 a b
        let inline df_da(cp, ap, at) = at * b / (ap * ap + b * b)
        let inline df_db(cp, bp, bt) = -bt * a / (a * a + bp * bp)
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp - bt * ap) / (ap * ap + bp * bp)
        let inline r_d_d(a, b) = Atan2_D_D(a, b)
        let inline r_d_c(a, b) = Atan2_D_DCons(a, b)
        let inline r_c_d(a, b) = Atan2_D_DCons(a, b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // D - float binary operations
    static member (+) (a:D, b:float) = a + (D b)
    static member (-) (a:D, b:float) = a - (D b)
    static member (*) (a:D, b:float) = a * (D b)
    static member (/) (a:D, b:float) = a / (D b)

    // float - D binary operations
    static member (+) (a:float, b:D) = (D a) + b
    static member (-) (a:float, b:D) = (D a) - b
    static member (*) (a:float, b:D) = (D a) * b
    static member (/) (a:float, b:D) = (D a) / b

    // D - int binary operations
    static member (+) (a:D, b:int) = a + (D (float b))
    static member (-) (a:D, b:int) = a - (D (float b))
    static member (*) (a:D, b:int) = a * (D (float b))
    static member (/) (a:D, b:int) = a / (D (float b))

    // int - D binary operations
    static member (+) (a:int, b:D) = (D (float a)) + b
    static member (-) (a:int, b:D) = (D (float a)) - b
    static member (*) (a:int, b:D) = (D (float a)) * b
    static member (/) (a:int, b:D) = (D (float a)) / b

    static member Log (a:D) =
        let inline ff(a) = log a
        let inline fd(a) = log a
        let inline df(cp, ap, at) = at / ap
        let inline r(a) = Log_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Log10 (a:D) =
        let inline ff(a) = log10 a
        let inline fd(a) = log10 a
        let inline df(cp, ap:D, at) = at / (ap * log10val)
        let inline r(a) = Log10_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Exp (a:D) =
        let inline ff(a) = exp a
        let inline fd(a) = exp a
        let inline df(cp, ap, at) = at * cp // cp = exp ap
        let inline r(a) = Exp_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Sin (a:D) =
        let inline ff(a) = sin a
        let inline fd(a) = sin a
        let inline df(cp, ap, at) = at * cos ap
        let inline r(a) = Sin_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Cos (a:D) =
        let inline ff(a) = cos a
        let inline fd(a) = cos a
        let inline df(cp, ap, at) = -at * sin ap
        let inline r(a) = Cos_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Tan (a:D) =
        let inline ff(a) = tan a
        let inline fd(a) = tan a
        let inline df(cp, ap, at) = let cosa = cos ap in at / (cosa * cosa)
        let inline r(a) = Tan_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member (~-) (a:D) =
        let inline ff(a) = -a
        let inline fd(a) = -a
        let inline df(cp, ap, at) = -at
        let inline r(a) = Neg_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Sqrt (a:D) =
        let inline ff(a) = sqrt a
        let inline fd(a) = sqrt a
        let inline df(cp, ap, at) = at / ((D 2.) * cp) // cp = sqrt ap
        let inline r(a) = Sqrt_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Sinh (a:D) =
        let inline ff(a) = sinh a
        let inline fd(a) = sinh a
        let inline df(cp, ap, at) = at * cosh ap
        let inline r(a) = Sinh_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Cosh (a:D) =
        let inline ff(a) = cosh a
        let inline fd(a) = cosh a
        let inline df(cp, ap, at) = at * sinh ap
        let inline r(a) = Cosh_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Tanh (a:D) =
        let inline ff(a) = tanh a
        let inline fd(a) = tanh a
        let inline df(cp, ap, at) = let cosha = cosh ap in at / (cosha * cosha)
        let inline r(a) = Tanh_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Asin (a:D) =
        let inline ff(a) = asin a
        let inline fd(a) = asin a
        let inline df(cp, ap, at) = at / sqrt (D 1. - ap * ap)
        let inline r(a) = Asin_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Acos (a:D) =
        let inline ff(a) = acos a
        let inline fd(a) = acos a
        let inline df(cp, ap, at) = -at / sqrt (D 1. - ap * ap)
        let inline r(a) = Acos_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Atan (a:D) =
        let inline ff(a) = atan a
        let inline fd(a) = atan a
        let inline df(cp, ap, at) = at / (D 1. + ap * ap)
        let inline r(a) = Atan_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Abs (a:D) =
        let inline ff(a) = abs a
        let inline fd(a) = abs a
        let inline df(cp, ap, at) = at * D.Sign(ap)
        let inline r(a) = Sign_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Sign (a:D) =
        let inline ff(a) = float (sign a)
        let inline fd(a) = D.Sign(a)
        let inline df(cp, ap, at) = D 0.
        let inline r(a) = Sign_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Floor (a:D) =
        let inline ff(a) = floor a
        let inline fd(a) = floor a
        let inline df(cp, ap, at) = D 0.
        let inline r(a) = Floor_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Ceiling (a:D) =
        let inline ff(a) = ceil a
        let inline fd(a) = ceil a
        let inline df(cp, ap, at) = D 0.
        let inline r(a) = Ceil_D(a)
        D.Op_D_D (a, ff, fd, df, r)
    static member Round (a:D) =
        let inline ff(a) = round a
        let inline fd(a) = round a
        let inline df(cp, ap, at) = D 0.
        let inline r(a) = Round_D(a)
        D.Op_D_D (a, ff, fd, df, r)


and DV =
    | DV of float[]
    | DVF of DV * DV * uint32
    | DVR of DV * (DV ref) * TraceOp * (uint32 ref) * uint32

    member d.P =
        match d with
        | DV(_) -> d
        | DVF(ap,_,_) -> ap
        | DVR(ap,_,_,_,_) -> ap
    member d.T =
        match d with
        | DV(_) -> DV Array.empty
        | DVF(_,at,_) -> at
        | DVR(_,_,_,_,_) -> failwith "Cannot get tangent value of DVR."
    member d.A
        with get() =
            match d with
            | DV(_) -> DV Array.empty
            | DVF(_,_,_) -> failwith "Cannot get adjoint value of DVF."
            | DVR(_,a,_,_,_) -> !a
        and set(v) =
            match d with
            | DV(_) -> ()
            | DVF(_,_,_) -> failwith "Cannot set adjoint value of DVF."
            | DVR(_,a,_,_,_) -> a := v
    member d.F
        with get() =
            match d with
            | DV(_) -> failwith "Cannot get fan-out value of DV."
            | DVF(_,_,_) -> failwith "Cannot get fan-out value of DVF."
            | DVR(_,_,_,f,_) -> !f
        and set(v) =
            match d with
            | DV(_) -> failwith "Cannot set fan-out value of DV."
            | DVF(_,_,_) -> failwith "Cannot set fan-out value of DVF."
            | DVR(_,_,_,f,_) -> f := v
    member d.Copy() =
        match d with
        | DV(ap) -> DV(Array.copy ap)
        | DVF(ap,at,ai) -> DVF(ap.Copy(), at.Copy(), ai)
        | DVR(ap,aa,at,af,ai) -> DVR(ap.Copy(), ref ((!aa).Copy()), at, ref (!af), ai)
    member d.Length =
        match d with
        | DV(ap) -> ap.Length
        | DVF(ap,_,_) -> ap.Length
        | DVR(ap,_,_,_,_) -> ap.Length
    member d.Item
        with get i =
            match d with
            | DV(ap) -> D(ap.[i])
            | DVF(ap,at,ai) -> DF(ap.[i], at.[i], ai)
            | DVR(ap,_,_,_,ai) -> DR(ap.[i], ref (D 0.), Item_DV(d, i), ref 0u, ai)

    member d.GetSlice(lower, upper) =
        let l = defaultArg lower 0
        let u = defaultArg upper (d.Length - 1)
        match d with
        | DV(ap) -> DV(ap.[l..u])
        | DVF(ap,at,ai) -> DVF(ap.[l..u], at.[l..u], ai)
        | DVR(ap,_,_,_,ai) -> let cp = ap.[l..u] in DVR(cp, ref (DV.ZeroN cp.Length), Slice_DV(d, l), ref 0u, ai)
    member d.L1Norm() =
        match d with
        | DV(ap) -> D(OpenBLAS.v_l1norm(ap))
        | DVF(ap,at,ai) -> DF(ap.L1Norm(), at * (DV.Sign ap), ai)
        | DVR(ap,_,_,_,ai) -> DR(ap.L1Norm(), ref (D 0.), L1Norm_DV(d), ref 0u, ai)
    member d.L2NormSq() =
        match d with
        | DV(ap) -> let l2norm = OpenBLAS.v_l2norm(ap) in D(l2norm * l2norm)
        | DVF(ap,at,ai) -> DF(ap.L2NormSq(), (D 2.) * (ap * at), ai)
        | DVR(ap,_,_,_,ai) -> DR(ap.L2NormSq(), ref (D 0.), L2NormSq_DV(d), ref 0u, ai)
    member d.L2Norm() =
        match d with
        | DV(ap) -> D(OpenBLAS.v_l2norm(ap))
        | DVF(ap,at,ai) -> let l2norm = ap.L2Norm() in DF(l2norm, (ap * at) / l2norm, ai)
        | DVR(ap,_,_,_,ai) -> DR(ap.L2Norm(), ref (D 0.), L2Norm_DV(d), ref 0u, ai)
    member d.Sum() =
        match d with
        | DV(ap) -> D(NonBLAS.v_sum(ap))
        | DVF(ap,at,ai) -> DF(ap.Sum(), at.Sum(), ai)
        | DVR(ap,_,_,_,ai) -> DR(ap.Sum(), ref (D 0.), Sum_DV(d), ref 0u, ai)
    member d.ToArray() =
        match d with
        | DV(ap) -> ap |> Array.Parallel.map D
        | DVF(ap,at,ai) ->
            Array.Parallel.init ap.Length (fun i -> DF(ap.[i], at.[i], ai))
        | DVR(ap,_,_,_,ai) ->
            Array.Parallel.init ap.Length (fun i -> DR(ap.[i], ref (D 0.), Item_DV(d, i), ref 0u, ai))
    member d.Split(n:seq<int>) =
        match d with
        | DV(ap) ->
            let i = ref 0
            seq {for j in n do yield Array.sub ap !i j |> DV; i := !i + j}
        | DVF(ap,at,ai) ->
            let aps = ap.Split(n)
            let ats = at.Split(n)
            Seq.map2 (fun p t -> DVF(p, t, ai)) aps ats
        | DVR(ap,_,_,_,ai) ->
            let aps = ap.Split(n)
            let ii = n |> Seq.mapFold (fun s i -> s, s + i) 0 |> fst
            Seq.mapi (fun i p -> DVR(p, ref (DV.ZeroN p.Length), Split_DV(d, ii |> Seq.item i), ref 0u, ai)) aps
    member d.ToRowMatrix() =
        match d with
        | DV(ap) -> seq [ap] |> array2D |> DM
        | DVF(ap,at,ai) -> DMF(ap.ToRowMatrix(), at.ToRowMatrix(), ai)
        | DVR(ap,_,_,_,ai) -> let cp = ap.ToRowMatrix() in DMR(cp, ref (DM.ZeroMN cp.Rows cp.Cols), RowMatrix_DV(d), ref 0u, ai)
    member d.ToColMatrix() = d.ToRowMatrix().Transpose()
    
    static member Zero = DV Array.empty
    static member ZeroN n = DV(Array.zeroCreate n)
    static member op_Explicit(d:DV):float[] =
        match d with
        | DV(ap) -> ap
        | DVF(ap,_,_) -> DV.op_Explicit(ap)
        | DVR(ap,_,_,_,_) -> DV.op_Explicit(ap)
    static member op_Explicit(d) = DV(d)
    static member ofArray(a:D[]) =
        // TODO: check to ensure that all elements in the array are of the same type (D, DF, or DR) and have the same nesting tag
        match a.[0] with
        | D(_) -> DV(a |> Array.Parallel.map float)
        | DF(_,_,ai) ->
            let ap = a |> Array.Parallel.map (fun x -> x.P)
            let at = a |> Array.Parallel.map (fun x -> x.T)
            DVF(DV.ofArray(ap), DV.ofArray(at), ai)
        | DR(_,_,_,_,ai) ->
            let ap = a |> Array.Parallel.map (fun x -> x.P)
            let cp = DV.ofArray(ap) in DVR(cp, ref (DV.ZeroN cp.Length), Make_DV(a), ref 0u, ai)


    static member inline Op_DV_DV (a, ff, fd, df, r) =
        match a with
        | DV(ap)                      -> DV(ff(ap))
        | DVF(ap, at, ai)             -> let cp = fd(ap) in DVF(cp, df(cp, ap, at), ai)
        | DVR(ap,_,_,_,ai)            -> let cp = fd(ap) in DVR(cp, ref (DV.ZeroN cp.Length), r(a), ref 0u, ai)

    static member inline Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DV(bp)                  -> DV(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp,  _,  _,  _, bi) -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi)
        | DVF(ap, at, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap,  _,  _,  _, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai) // ai > bi

    static member inline Op_DV_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DV(bp)                  -> DM(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp,  _,  _,  _, bi) -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi)
        | DVF(ap, at, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap,  _,  _,  _, ai) ->
            match b with
            | DV(_)                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> DMR(fd(ap, bp), ref DM.Zero, r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi

    static member inline Op_DV_DV_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DV(bp)                  -> D(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp,  _,  _,  _, bi) -> DR(fd(a, bp), ref (D 0.), r_c_d(a, b), ref 0u, bi)
        | DVF(ap, at, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> DR(fd(a, bp), ref (D 0.), r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap,  _,  _,  _, ai) ->
            match b with
            | DV(_)                   -> DR(fd(ap, b), ref (D 0.), r_d_c(a, b), ref 0u, ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> DR(fd(ap, b), ref (D 0.), r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> DR(fd(ap, bp), ref (D 0.), r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> DR(fd(a, bp), ref (D 0.), r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> DR(fd(ap, b), ref (D 0.), r_d_c(a, b), ref 0u, ai) // ai > bi

    static member inline Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | D(bp)                   -> DV(ff(ap, bp))
            | DF(bp, bt, bi)          -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DR(bp,  _,  _,  _, bi)  -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi)
        | DVF(ap, at, ai) ->
            match b with
            | D(_)                    -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | 0                    -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                   -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                    -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                   -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                    -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                    -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap,  _,  _,  _, ai) ->
            match b with
            | D(_)                    -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | -1                   -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                    -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                    -> failwith "Forward and reverse AD cannot run on the same level."
            | DR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                    -> let cp = fd(ap, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                   -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                    -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai) // ai > bi


    static member inline Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | D(ap) ->
            match b with
            | DV(bp)                  -> DV(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp,  _,  _,  _, bi) -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi)
        | DF(ap, at, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DR(ap,  _,  _,  _, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVR(cp, ref (DV.ZeroN cp.Length), r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVR(cp, ref (DV.ZeroN cp.Length), r_d_c(a, b), ref 0u, ai) // ai > bi

    static member (+) (a:DV, b:DV) =
        let inline ff(a, b) = OpenBLAS.v_add(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_DV(a, b)
        let inline r_d_c(a, b) = Add_DV_DVCons(a)
        let inline r_c_d(a, b) = Add_DV_DVCons(b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:DV, b:DV) =
        let inline ff(a, b) = OpenBLAS.v_sub(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DV_DV(a, b)
        let inline r_d_c(a, b) = Sub_DV_DVCons(a)
        let inline r_c_d(a, b) = Sub_DVCons_DV(b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // Inner (dot, scalar) product
    static member (*) (a:DV, b:DV) =
        let inline ff(a, b) = OpenBLAS.v_dot(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_Dot_DV_DV(a, b)
        let inline r_d_c(a, b) = Mul_Dot_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_Dot_DV_DVCons(b, a)
        DV.Op_DV_DV_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // Element-wise (Hadamard, Schur) product
    static member (.*) (a:DV, b:DV) =
        let inline ff(a, b) = NonBLAS.v_mul_hadamard(a, b)
        let inline fd(a, b) = a .* b
        let inline df_da(cp, ap, at) = at .* b
        let inline df_db(cp, bp, bt) = a .* bt
        let inline df_dab(cp, ap, at, bp, bt) = (at .* bp) + (ap .* bt)
        let inline r_d_d(a, b) = Mul_Had_DV_DV(a, b)
        let inline r_d_c(a, b) = Mul_Had_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_Had_DV_DVCons(b, a)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // Outer (dyadic, tensor) product of two vectors
    static member (&*) (a:DV, b:DV) =
        let inline ff(a, b) = OpenBLAS.v_mul_outer(a, b)
        let inline fd(a, b) = a &* b
        let inline df_da(cp, ap, at) = at &* b
        let inline df_db(cp, bp, bt) = bt &* a
        let inline df_dab(cp, ap, at, bp, bt) = (at &* bp) + (bt &* ap)
        let inline r_d_d(a, b) = Mul_Out_DV_DV(a, b)
        let inline r_d_c(a, b) = Mul_Out_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_Out_DVCons_DV(a, b)
        DV.Op_DV_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // Element-wise (Hadamard, Schur) division
    static member (./) (a:DV, b:DV) =
        let inline ff(a, b) = NonBLAS.v_div_hadamard(a, b)
        let inline fd(a, b) = a ./ b
        let inline df_da(cp, ap, at) = at ./ b
        let inline df_db(cp, bp, bt) = -bt .* cp ./ bp // cp = ap / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt .* cp) ./ bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_Had_DV_DV(a, b)
        let inline r_d_c(a, b) = Div_Had_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Div_Had_DV_DVCons(b, a)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:DV, b:D) =
        let inline ff(a, b) = OpenBLAS.v_scale(b, a)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DV_D(a, b)
        let inline r_d_c(a, b) = Mul_DV_DCons(a, b)
        let inline r_c_d(a, b) = Mul_DVCons_D(a, b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:D, b:DV) =
        let inline ff(a, b) = OpenBLAS.v_scale(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DV_D(b, a)
        let inline r_d_c(a, b) = Mul_DV_DCons(b, a)
        let inline r_c_d(a, b) = Mul_DVCons_D(b, a)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (/) (a:DV, b:D) =
        let inline ff(a, b) = OpenBLAS.v_scale(1. / b, a)
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt * cp / bp // cp = a / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt * cp) / bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_DV_D(a, b)
        let inline r_d_c(a, b) = Div_DV_DCons(a, b)
        let inline r_c_d(a, b) = Div_DVCons_D(a, b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (/) (a:D, b:DV) =
        let inline ff(a, b) = NonBLAS.sv_div(a, b)
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt .* (cp ./ bp) // cp = a / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt * cp) / bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_D_DV(a, b)
        let inline r_d_c(a, b) = Div_D_DVCons(a, b)
        let inline r_c_d(a, b) = Div_DCons_DV(a, b)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (+) (a:DV, b:D) =
        let inline ff(a, b) = NonBLAS.vs_add(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.ofArray(Array.create a.Length bt)
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_D(a, b)
        let inline r_d_c(a, b) = Add_DV_DCons(a)
        let inline r_c_d(a, b) = Add_DVCons_D(b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (+) (a:D, b:DV) =
        let inline ff(a, b) = NonBLAS.vs_add(b, a)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = DV.ofArray(Array.create b.Length at)
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_D(b, a)
        let inline r_d_c(a, b) = Add_DV_DCons(b)
        let inline r_c_d(a, b) = Add_DVCons_D(a)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:DV, b:D) =
        let inline ff(a, b) = NonBLAS.vs_sub(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.ofArray(Array.create a.Length -bt)
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DV_D(a, b)
        let inline r_d_c(a, b) = Sub_DV_DCons(a)
        let inline r_c_d(a, b) = Sub_DVCons_D(b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:D, b:DV) =
        let inline ff(a, b) = NonBLAS.sv_sub(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = DV.ofArray(Array.create b.Length at)
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_D_DV(a, b)
        let inline r_d_c(a, b) = Sub_D_DVCons(a)
        let inline r_c_d(a, b) = Sub_DCons_DV(b)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)
    

    static member AddItem (a:DV, i:int, b:D) =
        let inline ff(a, b) = let aa = Array.copy a in aa.[i] <- aa.[i] + b; aa
        let inline fd(a, b) = DV.AddItem(a, i, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.AddItem(DV.ZeroN a.Length, i, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DV.AddItem(at, i, bt)
        let inline r_d_d(a, b) = AddItem_DV_D(a, i, b)
        let inline r_d_c(a, b) = AddItem_DV_DCons(a)
        let inline r_c_d(a, b) = AddItem_DVCons_D(i, b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)
    
    static member AddSubVector (a:DV, i:int, b:DV) =
        let inline ff(a, b:_[]) = 
            let aa = Array.copy a 
            Parallel.For(0, b.Length, fun j -> aa.[i + j] <- b.[j]) |> ignore
            aa
        let inline fd(a, b) = DV.AddSubVector(a, i, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.AddSubVector(DV.ZeroN a.Length, i, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DV.AddSubVector(at, i, bt)
        let inline r_d_d(a, b) = AddSubVector_DV_DV(a, i, b)
        let inline r_d_c(a, b) = AddSubVector_DV_DVCons(a)
        let inline r_c_d(a, b) = AddSubVector_DVCons_DV(i, b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)


    // DV - float binary operations
    static member (*) (a:DV, b:float) = a * D b
    static member (+) (a:DV, b:float) = a + D b

    // float - DV binary operations
    static member (*) (a:float, b:DV) = b * D a
    static member (+) (a:float, b:DV) = b + D a

    // DV - int binary operations
    static member (*) (a:DV, b:int) = a * D (float b)
    static member (+) (a:DV, b:int) = a + D (float b)

    // int - DV binary operations
    static member (*) (a:int, b:DV) = b * D (float a)
    static member (+) (a:int, b:DV) = b + D (float a)

    static member (~-) (a:DV) =
        let inline ff(a) = OpenBLAS.v_scale(-1., a)
        let inline fd(a) = -a
        let inline df(cp, ap, at) = -at
        let inline r(a) = Neg_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Exp (a:DV) =
        let inline ff(a) = NonBLAS.v_exp(a)
        let inline fd(a) = exp a
        let inline df(cp, ap, at) = at .* cp // cp = exp ap
        let inline r(a) = Exp_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Abs (a:DV) =
        let inline ff(a) = NonBLAS.v_abs(a)
        let inline fd(a) = abs a
        let inline df(cp, ap, at) = at .* (DV.Sign ap)
        let inline r(a) = Abs_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Sign (a:DV) =
        let inline ff(a) = NonBLAS.v_sign(a)
        let inline fd(a) = DV.Sign a
        let inline df(cp, ap, at) = DV.Zero
        let inline r(a) = Sign_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Append (a:DV, b:DV) =
        let inline ff(a, b) = Array.append a b
        let inline fd(a, b) = DV.Append(a, b)
        let inline df_da(cp, ap, at) = DV.Append(at, b)
        let inline df_db(cp, bp, bt) = DV.Append(a, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DV.Append(at, bt)
        let inline r_d_d(a, b) = Append_DV_DV(a, b)
        let inline r_d_c(a, b) = Append_DV_DVCons(a)
        let inline r_c_d(a, b) = Append_DVCons_DV(b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)



and DM =
    | DM of float[,]
    | DMF of DM * DM * uint32
    | DMR of DM * (DM ref) * TraceOp * (uint32 ref) * uint32

    member d.P =
        match d with
        | DM(_) -> d
        | DMF(ap,_,_) -> ap
        | DMR(ap,_,_,_,_) -> ap
    member d.T =
        match d with
        | DM(_) -> DM Array2D.empty
        | DMF(_,at,_) -> at
        | DMR(_,_,_,_,_) -> failwith "Cannot get tangent value of DMR."
    member d.A
        with get() =
            match d with
            | DM(_) -> DM Array2D.empty
            | DMF(_,_,_) -> failwith "Cannot get adjoint value of DMF."
            | DMR(_,a,_,_,_) -> !a
        and set(v) =
            match d with
            | DM(_) -> ()
            | DMF(_,_,_) -> failwith "Cannot set adjoint value of DMF."
            | DMR(_,a,_,_,_) -> a := v
    member d.F
        with get() =
            match d with
            | DM(_) -> failwith "Cannot get fan-out value of DM."
            | DMF(_,_,_) -> failwith "Cannot get fan-out value of DMF."
            | DMR(_,_,_,f,_) -> !f
        and set(v) =
            match d with
            | DM(_) -> failwith "Cannot set fan-out value of DM."
            | DMF(_,_,_) -> failwith "Cannot set fan-out value of DMF."
            | DMR(_,_,_,f,_) -> f := v
    member d.Copy() =
        match d with
        | DM(ap) -> DM(Array2D.copy ap)
        | DMF(ap,at,ai) -> DMF(ap.Copy(), at.Copy(), ai)
        | DMR(ap,aa,at,af,ai) -> DMR(ap.Copy(), ref ((!aa).Copy()), at, ref (!af), ai)
    member d.Length =
        match d with
        | DM(ap) -> Array2D.length1 ap, Array2D.length2 ap
        | DMF(ap,_,_) -> ap.Length
        | DMR(ap,_,_,_,_) -> ap.Length
    member d.Rows =
        match d with
        | DM(ap) -> Array2D.length1 ap
        | DMF(ap,_,_) -> ap.Rows
        | DMR(ap,_,_,_,_) -> ap.Rows
    member d.Cols =
        match d with
        | DM(ap) -> Array2D.length2 ap
        | DMF(ap,_,_) -> ap.Cols
        | DMR(ap,_,_,_,_) -> ap.Cols
    member d.Item
        with get (i, j) =
            match d with
            | DM(ap) -> D(ap.[i, j])
            | DMF(ap,at,ai) -> DF(ap.[i,j], at.[i,j], ai)
            | DMR(ap,_,_,_,ai) -> DR(ap.[i,j], ref (D 0.), Item_DM(d, i, j), ref 0u, ai)

    member d.GetSlice(rowStart, rowFinish, colStart, colFinish) =
        let rowStart = defaultArg rowStart 0
        let rowFinish = defaultArg rowFinish (d.Rows - 1)
        let colStart = defaultArg colStart 0
        let colFinish = defaultArg colFinish (d.Cols - 1)
        match d with
        | DM(ap) -> DM(ap.[rowStart..rowFinish, colStart..colFinish])
        | DMF(ap,at,ai) -> DMF(ap.[rowStart..rowFinish, colStart..colFinish], at.[rowStart..rowFinish, colStart..colFinish], ai)
        | DMR(ap,_,_,_,ai) -> DMR(ap.[rowStart..rowFinish, colStart..colFinish], ref DM.Zero, Slice_DM(d, rowStart, rowFinish), ref 0u, ai)
    member d.GetSlice(row, colStart, colFinish) =
        let colStart = defaultArg colStart 0
        let colFinish = defaultArg colFinish (d.Cols - 1)
        match d with
        | DM(ap) -> DV(ap.[row, colStart..colFinish])
        | DMF(ap,at,ai) -> DVF(ap.[row, colStart..colFinish], at.[row, colStart..colFinish], ai)
        | DMR(ap,_,_,_,ai) -> DVR(ap.[row, colStart..colFinish], ref DV.Zero, SliceRow_DM(d, row, colStart), ref 0u, ai)
    member d.GetSlice(rowStart, rowFinish, col) =
        let rowStart = defaultArg rowStart 0
        let rowFinish = defaultArg rowFinish (d.Rows - 1)
        match d with
        | DM(ap) -> DV(ap.[rowStart..rowFinish, col])
        | DMF(ap,at,ai) -> DVF(ap.[rowStart..rowFinish, col], at.[rowStart..rowFinish, col], ai)
        | DMR(ap,_,_,_,ai) -> DVR(ap.[rowStart..rowFinish, col], ref DV.Zero, SliceCol_DM(d, rowStart, col), ref 0u, ai)

    member d.Sum() =
        match d with
        | DM(ap) -> D(NonBLAS.m_sum(ap))
        | DMF(ap,at,ai) -> DF(ap.Sum(), at.Sum(), ai)
        | DMR(ap,_,_,_,ai) -> DR(ap.Sum(), ref (D 0.), Sum_DM(d), ref 0u, ai)
    member d.Transpose() =
        match d with
        | DM(ap) -> DM(NonBLAS.m_transpose(ap))
        | DMF(ap,at,ai) -> DMF(ap.Transpose(), at.Transpose(), ai)
        | DMR(ap,_,_,_,ai) -> DMR(ap.Transpose(), ref DM.Zero, Transpose_DM(d), ref 0u, ai)
    member d.GetRows() =
        seq {for i = 0 to d.Rows - 1 do yield d.[i,*]}
    member d.GetCols() =
        seq {for j = 0 to d.Cols - 1 do yield d.[*,j]}

    static member Zero = DM Array2D.empty
    static member ZeroMN m n = DM (Array2D.zeroCreate m n)
    static member op_Explicit(d:DM):float[,] =
        match d with
        | DM(ap) -> ap
        | DMF(ap,_,_) -> DM.op_Explicit(ap)
        | DMR(ap,_,_,_,_) -> DM.op_Explicit(ap)
    static member op_Explicit(d) = DM(d)
    static member ofArray2D (a:D[,]) =
        // TODO: check to ensure that all elements in the array are of the same type (D, DF, or DR) and have the same nesting tag
        match a.[0, 0] with
        | D(_) -> DM (a |> Array2D.map float)
        | DF(_,_,ai) ->
            let ap = a |> Array2D.map (fun x -> x.P)
            let at = a |> Array2D.map (fun x -> x.T)
            DMF(DM.ofArray2D(ap), DM.ofArray2D(at), ai)
        | DR(_,_,_,_,ai) ->
            let ap = a |> Array2D.map (fun x -> x.P)
            DMR(DM.ofArray2D(ap), ref DM.Zero, Make_DM_ofD(a), ref 0u, ai)
    // Creates a matrix with `m` rows from array `a`, filling columns from left to right and rows from top to bottom. The number of columns will be deduced from `m` and the length of `a`. The length of `a` must be an integer multiple of `m`.
    static member ofArray (m:int, a:D[]) =
        let n = a.Length / m
        Array2D.init m n (fun i j -> a.[i * n + j]) |> DM.ofArray2D
    static member ofRows (s:seq<DV>) = 
        // TODO: check to ensure that all elements in the array are of the same type (D, DF, or DR) and have the same nesting tag
        match Seq.head s with
        | DV(_) ->
            s |> Seq.map DV.op_Explicit |> array2D |> DM
        | DVF(_,_,ai) ->
            let ap = s |> Seq.map (fun x -> x.P)
            let at = s |> Seq.map (fun x -> x.T)
            DMF(DM.ofRows(ap), DM.ofRows(at), ai)
        | DVR(_,_,_,_,ai) ->
            let ap = s |> Seq.map (fun x -> x.P)
            DMR(DM.ofRows(ap), ref DM.Zero, Make_DM_ofDV(s |> Seq.toArray), ref 0u, ai)

    static member inline Op_DM_DM (a, ff, fd, df, r) =
        match a with
        | DM(ap)                      -> DM(ff(ap))
        | DMF(ap, at, ai)             -> let cp = fd(ap) in DMF(cp, df(cp, ap, at), ai)
        | DMR(ap,_,_,_,ai)            -> DMR(fd(ap), ref DM.Zero, r(a), ref 0u, ai)

    static member inline Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | DM(bp)                  -> DM(ff(ap, bp))
            | DMF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DMR(bp,  _,  _,  _, bi) -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi)
        | DMF(ap, at, ai) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DMR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap,  _,  _,  _, ai) ->
            match b with
            | DM(_)                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DMR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> DMR(fd(ap, bp), ref DM.Zero, r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi

    static member inline Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | D(bp)                   -> DM(ff(ap, bp))
            | DF(bp, bt, bi)          -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DR(bp,  _,  _,  _, bi)  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi)
        | DMF(ap, at, ai) ->
            match b with
            | D(_)                    -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap,  _,  _,  _, ai) ->
            match b with
            | D(_)                    -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> DMR(fd(ap, bp), ref DM.Zero, r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> DMR(fd(a, bp), ref DM.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> DMR(fd(ap, b), ref DM.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi

    static member inline Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | DV(bp)                  -> DV(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp,  _,  _,  _, bi) -> DVR(fd(a, bp), ref DV.Zero, r_c_d(a, b), ref 0u, bi)
        | DMF(ap, at, ai) ->
            match b with
            | DV(_)                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | -1                  -> DVR(fd(a, bp), ref DV.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap,  _,  _,  _, ai) ->
            match b with
            | DV(_)                   -> DVR(fd(ap, b), ref DV.Zero, r_d_c(a, b), ref 0u, ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> DVR(fd(ap, b), ref DV.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp,  _,  _,  _, bi) ->
                match compare ai bi with
                | 0                   -> DVR(fd(ap, bp), ref DV.Zero, r_d_d(a, b), ref 0u, ai) // ai = bi
                | -1                  -> DVR(fd(a, bp), ref DV.Zero, r_c_d(a, b), ref 0u, bi) // ai < bi
                | _                   -> DVR(fd(ap, b), ref DV.Zero, r_d_c(a, b), ref 0u, ai) // ai > bi

    static member (+) (a:DM, b:DM) =
        let inline ff(a, b) = NonBLAS.m_add(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DM_DM(a, b)
        let inline r_d_c(a, b) = Add_DM_DMCons(a)
        let inline r_c_d(a, b) = Add_DM_DMCons(b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:DM, b:DM) =
        let inline ff(a, b) = NonBLAS.m_sub(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DM_DM(a, b)
        let inline r_d_c(a, b) = Sub_DM_DMCons(a)
        let inline r_c_d(a, b) = Sub_DMCons_DM(b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // Matrix multiplication
    static member (*) (a:DM, b:DM) =
        let inline ff(a, b) = OpenBLAS.m_mul(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_DM(a, b)
        let inline r_d_c(a, b) = Mul_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_DM(a, b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // Element-wise (Hadamard, Schur) product
    static member (.*) (a:DM, b:DM) =
        let inline ff(a, b) = NonBLAS.m_mul_hadamard(a, b)
        let inline fd(a, b) = a .* b
        let inline df_da(cp, ap, at) = at .* b
        let inline df_db(cp, bp, bt) = a .* bt
        let inline df_dab(cp, ap, at, bp, bt) = (at .* bp) + (ap .* bt)
        let inline r_d_d(a, b) = Mul_Had_DM_DM(a, b)
        let inline r_d_c(a, b) = Mul_Had_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Mul_Had_DM_DMCons(b, a)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:DM, b:DV) =
        let inline ff(a, b) = OpenBLAS.mv_mul(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_DV(a, b)
        let inline r_d_c(a, b) = Mul_DM_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_DV(a, b)
        DM.Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:DV, b:DM) =
        // TODO: reimplement faster
        b.Transpose() * a

    static member (*) (a:DM, b:D) =
        let inline ff(a, b) = OpenBLAS.m_scale(b, a)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_D(a, b)
        let inline r_d_c(a, b) = Mul_DM_DCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_D(a, b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:D, b:DM) = b * a

    static member (~-) (a:DM) =
        let inline ff(a) = OpenBLAS.m_scale(-1., a)
        let inline fd(a) = -a
        let inline df(cp, ap, at) = -at
        let inline r(a) = Neg_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Solve (a:DM, b:DV) =
        let inline ff(a, b) = match OpenBLAS.mv_solve(a, b) with Some(x) -> x | _ -> ErrorMessages.invalidArgSolve()
        let inline fd(a, b) = DM.Solve(a, b)
        let inline df_da(cp, ap, at) = DM.Solve(ap, -at * cp) // cp = DM.Solve(ap, b)
        let inline df_db(cp, bp, bt) = DM.Solve(a, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.Solve(ap, bt - at * cp) // cp = DM.Solve(ap, bp)
        let inline r_d_d(a, b) = Mul_DM_DV(a, b)
        let inline r_d_c(a, b) = Mul_DM_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_DV(a, b)
        DM.Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member SolveSymmetric (a:DM, b:DV) =
        let inline ff(a, b) = match OpenBLAS.mv_solve_symmetric(a, b) with Some(x) -> x | _ -> ErrorMessages.invalidArgSolve()
        let inline fd(a, b) = DM.SolveSymmetric(a, b)
        let inline df_da(cp, ap, at) = DM.SolveSymmetric(ap, -at * cp) // cp = DM.Solve(ap, b)
        let inline df_db(cp, bp, bt) = DM.SolveSymmetric(a, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.SolveSymmetric(ap, bt - at * cp) // cp = DM.Solve(ap, bp)
        let inline r_d_d(a, b) = Mul_DM_DV(a, b)
        let inline r_d_c(a, b) = Mul_DM_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_DV(a, b)
        DM.Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member AddItem (a:DM, i:int, j:int, b:D) =
        let inline ff(a, b) = let aa = Array2D.copy a in aa.[i, j] <- aa.[i, j] + b; aa
        let inline fd(a, b) = DM.AddItem(a, i, j, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.AddItem(DM.ZeroMN a.Rows a.Cols, i, j, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.AddItem(at, i, j, bt)
        let inline r_d_d(a, b) = AddItem_DM_D(a, i, j, b)
        let inline r_d_c(a, b) = AddItem_DM_DCons(a)
        let inline r_c_d(a, b) = AddItem_DMCons_D(i, j, b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)
    
    static member AddSubMatrix (a:DM, i:int, j:int, b:DM) =
        let inline ff(a:float[,], bb:float[,]) = 
            let aa = Array2D.copy a 
            for ii = 0 to b.Rows - 1 do
                for jj = 0 to b.Cols - 1 do
                    aa.[i + ii, j + jj] <- bb.[ii, jj]
            aa
        let inline fd(a, b) = DM.AddSubMatrix(a, i, j, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.AddSubMatrix(DM.ZeroMN a.Rows a.Cols, i, j, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.AddSubMatrix(at, i, j, bt)
        let inline r_d_d(a, b) = AddSubMatrix_DM_DM(a, i, j, b)
        let inline r_d_c(a, b) = AddSubMatrix_DM_DMCons(a)
        let inline r_c_d(a, b) = AddSubMatrix_DMCons_DM(i, j, b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

and TraceOp =
    | Add_D_D                of D * D
    | Add_D_DCons            of D
    | Sub_D_D                of D * D
    | Sub_D_DCons            of D
    | Sub_DCons_D            of D
    | Mul_D_D                of D * D
    | Mul_D_DCons            of D * D
    | Div_D_D                of D * D
    | Div_D_DCons            of D * D
    | Div_DCons_D            of D * D
    | Pow_D_D                of D * D
    | Pow_D_DCons            of D * D
    | Pow_DCons_D            of D * D
    | Atan2_D_D              of D * D
    | Atan2_D_DCons          of D * D
    | Atan2_DCons_D          of D * D
    | Log_D                  of D
    | Log10_D                of D
    | Exp_D                  of D
    | Sin_D                  of D
    | Cos_D                  of D
    | Tan_D                  of D
    | Neg_D                  of D
    | Sqrt_D                 of D
    | Sinh_D                 of D
    | Cosh_D                 of D
    | Tanh_D                 of D
    | Asin_D                 of D
    | Acos_D                 of D
    | Atan_D                 of D
    | Abs_D                  of D
    | Sign_D                 of D
    | Floor_D                of D
    | Ceil_D                 of D
    | Round_D                of D
    | Mul_Dot_DV_DV          of DV * DV
    | Mul_Dot_DV_DVCons      of DV * DV
    | Sum_DV                 of DV
    | L1Norm_DV              of DV
    | L2NormSq_DV            of DV
    | L2Norm_DV              of DV
    | Item_DV                of DV * int
    | Sum_DM                 of DM
    | Item_DM                of DM * int * int
    
    | Add_DV_DV              of DV * DV
    | Add_DV_DVCons          of DV
    | Add_DV_D               of DV * D
    | Add_DV_DCons           of DV
    | Add_DVCons_D           of D
    | Sub_DV_DV              of DV * DV
    | Sub_DV_DVCons          of DV
    | Sub_DVCons_DV          of DV
    | Sub_DV_D               of DV * D
    | Sub_DV_DCons           of DV
    | Sub_DVCons_D           of D
    | Sub_D_DV               of D * DV
    | Sub_D_DVCons           of D
    | Sub_DCons_DV           of DV
    | Mul_Had_DV_DV          of DV * DV
    | Mul_Had_DV_DVCons      of DV * DV
    | Mul_DV_D               of DV * D
    | Mul_DV_DCons           of DV * D
    | Mul_DVCons_D           of DV * D
    | Mul_DM_DV              of DM * DV
    | Mul_DM_DVCons          of DM * DV
    | Mul_DMCons_DV          of DM * DV
    | Div_Had_DV_DV          of DV * DV
    | Div_Had_DV_DVCons      of DV * DV
    | Div_Had_DVCons_DV      of DV * DV
    | Div_DV_D               of DV * D
    | Div_DV_DCons           of DV * D
    | Div_DVCons_D           of DV * D
    | Div_D_DV               of D * DV
    | Div_D_DVCons           of D * DV
    | Div_DCons_DV           of D * DV
    | Exp_DV                 of DV
    | Neg_DV                 of DV
    | Abs_DV                 of DV
    | Sign_DV                of DV
    | Make_DV                of D[]
    | SliceRow_DM            of DM * int * int
    | SliceCol_DM            of DM * int * int
    | Solve_DM_DV            of DM * DV
    | Solve_DM_DVCons        of DM * DV
    | Solve_DMCons_DV        of DM * DV
    | Append_DV_DV           of DV * DV
    | Append_DV_DVCons       of DV
    | Append_DVCons_DV       of DV
    | Split_DV               of DV * int
    | AddItem_DV_D           of DV * int * D
    | AddItem_DV_DCons       of DV
    | AddItem_DVCons_D       of int * D
    | AddSubVector_DV_DV     of DV * int * DV
    | AddSubVector_DV_DVCons of DV
    | AddSubVector_DVCons_DV of int * DV
    | Slice_DV               of DV * int
        
    | Add_DM_DM              of DM * DM
    | Add_DM_DMCons          of DM
    | Sub_DM_DM              of DM * DM
    | Sub_DM_DMCons          of DM
    | Sub_DMCons_DM          of DM
    | Mul_DM_DM              of DM * DM
    | Mul_DM_DMCons          of DM * DM
    | Mul_DMCons_DM          of DM * DM
    | Mul_Had_DM_DM          of DM * DM
    | Mul_Had_DM_DMCons      of DM * DM
    | Mul_DM_D               of DM * D
    | Mul_DM_DCons           of DM * D
    | Mul_DMCons_D           of DM * D
    | Mul_Out_DV_DV          of DV * DV
    | Mul_Out_DV_DVCons      of DV * DV
    | Mul_Out_DVCons_DV      of DV * DV
    | Neg_DM                 of DM
    | Transpose_DM           of DM
    | Make_DM_ofD            of D[,]
    | Make_DM_ofDV           of DV[]
    | AddItem_DM_D           of DM * int * int * D
    | AddItem_DM_DCons       of DM
    | AddItem_DMCons_D       of int * int * D
    | AddSubMatrix_DM_DM     of DM * int * int * DM
    | AddSubMatrix_DM_DMCons of DM
    | AddSubMatrix_DMCons_DM of int * int * DM
    | Slice_DM               of DM * int * int
    | RowMatrix_DV           of DV
    
    | Noop


[<RequireQualifiedAccess>]
module Vector =
    let inline primal (v:DV) = v.P
    let inline tangent (v:DV) = v.T
    let inline adjoint (v:DV) = v.A
    let inline primalTangent (v:DV) = v.P, v.T
    let inline makeForward i t p = DVF(p, t, i)
    let inline makeReverse i p = DVR(p, ref (DV.ZeroN p.Length), Noop, ref 0u, i)

    let inline ofArray a = DV.ofArray(a)
    let inline toArray (v:DV) = v.ToArray()
    let inline toRowMatrix (v:DV) = v.ToRowMatrix()
    let inline toColMatrix (v:DV) = v.ToColMatrix()
    let inline create n v = DV.ofArray(Array.create n v)
    let inline zeroCreate n = DV.ZeroN n
    let inline init n f = DV.ofArray(Array.init n f)
    let inline copy (v:DV) = v.Copy()
    let inline l1norm (v:DV) = v.L1Norm()
    let inline l2norm (v:DV) = v.L2Norm()
    let inline l2normSq (v:DV) = v.L2NormSq()
    let inline norm (v:DV) = v.L2Norm()
    let inline normSq(v:DV) = v.L2NormSq()
    // TODO: implement supNorm (infinity norm, with BLAS IDAMAX)
    let inline append (v1:DV) (v2:DV) = DV.Append(v1, v2)
    let inline prepend (v1:DV) (v2:DV) = DV.Append(v2, v1)
    let inline split (n:seq<int>) (v:DV) = v.Split(n)
    let inline sum (v:DV) = v.Sum()

[<RequireQualifiedAccess>]
module Matrix =
    let inline primal (m:DM) = m.P
    let inline tangent (m:DM) = m.T
    let inline adjoint (m:DM) = m.A
    let inline primalTangent (m:DM) = m.P, m.T
    let inline makeForward i t p = DMF(p, t, i)
    let inline makeReverse i p = DMR(p, ref DM.Zero, Noop, ref 0u, i)

    let inline ofArray2D a = DM.ofArray2D(a)
    let inline ofArray m a = DM.ofArray(m, a)
    let inline ofRows s = DM.ofRows(s)
    let inline transpose (m:DM) = m.Transpose()
    let inline ofCols (s:seq<DV>) = s |> ofRows |> transpose
    let inline toRows (m:DM) = m.GetRows()
    let inline toCols (m:DM) = m.GetCols()
    let inline toVector (m:DM) = m.GetRows () |> Seq.fold Vector.append DV.Zero
    let inline ofVector (m:int) (v:DV) = let n = v.Length / m in v |> Vector.split (Array.create m n) |> DM.ofRows
    let inline create m n v = ofArray2D(Array2D.create m n v)
    let inline zeroCreate m n = DM.ZeroMN m n
    let inline init m n f = ofArray2D(Array2D.init m n f)
    let inline copy (m:DM) = m.Copy()
    let inline solve (m:DM) (v:DV) = DM.Solve(m, v)
    let inline solveSymmetric (m:DM) (v:DV) = DM.SolveSymmetric(m, v)
    let inline appendRow (v:DV) (m:DM) = let rows = m |> toRows in Seq.append rows (seq [v]) |> ofRows
    let inline prependRow (v:DV) (m:DM) = let rows = m |> toRows in Seq.append (seq [v]) rows |> ofRows
    let inline appendCol (v:DV) (m:DM) = let cols = m |> toCols in Seq.append cols (seq [v]) |> ofCols
    let inline prependCol (v:DV) (m:DM) = let cols = m |> toCols in Seq.append (seq [v]) cols |> ofCols


[<AutoOpen>]
module Util =
    let inline standardBasis (n:int) (i:int) = DV(Array.init n (fun j -> if i = j then 1. else 0.))


[<AutoOpen>]
module DOps =
    let inline convert (v:^a) : ^b = ((^a or ^b) : (static member op_Explicit: ^a -> ^b) v)
    let inline vector (v:seq<D>) = v |> Seq.toArray |> Vector.ofArray
    let inline matrix (m:seq<seq<D>>) = m |> array2D |> Matrix.ofArray2D
    let inline makeForward i t p = DF(p, t, i)
    let inline makeReverse i p = DR(p, ref (D 0.), Noop, ref 0u, i)
    let inline primal (d:D) = d.P
    let inline tangent (d:D) = d.T
    let inline adjoint (d:D) = d.A
    let inline primalTangent (d:D) = d.P, d.T
    let rec reversePush (v:obj) (d:obj) =
        match d with
        | :? D as d ->
            match d with
            | DR(_,_,o,_,_) ->
                d.A <- d.A + (v :?> D)
                d.F <- d.F - 1u
                if d.F = 0u then
                    match o with
                    | Add_D_D(a, b) -> reversePush d.A a; reversePush d.A b
                    | Add_D_DCons(a) -> reversePush d.A a
                    | Sub_D_D(a, b) -> reversePush d.A a; reversePush -d.A b
                    | Sub_D_DCons(a) -> reversePush d.A a
                    | Sub_DCons_D(a) -> reversePush -d.A a
                    | Mul_D_D(a, b) -> reversePush (d.A * b.P) a; reversePush (d.A * a.P) b
                    | Mul_D_DCons(a, cons) -> reversePush (d.A * cons) a
                    | Div_D_D(a, b) -> reversePush (d.A / b.P) a; reversePush (d.A * (-a.P / (b.P * b.P))) b
                    | Div_D_DCons(a, cons) -> reversePush (d.A / cons) a
                    | Div_DCons_D(cons, b) -> reversePush (d.A * (-cons / (b.P * b.P))) b
                    | Pow_D_D(a, b) -> reversePush (d.A * (a.P ** (b.P - D 1.)) * b.P) a; reversePush (d.A * (a.P ** b.P) * log a.P) b
                    | Pow_D_DCons(a, cons) -> reversePush (d.A * (a.P ** (cons - D 1.)) * cons) a
                    | Pow_DCons_D(cons, b) -> reversePush (d.A * (cons ** b.P) * log cons) b
                    | Atan2_D_D(a, b) -> let denom = a.P * a.P + b.P * b.P in reversePush (d.A * b.P / denom) a; reversePush (d.A * (-a.P) / denom) b
                    | Atan2_D_DCons(a, cons) -> reversePush (d.A * cons / (a.P * a.P + cons * cons)) a
                    | Atan2_DCons_D(cons, b) -> reversePush (d.A * (-cons) / (cons * cons + b.P * b.P)) b
                    | Log_D(a) -> reversePush (d.A / d.P) a
                    | Log10_D(a) -> reversePush (d.A / (a.P * log10val)) a
                    | Exp_D(a) -> reversePush (d.A * d.P) a // d.P = exp a.P
                    | Sin_D(a) -> reversePush (d.A * cos a.P) a
                    | Cos_D(a) -> reversePush (d.A * (-sin a.P)) a
                    | Tan_D(a) -> let seca = D 1. / cos a.P in reversePush (d.A * seca * seca) a
                    | Neg_D(a) -> reversePush -d.A a
                    | Sqrt_D(a) -> reversePush (d.A / (D 2. * d.P)) a // d.P = sqrt a.P
                    | Sinh_D(a) -> reversePush (d.A * cosh a.P) a
                    | Cosh_D(a) -> reversePush (d.A * sinh a.P) a
                    | Tanh_D(a) -> let secha = D 1. / cosh a.P in reversePush (d.A * secha * secha) a
                    | Asin_D(a) -> reversePush (d.A / sqrt (D 1. - a.P * a.P)) a
                    | Acos_D(a) -> reversePush (-d.A / sqrt (D 1. - a.P * a.P)) a
                    | Atan_D(a) -> reversePush (d.A / (D 1. + a.P * a.P)) a
                    | Abs_D(a) -> reversePush (d.A * float (sign (float a.P))) a
                    | Sign_D(_) -> ()
                    | Floor_D(_) -> ()
                    | Ceil_D(_) -> ()
                    | Round_D(_) -> ()
                    | Mul_Dot_DV_DV(a, b) -> reversePush (d.A * b.P) a; reversePush (d.A * a.P) b
                    | Mul_Dot_DV_DVCons(a, cons) -> reversePush (d.A * cons) a
                    | Sum_DV(a) -> reversePush (Vector.create a.Length d.A) a
                    | L1Norm_DV(a) -> reversePush (d.A * DV.Sign a.P) a
                    | L2NormSq_DV(a) -> reversePush (d.A * (D 2.) * a.P) a
                    | L2Norm_DV(a) -> reversePush ((d.A / d.P) * a.P) a
                    | Item_DV(a, i) -> a.A <- DV.AddItem(a.A, i, d.A); reversePush DV.Zero a
                    | Sum_DM(a) -> reversePush (Matrix.create a.Rows a.Cols d.A) a
                    | Item_DM(a, i, j) -> a.A <- DM.AddItem(a.A, i, j, d.A); reversePush DM.Zero a
                    | _ -> ()
            | _ -> ()
        | :? DV as d ->
            match d with
            | DVR(_,_,o,_,_) ->
                d.A <- d.A + (v :?> DV)
                d.F <- d.F - 1u
                if d.F = 0u then
                    match o with
                    | Add_DV_DV(a, b) -> reversePush d.A a; reversePush d.A b
                    | Add_DV_DVCons(a) -> reversePush d.A a
                    | Add_DV_D(a, b) -> reversePush d.A a; reversePush (d.A.Sum()) b
                    | Add_DV_DCons(a) -> reversePush d.A a
                    | Add_DVCons_D(b) -> reversePush (d.A.Sum()) b
                    | Sub_DV_DV(a, b) -> reversePush d.A a; reversePush -d.A b
                    | Sub_DV_DVCons(a) -> reversePush d.A a
                    | Sub_DVCons_DV(a) -> reversePush -d.A a
                    | Sub_DV_D(a, b) -> reversePush d.A a; reversePush -(d.A.Sum()) b
                    | Sub_DV_DCons(a) -> reversePush d.A a
                    | Sub_DVCons_D(b) -> reversePush -(d.A.Sum()) b
                    | Sub_D_DV(a, b) -> reversePush (d.A.Sum()) a; reversePush d.A b
                    | Sub_D_DVCons(a) -> reversePush (d.A.Sum()) a
                    | Sub_DCons_DV(b) -> reversePush d.A b
                    | Mul_Had_DV_DV(a, b) -> reversePush (d.A .* b.P) a; reversePush (d.A .* a.P) b
                    | Mul_Had_DV_DVCons(a, cons) -> reversePush (d.A .* cons) a
                    | Mul_DV_D(a, b) -> reversePush (d.A * b.P) a; reversePush (d.A * a.P) b
                    | Mul_DV_DCons(a, cons) -> reversePush (d.A * cons) a
                    | Mul_DVCons_D(cons, b) -> reversePush (d.A * cons) b
                    | Mul_DM_DV(a, b) -> reversePush (d.A &* b.P) a; reversePush (a.P.Transpose() * d.A) b
                    | Mul_DM_DVCons(a, cons) -> reversePush (d.A &* cons) a
                    | Mul_DMCons_DV(cons, b) -> reversePush (cons.Transpose() * d.A) b
                    | Div_Had_DV_DV(a, b) -> reversePush (d.A ./ b.P) a; reversePush (d.A .* (-a.P ./ (b.P .* b.P))) b
                    | Div_Had_DV_DVCons(a, cons) -> reversePush (d.A ./ cons) a
                    | Div_Had_DVCons_DV(cons, b) -> reversePush (d.A .* (-cons ./ (b.P .* b.P))) b
                    | Div_DV_D(a, b) -> reversePush (d.A / b.P) a; reversePush (d.A * (-a.P / (b.P * b.P))) b
                    | Div_DV_DCons(a, cons) -> reversePush (d.A / cons) a
                    | Div_DVCons_D(cons, b) -> reversePush (d.A * (-cons / (b.P * b.P))) b
                    | Div_D_DV(a, b) -> reversePush ((d.A ./ b.P).Sum()) a; reversePush (d.A .* (-a.P / (b.P .* b.P))) b
                    | Div_D_DVCons(a, cons) -> reversePush ((d.A ./ cons).Sum()) a
                    | Div_DCons_DV(cons, b) -> reversePush (d.A .* (-cons / (b.P .* b.P))) b
                    | Exp_DV(a) -> reversePush (d.A .* d.P) a // d.P = exp a.P
                    | Neg_DV(a) -> reversePush -d.A a
                    | Abs_DV(a) -> reversePush (d.A .* DV.Sign a.P) a
                    | Sign_DV(a) -> reversePush DV.Zero a
                    | Make_DV(a) -> a |> Array.iteri (fun i v -> reversePush d.A.[i] v)
                    | SliceRow_DM(a, i, j) ->
                        a.A <- DM.AddSubMatrix(a.A, i, j, d.A.ToRowMatrix())
                        reversePush DM.Zero a
                    | SliceCol_DM(a, i, j) ->
                        a.A <- DM.AddSubMatrix(a.A, i, j, d.A.ToColMatrix())
                        reversePush DM.Zero a
                    | Solve_DM_DV(a, b) -> let ba = DM.Solve(a.Transpose(), d.A) in reversePush (-ba &* d.A) a; reversePush (ba) b
                    | Solve_DM_DVCons(a, cons) -> let ba = DM.Solve(a.Transpose(), d.A) in reversePush (-ba &* d.A) a
                    | Solve_DMCons_DV(cons, b) -> let ba = DM.Solve(cons.Transpose(), d.A) in reversePush ba b
                    | Append_DV_DV(a, b) ->
                        a.A <- a.A + d.A.[..(a.Length - 1)]
                        reversePush DV.Zero a
                        b.A <- b.A + d.A.[a.Length..]
                        reversePush DV.Zero b
                    | Append_DV_DVCons(a) ->
                        a.A <- a.A + d.A.[..(a.Length - 1)]
                        reversePush DV.Zero a
                    | Append_DVCons_DV(b) ->
                        b.A <- b.A + d.A.[(d.Length - b.Length)..]
                        reversePush DV.Zero b
                    | Split_DV(a, i) ->
                        a.A <- DV.AddSubVector(a.A, i, d.A)
                        reversePush DV.Zero a
                    | AddItem_DV_D(a, i, b) -> reversePush d.A a; reversePush (d.A.[i]) b
                    | AddItem_DV_DCons(a) -> reversePush d.A a
                    | AddItem_DVCons_D(i, b) -> reversePush d.A.[i] b
                    | AddSubVector_DV_DV(a, i, b) -> reversePush d.A a; reversePush (d.A.[i..(i + b.Length - 1)]) b
                    | AddSubVector_DV_DVCons(a) -> reversePush d.A a
                    | AddSubVector_DVCons_DV(i, b) -> reversePush (d.A.[i..(i + b.Length - 1)]) b
                    | Slice_DV(a, i) ->
                        a.A <- DV.AddSubVector(a.A, i, d.A)
                        reversePush DV.Zero a
                    | _ -> ()
            | _ -> ()
        | :? DM as d ->
            match d with
            | DMR(_,_,o,_,_) ->
                d.A <- d.A + (v :?> DM)
                d.F <- d.F - 1u
                if d.F = 0u then
                    match o with
                    | Add_DM_DM(a, b) -> reversePush d.A a; reversePush d.A b
                    | Add_DM_DMCons(a) -> reversePush d.A a
                    | Sub_DM_DM(a, b) -> reversePush d.A a; reversePush -d.A b
                    | Sub_DM_DMCons(a) -> reversePush d.A a
                    | Sub_DMCons_DM(a) -> reversePush -d.A a
                    | Mul_DM_DM(a, b) -> reversePush (d.A * b.P.Transpose()) a; reversePush (a.P.Transpose() * d.A) b
                    | Mul_DM_DMCons(a, cons) -> reversePush (d.A * cons.Transpose()) a
                    | Mul_DMCons_DM(cons, b) -> reversePush (cons.Transpose() * d.A) b
                    | Mul_Had_DM_DM(a, b) -> reversePush (d.A .* b.P) a; reversePush (d.A .* a.P) b
                    | Mul_Had_DM_DMCons(a, cons) -> reversePush (d.A .* cons) a
                    | Mul_DM_D(a, b) -> reversePush (d.A * b.P) a; reversePush ((d.A .* a.P).Sum()) b
                    | Mul_DM_DCons(a, cons) -> reversePush (d.A * cons) a
                    | Mul_DMCons_D(cons, b) -> reversePush ((d.A .* cons).Sum()) b
                    | Mul_Out_DV_DV(a, b) -> reversePush (d.A * b.P) a; reversePush (d.A.Transpose() * a.P) b
                    | Mul_Out_DV_DVCons(a, cons) -> reversePush (d.A * cons) a
                    | Mul_Out_DVCons_DV(cons, b) -> reversePush (d.A.Transpose() * cons) b
                    | Transpose_DM(a) -> reversePush (d.A.Transpose()) a
                    | Make_DM_ofD(a) -> a |> Array2D.iteri (fun i j v -> reversePush d.A.[i, j] v)
                    | Make_DM_ofDV(a) -> a |> Array.iteri (fun i v -> reversePush d.A.[i, *] v)
                    | AddItem_DM_D(a, i, j, b) -> reversePush d.A a; reversePush (d.A.[i, j]) b
                    | AddItem_DM_DCons(a) -> reversePush d.A a
                    | AddItem_DMCons_D(i, j, b) -> reversePush d.A.[i, j] b
                    | AddSubMatrix_DM_DM(a, i, j, b) -> reversePush d.A a; reversePush (d.A.[i..(i + b.Rows - 1), j..(j + b.Cols - 1)]) b
                    | AddSubMatrix_DM_DMCons(a) -> reversePush d.A a
                    | AddSubMatrix_DMCons_DM(i, j, b) -> reversePush (d.A.[i..(i + b.Rows - 1), j..(j + b.Cols - 1)]) b
                    | Slice_DM(a, i, j) ->
                        a.A <- DM.AddSubMatrix(a.A, i, j, d.A)
                        reversePush DM.Zero a
                    | RowMatrix_DV(a) -> reversePush (d.A.[0,*]) a
                    | _ -> ()
            | _ -> ()
        | _ -> ()
    let rec reverseReset (d:obj) =
        match d with
        | :? D as d ->
            match d with
            | DR(_,_,o,_,_) ->
                d.A <- D 0.
                d.F <- d.F + 1u
                if d.F = 1u then
                    match o with
                    | Add_D_D(a, b) -> reverseReset a; reverseReset b
                    | Add_D_DCons(a) -> reverseReset a
                    | Sub_D_D(a, b) -> reverseReset a; reverseReset b
                    | Sub_D_DCons(a) -> reverseReset a
                    | Sub_DCons_D(a) -> reverseReset a
                    | Mul_D_D(a, b) -> reverseReset a; reverseReset b
                    | Mul_D_DCons(a, _) -> reverseReset a
                    | Div_D_D(a, b) -> reverseReset a; reverseReset b
                    | Div_D_DCons(a, _) -> reverseReset a
                    | Div_DCons_D(_, b) -> reverseReset b
                    | Pow_D_D(a, b) -> reverseReset a; reverseReset b
                    | Pow_D_DCons(a, _) -> reverseReset a
                    | Pow_DCons_D(_, b) -> reverseReset b
                    | Atan2_D_D(a, b) -> reverseReset a; reverseReset b
                    | Atan2_D_DCons(a, _) -> reverseReset a
                    | Atan2_DCons_D(_, b) -> reverseReset b
                    | Log_D(a) -> reverseReset a
                    | Log10_D(a) -> reverseReset a
                    | Exp_D(a) -> reverseReset a
                    | Sin_D(a) -> reverseReset a
                    | Cos_D(a) -> reverseReset a
                    | Tan_D(a) -> reverseReset a
                    | Neg_D(a) -> reverseReset a
                    | Sqrt_D(a) -> reverseReset a
                    | Sinh_D(a) -> reverseReset a
                    | Cosh_D(a) -> reverseReset a
                    | Tanh_D(a) -> reverseReset a
                    | Asin_D(a) -> reverseReset a
                    | Acos_D(a) -> reverseReset a
                    | Atan_D(a) -> reverseReset a
                    | Abs_D(a) -> reverseReset a
                    | Sign_D(a) -> reverseReset a
                    | Floor_D(a) -> reverseReset a
                    | Ceil_D(a) -> reverseReset a
                    | Round_D(a) -> reverseReset a
                    | Mul_Dot_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Mul_Dot_DV_DVCons(a, _) -> reverseReset a
                    | Sum_DV(a) -> reverseReset a
                    | L1Norm_DV(a) -> reverseReset a
                    | L2NormSq_DV(a) -> reverseReset a
                    | L2Norm_DV(a) -> reverseReset a
                    | Item_DV(a, _) -> reverseReset a
                    | Sum_DM(a) -> reverseReset a
                    | Item_DM(a, _, _) -> reverseReset a
                    | _ -> ()
            | _ -> ()
        | :? DV as d ->
            match d with
            | DVR(_,_,o,_,_) ->
                d.A <- DV.ZeroN d.Length
                d.F <- d.F + 1u
                if d.F = 1u then
                    match o with
                    | Add_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Add_DV_DVCons(a) -> reverseReset a
                    | Add_DV_D(a, b) -> reverseReset a; reverseReset b
                    | Add_DV_DCons(a) -> reverseReset a
                    | Add_DVCons_D(b) -> reverseReset b
                    | Sub_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Sub_DV_DVCons(a) -> reverseReset a
                    | Sub_DVCons_DV(a) -> reverseReset a
                    | Sub_DV_D(a, b) -> reverseReset a; reverseReset b
                    | Sub_DV_DCons(a) -> reverseReset a
                    | Sub_DVCons_D(b) -> reverseReset b
                    | Sub_D_DV(a, b) -> reverseReset a; reverseReset b
                    | Sub_D_DVCons(a) -> reverseReset a
                    | Sub_DCons_DV(b) -> reverseReset b
                    | Mul_Had_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Mul_Had_DV_DVCons(a, _) -> reverseReset a
                    | Mul_DV_D(a, b) -> reverseReset a; reverseReset b
                    | Mul_DV_DCons(a, _) -> reverseReset a
                    | Mul_DVCons_D(_, b) -> reverseReset b
                    | Mul_DM_DV(a, b) -> reverseReset a; reverseReset b
                    | Mul_DM_DVCons(a, _) -> reverseReset a
                    | Mul_DMCons_DV(_, b) -> reverseReset b
                    | Div_Had_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Div_Had_DV_DVCons(a, _) -> reverseReset a
                    | Div_Had_DVCons_DV(_, b) -> reverseReset b
                    | Div_DV_D(a, b) -> reverseReset a; reverseReset b
                    | Div_DV_DCons(a, _) -> reverseReset a
                    | Div_DVCons_D(_, b) -> reverseReset b
                    | Div_D_DV(a, b) -> reverseReset a; reverseReset b
                    | Div_D_DVCons(a, _) -> reverseReset a
                    | Div_DCons_DV(_, b) -> reverseReset b
                    | Exp_DV(a) -> reverseReset a
                    | Neg_DV(a) -> reverseReset a
                    | Abs_DV(a) -> reverseReset a
                    | Sign_DV(a) -> reverseReset a
                    | Make_DV(a) -> a |> Array.iter (fun v -> reverseReset v)
                    | SliceRow_DM(a,_,_) -> reverseReset a
                    | SliceCol_DM(a,_,_) -> reverseReset a
                    | Solve_DM_DV(a, b) -> reverseReset a; reverseReset b
                    | Solve_DM_DVCons(a, _) -> reverseReset a
                    | Solve_DMCons_DV(_, b) -> reverseReset b
                    | Append_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Append_DV_DVCons(a) -> reverseReset a
                    | Append_DVCons_DV(b) -> reverseReset b
                    | Split_DV(a,_) -> reverseReset a
                    | AddItem_DV_D(a,_,b) -> reverseReset a; reverseReset b
                    | AddItem_DV_DCons(a) -> reverseReset a
                    | AddItem_DVCons_D(_,b) -> reverseReset b
                    | AddSubVector_DV_DV(a,_,b) -> reverseReset a; reverseReset b
                    | AddSubVector_DV_DVCons(a) -> reverseReset a
                    | AddSubVector_DVCons_DV(_,b) -> reverseReset b
                    | Slice_DV(a,_) -> reverseReset a
                    | _ -> ()
            | _ -> ()
        | :? DM as d ->
            match d with
            | DMR(_,_,o,_,_) ->
                d.A <- DM Array2D.empty
                d.F <- d.F + 1u
                if d.F = 1u then
                    match o with
                    | Add_DM_DM(a, b) -> reverseReset a; reverseReset b
                    | Add_DM_DMCons(a) -> reverseReset a
                    | Sub_DM_DM(a, b) -> reverseReset a; reverseReset b
                    | Sub_DM_DMCons(a) -> reverseReset a
                    | Sub_DMCons_DM(a) -> reverseReset a
                    | Mul_DM_DM(a, b) -> reverseReset a; reverseReset b
                    | Mul_DM_DMCons(a, _) -> reverseReset a
                    | Mul_Had_DM_DM(a, b) -> reverseReset a; reverseReset b
                    | Mul_Had_DM_DMCons(a, _) -> reverseReset a
                    | Mul_DM_D(a, b) -> reverseReset a; reverseReset b
                    | Mul_DM_DCons(a, _) -> reverseReset a
                    | Mul_DMCons_D(_, b) -> reverseReset b
                    | Mul_Out_DV_DV(a, b) -> reverseReset a; reverseReset b
                    | Mul_Out_DV_DVCons(a, _) -> reverseReset a
                    | Mul_Out_DVCons_DV(_, b) -> reverseReset b
                    | Transpose_DM(a) -> reverseReset a
                    | Make_DM_ofD(a) -> a |> Array2D.iter (fun v -> reverseReset v)
                    | Make_DM_ofDV(a) -> a |> Array.iter (fun v -> reverseReset v)
                    | AddItem_DM_D(a, _, _, b) -> reverseReset a; reverseReset b
                    | AddItem_DM_DCons(a) -> reverseReset a
                    | AddItem_DMCons_D(_, _, b) -> reverseReset b
                    | AddSubMatrix_DM_DM(a,_,_,b) -> reverseReset a; reverseReset b
                    | AddSubMatrix_DM_DMCons(a) -> reverseReset a
                    | AddSubMatrix_DMCons_DM(_,_,b) -> reverseReset b
                    | Slice_DM(a,_,_) -> reverseReset a
                    | RowMatrix_DV(a) -> reverseReset a
                    | _ -> ()
            | _ -> ()
        | _ -> ()
    let reverseProp (v:obj) (d:obj) =
        d |> reverseReset
        d |> reversePush v

[<AutoOpen>]
module DiffOps =
    let inline diff' f x =
        x |> makeForward GlobalTagger.Next (D 1.) |> f |> primalTangent

    let inline diff f x = diff' f x |> snd

    let inline diff2 f x =
        diff (diff f) x

    let inline diff2'' f x =
        let v, d = diff' f x
        let d2 = diff2 f x
        (v, d, d2)

    let inline diff2' f x =
        diff2'' f x |> fsttrd

    let inline diffn n f x =
        if n < 0 then ErrorMessages.invalidArgDiffn()
        elif n = 0 then x |> f
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f

    let inline diffn' n f x =
        (x |> f, diffn n f x)

    let inline gradv' f x v =
        x |> Vector.makeForward GlobalTagger.Next v |> f |> primalTangent

    let inline gradv f x v = // TODO: optimize
        gradv' f x v |> snd

    let inline grad' f x =
        let xa = x |> Vector.makeReverse GlobalTagger.Next
        let z:D = f xa
        z |> reverseReset
        z |> reversePush (D 1.)
        (z.P, xa.A)

    let inline grad f x = // TODO: optimize
        grad' f x |> snd

    //let inline laplacian'

    //let inline laplacian

    let inline jacobianv' f x v =
        x |> Vector.makeForward GlobalTagger.Next v |> f |> Vector.primalTangent

    let inline jacobianv f x v = // TODO: optimize
        jacobianv' f x v |> snd

    let inline jacobianTv'' f x =
        let xa = x |> Vector.makeReverse GlobalTagger.Next
        let z:DV = f xa
        let r1 = z |> Vector.primal
        let r2 =
            fun (v:DV) ->
                z |> reverseReset
                z |> reversePush v
                xa |> Vector.adjoint
        (r1, r2)

    let inline jacobianTv' f x v =
        let r1, r2 = jacobianTv'' f x
        (r1, r2 v)

    let inline jacobianTv f x v = // TODO: optimize
        jacobianTv' f x v |> snd

    let inline jacobian' f (x:DV) =
        let o = x |> f |> Vector.primal
        if x.Length > o.Length then
            let r = jacobianTv f x
            (o, Array.init o.Length (fun j -> r (standardBasis o.Length j)) |> Matrix.ofRows)
        else
            (o, Array.init x.Length (fun i -> jacobianv f x (standardBasis x.Length i)) |> Matrix.ofCols)

    let inline jacobian f x =
        jacobian' f x |> snd

    let inline jacobianT' f x =
        jacobian' f x |> fun (r, j) -> (r, Matrix.transpose j)

    let inline jacobianT f x =
        jacobianT' f x |> snd

    let inline gradhessian f x =
        jacobian' (grad f) x

    let inline gradhessian' f x =
        let g, h = gradhessian f x
        (x |> f , g, h)

    let inline hessian f x =
        jacobian (grad f) x

    let inline hessian' f x =
        (x |> f, hessian f x)

    let inline gradhessianv' f x v =
        let gv, hv = grad' (fun xx -> gradv f xx v) x
        (x |> f, gv, hv)

    let inline gradhessianv f x v =
        gradhessianv' f x v |> sndtrd

    let inline hessianv' f x v =
        gradhessianv' f x v |> fsttrd

    let inline hessianv f x v =
        hessianv' f x v |> snd