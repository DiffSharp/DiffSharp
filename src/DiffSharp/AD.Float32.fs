// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

/// Nested forward and reverse mode automatic differentiation module
module DiffSharp.AD.Float32

#nowarn "77"

open DiffSharp.Util
open DiffSharp.Config
open System.Collections.Generic

type number = float32
let inline Backend<'T>               = GlobalConfig.Float32Backend
let inline VisualizationContrast<'T> = GlobalConfig.Float32VisualizationContrast
let inline FixedPointEpsilon<'T>     = GlobalConfig.Float32FixedPointEpsilon

module N =
    let inline toNumber x = float32 x
    let inline failWithInvalidTypeMessage () = failwith "Unsupported type. Expecting D, float32, or int."

    let [<Literal>] internal minus1 = -1.f
    let [<Literal>] internal half   = 0.5f
    let [<Literal>] internal zero   = 0.0f
    let [<Literal>] internal one    = 1.0f
    let [<Literal>] internal two    = 2.0f

    let inline log10Val<'T>              = log10ValFloat32

/// Scalar numeric type keeping dual numbers for forward mode and adjoints and tapes for reverse mode AD,
/// with nesting capability, using tags to avoid perturbation confusion
[<CustomEquality; CustomComparison>]
type D =

    /// Primal
    | D of number

    /// Primal, tangent, layer tag (for forward mode)
    | DF of primal: D * tanget: D * tag: uint32 

    /// Primal, parent, layer tag (for reverse mode)
    | DR of primal: D * parentOperation: TraceOp * tag: uint32 * uniq: int32

    interface dobj

    /// Make a reverse node
    static member R(d, op, ai) = DR(d, op, ai, UniqueTagger.Next())

    /// Primal value of this D
    member d.P =
        match d with
        | D _ -> d
        | DF(ap, _, _) -> ap
        | DR(ap, _, _, _) -> ap

    /// Deepest primal value of this D
    member d.PD =
        let rec prec x =
            match x with
            | D _ -> x
            | DF(xp, _, _) -> prec xp
            | DR(xp, _, _, _) -> prec xp
        prec d

    /// Tangent value of this D
    member d.T =
        match d with
        | D _ -> D.Zero
        | DF(_, at, _) -> at
        | DR _ -> failwith "Cannot get tangent value of DR."

    member d.GetForward(t:D, i:uint32) = DF(d, t, i)

    member d.GetReverse(i:uint32) = D.R(d, Noop, i)

    static member Zero = D N.zero

    static member One = D N.one

    static member op_Explicit(d:D):number =
        let rec prec x =
            match x with
            | D(p) -> p
            | DF(xp, _, _) -> prec xp
            | DR(xp, _, _, _) -> prec xp
        prec d

    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> compare (N.toNumber d) (N.toNumber d2)
            | _ -> invalidArg "" "Cannot compare this D with another type."

    override d.Equals(other) =
        match other with
        | :? D as d2 -> compare (N.toNumber d) (N.toNumber d2) = 0
        | _ -> false

    override d.GetHashCode() =
        match d with
        | D(ap) -> hash [|ap|]
        | DF(ap, at, ai) -> hash [|ap; at; ai|]
        | DR(ap, ao, ai, _) -> hash [|ap; ao; ai|]

    override d.ToString() =
        let (d':number) = D.op_Explicit(d)
        match d with
        | D _ -> sprintf "D % e" d'
        | DF(_) -> sprintf "DF % e" d'
        | DR(_) -> sprintf "DR % e" d'

    static member inline Op_D_D (a, ff, fd, df, r) =
        match a with
        | D(ap)                  -> D(ff(ap))
        | DF(ap, at, ai)           -> let cp = fd(ap) in DF(cp, df(cp, ap, at), ai)
        | DR(ap, _, ai, _)            -> D.R(fd(ap), r(a), ai)

    static member inline Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | D(ap) ->
            match b with
            | D(bp)                  -> D(ff(ap, bp))
            | DF(bp, bt, bi)         -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi)
            | DR(bp, _,  bi, _)         -> D.R(fd(a, bp), r_c_d(a, b), bi)
        | DF(ap, at, ai) ->
            match b with
            | D _                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | 0                  -> let cp = fd(ap, bp) in DF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                 -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                  -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                 -> D.R(fd(a, bp), r_c_d(a, b), bi) // ai < bi
                | 1                  -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                  -> failwith "Forward and reverse AD cannot run on the same level."
        | DR(ap, _,  ai, _) ->
            match b with
            | D _                   -> D.R(fd(ap, b), r_d_c(a, b), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | -1                 -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                  -> D.R(fd(ap, b), r_d_c(a, b), ai) // ai > bi
                | _                  -> failwith "Forward and reverse AD cannot run on the same level."
            | DR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                  -> D.R(fd(ap, bp), r_d_d(a, b), ai) // ai = bi
                | -1                 -> D.R(fd(a, bp), r_c_d(a, b), bi) // ai < bi
                | _                  -> D.R(fd(ap, b), r_d_c(a, b), ai) // ai > bi

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
        let inline r_c_d(a, b) = Div_DCons_D(a, b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Pow (a:D, b:D) =
        let inline ff(a, b) = a ** b
        let inline fd(a:D, b:D) = a ** b
        let inline df_da(cp, ap, at) = at * (ap ** (b - D.One)) * b
        let inline df_db(cp, bp, bt) = bt * cp * log a // cp = a ** bp
        let inline df_dab(cp:D, ap:D, at:D, bp:D, bt:D) = (ap ** (bp - D.One)) * (at * bp + ap * bt * log ap)
        let inline r_d_d(a, b) = Pow_D_D(a, b)
        let inline r_d_c(a, b) = Pow_D_DCons(a, b)
        let inline r_c_d(a, b) = Pow_DCons_D(a, b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Atan2 (a:D, b:D) =
        let inline ff(a, b) = atan2 a b
        let inline fd(a, b) = atan2 a b
        let inline df_da(cp, ap, at) = at * b / (ap * ap + b * b)
        let inline df_db(cp, bp, bt) = -bt * a / (a * a + bp * bp)
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp - bt * ap) / (ap * ap + bp * bp)
        let inline r_d_d(a, b) = Atan2_D_D(a, b)
        let inline r_d_c(a, b) = Atan2_D_DCons(a, b)
        let inline r_c_d(a, b) = Atan2_DCons_D(a, b)
        D.Op_D_D_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // D - number binary operations
    static member (+) (a:D, b:number) = a + (D b)
    static member (-) (a:D, b:number) = a - (D b)
    static member (*) (a:D, b:number) = a * (D b)
    static member (/) (a:D, b:number) = a / (D b)
    static member Pow (a:D, b:number) = a ** (D b)
    static member Atan2 (a:D, b:number) = atan2 a (D b)

    // number - D binary operations
    static member (+) (a:number, b:D) = (D a) + b
    static member (-) (a:number, b:D) = (D a) - b
    static member (*) (a:number, b:D) = (D a) * b
    static member (/) (a:number, b:D) = (D a) / b
    static member Pow (a:number, b:D) = (D a) ** b
    static member Atan2 (a:number, b:D) = atan2 (D a) b

    // D - int binary operations
    static member (+) (a:D, b:int) = a + (D (N.toNumber b))
    static member (-) (a:D, b:int) = a - (D (N.toNumber b))
    static member (*) (a:D, b:int) = a * (D (N.toNumber b))
    static member (/) (a:D, b:int) = a / (D (N.toNumber b))
    static member Pow (a:D, b:int) = a ** (D (N.toNumber b))
    static member Atan2 (a:D, b:int) = atan2 a (D (N.toNumber b))

    // int - D binary operations
    static member (+) (a:int, b:D) = (D (N.toNumber a)) + b
    static member (-) (a:int, b:D) = (D (N.toNumber a)) - b
    static member (*) (a:int, b:D) = (D (N.toNumber a)) * b
    static member (/) (a:int, b:D) = (D (N.toNumber a)) / b
    static member Pow (a:int, b:D) = (D (N.toNumber a)) ** b
    static member Atan2 (a:int, b:D) = atan2 (D (N.toNumber a)) b

    static member Log (a:D) =
        let inline ff(a) = log a
        let inline fd(a) = log a
        let inline df(cp, ap, at) = at / ap
        let inline r(a) = Log_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Log10 (a:D) =
        let inline ff(a) = log10 a
        let inline fd(a) = log10 a
        let inline df(cp, ap:D, at) = at / (ap * N.log10Val)
        let inline r(a) = Log10_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Exp (a:D) =
        let inline ff(a) = exp a
        let inline fd(a) = exp a
        let inline df(cp:D, ap:D, at:D) = at * cp // cp = exp ap
        let inline r(a) = Exp_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Sin (a:D) =
        let inline ff(a) = sin a
        let inline fd(a) = sin a
        let inline df(cp:D, ap:D, at:D) = at * cos ap
        let inline r(a) = Sin_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Cos (a:D) =
        let inline ff(a) = cos a
        let inline fd(a) = cos a
        let inline df(cp:D, ap:D, at:D) = -at * sin ap
        let inline r(a) = Cos_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Tan (a:D) =
        let inline ff(a) = tan a
        let inline fd(a) = tan a
        let inline df(cp:D, ap:D, at:D) = let cosa = cos ap in at / (cosa * cosa)
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
        let inline df(cp, ap, at) = at / ((D N.two) * cp) // cp = sqrt ap
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
        let inline ff(a:number) = asin a
        let inline fd(a:D) = asin a
        let inline df(cp, ap, at) = at / sqrt (D.One - ap * ap)
        let inline r(a) = Asin_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Acos (a:D) =
        let inline ff(a) = acos a
        let inline fd(a) = acos a
        let inline df(cp, ap, at) = -at / sqrt (D.One - ap * ap)
        let inline r(a) = Acos_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Atan (a:D) =
        let inline ff(a) = atan a
        let inline fd(a) = atan a
        let inline df(cp, ap, at) = at / (D.One + ap * ap)
        let inline r(a) = Atan_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Abs (a:D) =
        let inline ff(a) = abs a
        let inline fd(a) = abs a
        let inline df(cp, ap, at) = at * D.Sign(ap)
        let inline r(a) = Abs_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Sign (a:D) =
        let inline ff(a) = signummod a
        let inline fd(a) = D.Sign(a)
        let inline df(cp, ap, at) = D.Zero
        let inline r(a) = Sign_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Floor (a:D) =
        let inline ff(a) = floor a
        let inline fd(a) = floor a
        let inline df(cp:D, ap:D, at:D) = D.Zero
        let inline r(a) = Floor_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Ceiling (a:D) =
        let inline ff(a) = ceil a
        let inline fd(a) = ceil a
        let inline df(cp, ap, at) = D.Zero
        let inline r(a) = Ceil_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Round (a:D) =
        let inline ff(a) = round a
        let inline fd(a) = round a
        let inline df(cp, ap, at) = D.Zero
        let inline r(a) = Round_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member ReLU (a:D) =
        let inline ff(a) = max N.zero a
        let inline fd(a) = D.ReLU(a)
        let inline df(cp, ap, at:D) = at * (N.one + D.Sign(ap)) / N.two
        let inline r(a) = ReLU_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member Sigmoid (a:D) =
        let inline ff(a) = N.one / (N.one + exp -a)
        let inline fd(a) = D.Sigmoid(a)
        let inline df(cp:D, ap, at) = at * cp * (N.one - cp)
        let inline r(a) = Sigmoid_D(a)
        D.Op_D_D (a, ff, fd, df, r)

    static member SoftPlus (a:D) = log (N.one + exp a)

    static member SoftSign (a:D) = a / (N.one + abs a)

    static member LogSumExp (a:D) = a

    static member Max (a:D, b:D) = ((a + b) + abs (b - a)) / N.two

    static member Min (a:D, b:D) = ((a + b) - abs (a - b)) / N.two

    static member FixedPoint (g:D->D->D) (a0:D) (b:D) =
        let imax = DiffSharp.Config.GlobalConfig.FixedPointMaxIterations
        let eps = D FixedPointEpsilon

        let mutable a = a0
        let mutable i = 0

        match b with
        | D(bp) ->
            while i < imax do
                i <- i + 1
                if i >= imax then
                    //printfn "Fixed point iteration timeout, i = %i" i
                    ignore()
                else
                    let aa = g a b
                    if abs (aa - a) <= eps then
                        //printfn "Fixed point iteration converged, i = %i" i
                        i <- imax
                    a <- aa
            D (N.toNumber a)
        | DF(bp, bt, bi) ->
            while i < imax do
                i <- i + 1
                if i >= imax then
                    //printfn "Fixed point iteration timeout, i = %i" i
                    ignore()
                else
                    let aa = g a b
                    if (abs (aa.P - a.P) <= eps) && (abs (aa.T - a.T) <= eps) then
                        //printfn "Fixed point iteration converged, i = %i" i
                        i <- imax
                    a <- aa
            DF(a.P, a.T, bi)
        | DR(bp, _, bi, _) ->
            let bfirst = D.R(bp, Noop, bi) // Cut the connection between b and bfirst ("switch of graph construction" involving b beyond this point)
            while i < imax do
                i <- i + 1
                if i >= imax then
                    //printfn "Fixed point iteration timeout, i = %i" i
                    ignore()
                else
                    let aa = g a bfirst
                    if abs (aa - a) <= eps then
                        //printfn "Fixed point iteration converged, i = %i" i
                        i <- imax
                    a <- aa
            let aprev = D.R(a.P, Noop, bi)
            let alast = g aprev bfirst
            D.R(a.P, FixedPoint_D(b, bfirst, aprev, alast), bi)

/// Vector numeric type keeping dual numbers for forward mode and adjoints and tapes for reverse mode AD, with nesting
/// capability, using tags to avoid perturbation confusion
and DV =
    | DV of number[] // Primal
    | DVF of DV * DV * uint32 // Primal, tangent, layer tag
    | DVR of DV * TraceOp * uint32 * int32 // Primal, parent operation, layer tag, unique

    interface dobj

    /// Primal value of this DV
    member d.P =
        match d with
        | DV _ -> d
        | DVF(ap, _, _) -> ap
        | DVR(ap, _, _, _) -> ap

    /// Deepest primal value of this DV
    member d.PD =
        let rec prec x =
            match x with
            | DV _ -> x
            | DVF(xp, _, _) -> prec xp
            | DVR(xp, _, _, _) -> prec xp
        prec d

    /// Tangent value of this DV
    member d.T =
        match d with
        | DV _ -> DV.ZeroN d.Length
        | DVF(_, at, _) -> at
        | DVR _ -> failwith "Cannot get tangent value of DVR."

    /// Convert to use forward AD at this layer
    member d.GetForward(t:DV, i:uint32) = DVF(d, t, i)

    /// Convert to use reverse AD at this layer
    member d.GetReverse(i:uint32) = DV.R(d, Noop, i)

    /// Make a reverse node
    static member R(d, op, ai) = DVR(d, op, ai, UniqueTagger.Next())

    member d.Length =
        match d with
        | DV(ap) -> ap.Length
        | DVF(ap, _, _) -> ap.Length
        | DVR(ap, _, _, _) -> ap.Length

    member d.Item
        with get i =
            match d with
            | DV(ap) -> D(ap.[i])
            | DVF(ap, at, ai) -> DF(ap.[i], at.[i], ai)
            | DVR(ap, _, ai, _) -> D.R(ap.[i], Item_DV(d, i), ai)

    member d.GetSlice(lower, upper) =
        let l = defaultArg lower 0
        let u = defaultArg upper (d.Length - 1)
        match d with
        | DV(ap) -> DV(ap.[l..u])
        | DVF(ap, at, ai) -> DVF(ap.[l..u], at.[l..u], ai)
        | DVR(ap, _, ai, _) -> let cp = ap.[l..u] in DV.R(cp, Slice_DV(d, l), ai)

    member d.ToArray() =
        match d with
        | DV(ap) -> ap |> Array.map D
        | DVF(ap, at, ai) ->
            Array.init ap.Length (fun i -> DF(ap.[i], at.[i], ai))
        | DVR(ap, _, ai, _) ->
            Array.init ap.Length (fun i -> D.R(ap.[i], Item_DV(d, i), ai))

    member d.ToRowDM() =
        match d with
        | DV(ap) -> seq [ap] |> array2D |> DM
        | DVF(ap, at, ai) -> DMF(ap.ToRowDM(), at.ToRowDM(), ai)
        | DVR(ap, _, ai, _) -> let cp = ap.ToRowDM() in DM.R(cp, RowMatrix_DV(d), ai)

    member d.ToColDM() = DM.Transpose(d.ToRowDM())

    override d.ToString() =
        let (d':number[]) = DV.op_Explicit(d)
        let sb = System.Text.StringBuilder()
        match d with
        | DV _ -> sb.AppendLine(sprintf "DV : %i" d.Length) |> ignore
        | DVF(_) -> sb.AppendLine(sprintf "DVF: %i" d.Length) |> ignore
        | DVR(_) -> sb.AppendLine(sprintf "DVR: %i" d.Length) |> ignore
        for i = 0 to d.Length - 1 do
            sb.Append(sprintf "% 9.3g " d'.[i]) |> ignore
        sb.ToString()

    member d.ToMathematicaString() =
        let (d':number[]) = DV.op_Explicit(d)
        let sb = System.Text.StringBuilder()
        sb.Append("{") |> ignore
        for i = 0 to d.Length - 1 do
            sb.Append(sprintf "%.2f" d'.[i]) |> ignore
            if i < d.Length - 1 then sb.Append(", ") |> ignore
        sb.Append("}") |> ignore
        sb.ToString()

    member d.ToMatlabString() =
        let (d':number[]) = DV.op_Explicit(d)
        let sb = System.Text.StringBuilder()
        sb.Append("[") |> ignore
        for i = 0 to d.Length - 1 do
            sb.Append(sprintf "%.2f" d'.[i]) |> ignore
            if i < d.Length - 1 then sb.Append(" ") |> ignore
        sb.Append("]") |> ignore
        sb.ToString()

    static member Zero = DV Array.empty

    static member ZeroN n = DV(Array.zeroCreate n)

    static member op_Explicit(d:DV):number[] =
        let rec prec x =
            match x with
            | DV(p) -> p
            | DVF(xp, _, _) -> prec xp
            | DVR(xp, _, _, _) -> prec xp
        prec d

    static member op_Explicit(d) = DV(d)

    static member OfArray (a:D[]) =
        // TODO: check to ensure that all elements in the array are of the same type (D, DF, or DR) and have the same nesting tag
        match a.[0] with
        | D _ -> DV(a |> Array.map N.toNumber)
        | DF(_, _, ai) ->
            let ap = a |> Array.map (fun x -> x.P)
            let at = a |> Array.map (fun x -> x.T)
            DVF(DV.OfArray(ap), DV.OfArray(at), ai)
        | DR(_, _, ai, _) ->
            let ap = a |> Array.map (fun x -> x.P)
            let cp = DV.OfArray(ap) in DV.R(cp, Make_DV_ofDs(a), ai)

    static member Split(d:DV, n:seq<int>) =
        match d with
        | DV(ap) ->
            seq {let i = ref 0;
                 for j in n do yield Array.sub ap !i j |> DV; i := !i + j}
        | DVF(ap, at, ai) ->
            let aps = DV.Split(ap, n)
            let ats = DV.Split(at, n)
            Seq.map2 (fun p t -> DVF(p, t, ai)) aps ats
        | DVR(ap, _, ai, _) ->
            let aps = DV.Split(ap, n)
            let ii = n |> Seq.mapFold (fun s i -> s, s + i) 0 |> fst |> Array.ofSeq
            Seq.mapi (fun i p -> DV.R(p, Split_DV(d, ii.[i]), ai)) aps


    static member inline Op_DV_DV (a, ff, fd, df, r) =
        match a with
        | DV(ap)                      -> DV(ff(ap))
        | DVF(ap, at, ai)             -> let cp = fd(ap) in DVF(cp, df(cp, ap, at), ai)
        | DVR(ap, _, ai, _)            -> let cp = fd(ap) in DV.R(cp, r(a), ai)

    static member inline Op_DV_DM (a, ff, fd, df, r) =
        match a with
        | DV(ap)                      -> DM(ff(ap))
        | DVF(ap, at, ai)             -> let cp = fd(ap) in DMF(cp, df(cp, ap, at), ai)
        | DVR(ap, _, ai, _)            -> let cp = fd(ap) in DM.R(cp, r(a), ai)

    static member inline Op_DV_D (a, ff, fd, df, r) =
        match a with
        | DV(ap)                      -> D(ff(ap))
        | DVF(ap, at, ai)             -> let cp = fd(ap) in DF(cp, df(cp, ap, at), ai)
        | DVR(ap, _, ai, _)            -> let cp = fd(ap) in D.R(cp, r(a), ai)

    static member inline Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DV(bp)                  -> DV(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp, _, bi, _) -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi)
        | DVF(ap, at, ai) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap, _, ai, _) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DV.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DV_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DV(bp)                  -> DM(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp, _, bi, _) -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi)
        | DVF(ap, at, ai) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap, _, ai, _) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DM.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DV_DV_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DV(bp)                  -> D(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp, _, bi, _) -> D.R(fd(a, bp), r_c_d(a, b), bi)
        | DVF(ap, at, ai) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> D.R(fd(a, bp), r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap, _, ai, _) ->
            match b with
            | DV _                   -> D.R(fd(ap, b), r_d_c(a, b), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> D.R(fd(ap, b), r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> D.R(fd(ap, bp), r_d_d(a, b), ai) // ai = bi
                | -1                  -> D.R(fd(a, bp), r_c_d(a, b), bi) // ai < bi
                | _                   -> D.R(fd(ap, b), r_d_c(a, b), ai) // ai > bi

    static member inline Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | D(bp)                   -> DV(ff(ap, bp))
            | DF(bp, bt, bi)          -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DR(bp, _, bi, _)  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi)
        | DVF(ap, at, ai) ->
            match b with
            | D _                    -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | 0                    -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                   -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                    -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                   -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                    -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                    -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap, _, ai, _) ->
            match b with
            | D _                    -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | -1                   -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                    -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                    -> failwith "Forward and reverse AD cannot run on the same level."
            | DR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                    -> let cp = fd(ap, bp) in DV.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                   -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                    -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi


    static member inline Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | D(ap) ->
            match b with
            | DV(bp)                  -> DV(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp, _, bi, _) -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi)
        | DF(ap, at, ai) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DR(ap, _, ai, _) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DV.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi

    /// Element-wise addition of `a` and `b`
    static member (+) (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Add_V_V(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_DV(a, b)
        let inline r_d_c(a, b) = Add_DV_DVCons(a)
        let inline r_c_d(a, b) = Add_DV_DVCons(b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise addition of `a` and `b`, potentially destructive of the storage of raw matrices in 'b'
    static member Add_V_V_Inplace (a:DV, b:DV) =
        let inline ff(a:number[], b:number[]) = Backend.Add_V_V_Inplace(a, b); b
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_DV(a, b)
        let inline r_d_c(a, b) = Add_DV_DVCons(a)
        let inline r_c_d(a, b) = Add_DV_DVCons(b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise subtraction of `a` and `b`
    static member (-) (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Sub_V_V(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DV_DV(a, b)
        let inline r_d_c(a, b) = Sub_DV_DVCons(a)
        let inline r_c_d(a, b) = Sub_DVCons_DV(b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Inner (dot, scalar) product of `a` and `b`
    static member (*) (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Mul_Dot_V_V(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_Dot_DV_DV(a, b)
        let inline r_d_c(a, b) = Mul_Dot_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_Dot_DV_DVCons(b, a)
        DV.Op_DV_DV_D (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise (Hadamard, Schur) product of `a` and `b`
    static member (.*) (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Map2_F_V_V((*), a, b)
        let inline fd(a, b) = a .* b
        let inline df_da(cp, ap, at) = at .* b
        let inline df_db(cp, bp, bt) = a .* bt
        let inline df_dab(cp, ap, at, bp, bt) = (at .* bp) + (ap .* bt)
        let inline r_d_d(a, b) = Mul_Had_DV_DV(a, b)
        let inline r_d_c(a, b) = Mul_Had_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_Had_DV_DVCons(b, a)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Outer (dyadic, tensor) product of `a` and `b`
    static member (&*) (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Mul_Out_V_V(a, b)
        let inline fd(a, b) = a &* b
        let inline df_da(cp, ap, at) = at &* b
        let inline df_db(cp, bp, bt) = a &* bt
        let inline df_dab(cp, ap, at, bp, bt) = (at &* bp) + (ap &* bt)
        let inline r_d_d(a, b) = Mul_Out_DV_DV(a, b)
        let inline r_d_c(a, b) = Mul_Out_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_Out_DVCons_DV(a, b)
        DV.Op_DV_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise (Hadamard, Schur) division of `a` and `b`
    static member (./) (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Map2_F_V_V((/), a, b)
        let inline fd(a, b) = a ./ b
        let inline df_da(cp, ap, at) = at ./ b
        let inline df_db(cp, bp, bt) = -bt .* cp ./ bp // cp = ap / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt .* cp) ./ bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_Had_DV_DV(a, b)
        let inline r_d_c(a, b) = Div_Had_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Div_Had_DVCons_DV(a, b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise power of `a` and `b`
    static member Pow (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Map2_F_V_V((fun x y -> x ** y), a, b)
        let inline fd(a:DV, b:DV) = a ** b
        let inline df_da(cp:DV, ap:DV, at:DV) = at .* (ap ** (b - D.One)) .* b
        let inline df_db(cp, bp, bt) = bt .* cp .* log a // cp = a ** bp
        let inline df_dab(cp:DV, ap:DV, at:DV, bp:DV, bt:DV) = (ap ** (bp - D.One)) .* ((at .* bp) + (ap .* bt .* log ap))
        let inline r_d_d(a, b) = Pow_DV_DV(a, b)
        let inline r_d_c(a, b) = Pow_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Pow_DVCons_DV(a, b)
        DV.Op_DV_DV_DV(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise atan2 of `a` and `b`
    static member Atan2 (a:DV, b:DV) =
        let inline ff(a, b) = Backend.Map2_F_V_V(atan2, a, b)
        let inline fd(a, b) = atan2 a b
        let inline df_da(cp, ap, at) = (at .* b) ./ ((ap .* ap) + (b .* b))
        let inline df_db(cp, bp, bt) = (-bt .* a) ./ ((a .* a) + (bp .* bp))
        let inline df_dab(cp, ap, at, bp, bt) = ((at .* bp) - (bt .* ap)) ./ ((ap .* ap) + (bp .* bp))
        let inline r_d_d(a, b) = Atan2_DV_DV(a, b)
        let inline r_d_c(a, b) = Atan2_DV_DVCons(a, b)
        let inline r_c_d(a, b) = Atan2_DVCons_DV(a, b)
        DV.Op_DV_DV_DV(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Multiply vector `a` by scalar `b`
    static member (*) (a:DV, b:D) =
        let inline ff(a, b) = Backend.Mul_S_V(b, a)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DV_D(a, b)
        let inline r_d_c(a, b) = Mul_DV_DCons(a, b)
        let inline r_c_d(a, b) = Mul_DVCons_D(a, b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Multiply vector `b` by scalar `a`
    static member (*) (a:D, b:DV) =
        let inline ff(a, b) = Backend.Mul_S_V(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DV_D(b, a)
        let inline r_d_c(a, b) = Mul_DVCons_D(b, a)
        let inline r_c_d(a, b) = Mul_DV_DCons(b, a)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Divide vector `a` by scalar `b`
    static member (/) (a:DV, b:D) =
        let inline ff(a, b) = Backend.Mul_S_V(N.one / b, a)
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt * cp / bp // cp = a / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt * cp) / bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_DV_D(a, b)
        let inline r_d_c(a, b) = Div_DV_DCons(a, b)
        let inline r_c_d(a, b) = Div_DVCons_D(a, b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Generate a vector where each element is scalar `a` divided by the corresponding element of vector `b`
    static member (/) (a:D, b:DV) =
        let inline ff(a, b) = Backend.Map_F_V((fun v -> a / v), b)
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt .* (cp ./ bp) // cp = a / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt * cp) / bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_D_DV(a, b)
        let inline r_d_c(a, b) = Div_D_DVCons(a, b)
        let inline r_c_d(a, b) = Div_DCons_DV(a, b)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Add scalar `b` to vector `a`
    static member (+) (a:DV, b:D) =
        let inline ff(a, b) = Backend.Add_S_V(b, a)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.OfArray(Array.create a.Length bt)
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_D(a, b)
        let inline r_d_c(a, b) = Add_DV_DCons(a)
        let inline r_c_d(a, b) = Add_DVCons_D(b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Add scalar `a` to vector `b`
    static member (+) (a:D, b:DV) =
        let inline ff(a, b) = Backend.Add_S_V(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = DV.OfArray(Array.create b.Length at)
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DV_D(b, a)
        let inline r_d_c(a, b) = Add_DVCons_D(a)
        let inline r_c_d(a, b) = Add_DV_DCons(b)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Subtract scalar `b` from vector `a`
    static member (-) (a:DV, b:D) =
        let inline ff(a, b) = Backend.Sub_V_S(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.OfArray(Array.create a.Length -bt)
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DV_D(a, b)
        let inline r_d_c(a, b) = Sub_DV_DCons(a)
        let inline r_c_d(a, b) = Sub_DVCons_D(b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Generate a vector where each element is the corresponding element of vector `b` subtracted from scalar `a`
    static member (-) (a:D, b:DV) =
        let inline ff(a, b) = Backend.Sub_S_V(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = DV.OfArray(Array.create b.Length at)
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_D_DV(a, b)
        let inline r_d_c(a, b) = Sub_D_DVCons(a)
        let inline r_c_d(a, b) = Sub_DCons_DV(b)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Generate a vector where each corresponding element of vector `a` is raised to the power of scalar `b`
    static member Pow (a:DV, b:D) =
        let inline ff(a, b) = Backend.Map_F_V((fun v -> v ** b), a)
        let inline fd(a:DV, b:D) = a ** b
        let inline df_da(cp, ap:DV, at:DV) = at .* (ap ** (b - D.One)) * b
        let inline df_db(cp, bp, bt) = bt * cp .* log a // cp = a ** bp
        let inline df_dab(cp, ap:DV, at:DV, bp:D, bt:D) = (ap ** (bp - D.One)) .* ((at * bp) + (ap * bt .* log ap))
        let inline r_d_d(a, b) = Pow_DV_D(a, b)
        let inline r_d_c(a, b) = Pow_DV_DCons(a, b)
        let inline r_c_d(a, b) = Pow_DVCons_D(a, b)
        DV.Op_DV_D_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Generate a vector where scalar `a` is raised to the power of each corresponding element of vector `b`
    static member Pow (a:D, b:DV) =
        let inline ff(a, b) = Backend.Map_F_V((fun v -> a ** v), b)
        let inline fd(a:D, b:DV) = DV.Pow(a, b)
        let inline df_da(cp, ap:D, at:D) = (at * (DV.Pow(ap, b - D.One))) .* b
        let inline df_db(cp, bp, bt) = bt .* cp * log a // cp = a ** bp
        let inline df_dab(cp, ap:D, at:D, bp:DV, bt:DV) = (DV.Pow(ap, bp - D.One)) .* ((at * bp) + (ap * bt * log ap))
        let inline r_d_d(a, b) = Pow_D_DV(a, b)
        let inline r_d_c(a, b) = Pow_D_DVCons(a, b)
        let inline r_c_d(a, b) = Pow_DCons_DV(a, b)
        DV.Op_D_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Generate a vector where each corresponding element of vector `a` is raised to the power of scalar `b`
    static member Atan2 (a:DV, b:D) =
        let inline ff(a, b) = Backend.Map_F_V((fun v -> atan2 v b), a)
        let inline fd(a:DV, b:D) = DV.Atan2(a, b)
        let inline df_da(cp, ap, at) = (at * b) ./ ((ap .* ap) + (b * b))
        let inline df_db(cp, bp, bt) = (-bt * a) ./ ((a .* a) + (bp * bp))
        let inline df_dab(cp, ap, at, bp, bt) = ((at * bp) - (bt * ap)) ./ ((ap .* ap) + (bp * bp))
        let inline r_d_d(a, b) = Atan2_DV_D(a, b)
        let inline r_d_c(a, b) = Atan2_DV_DCons(a, b)
        let inline r_c_d(a, b) = Atan2_DVCons_D(a, b)
        DV.Op_DV_D_DV(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Generate a vector where scalar `a` is raised to the power of each corresponding element of vector `b`
    static member Atan2 (a:D, b:DV) =
        let inline ff(a, b) = Backend.Map_F_V((fun v -> atan2 a v), b)
        let inline fd(a:D, b:DV) = DV.Atan2(a, b)
        let inline df_da(cp, ap, at) = (at * b) ./ ((ap * ap) + (b .* b))
        let inline df_db(cp, bp, bt) = (-bt * a) ./ ((a * a) + (bp .* bp))
        let inline df_dab(cp, ap, at, bp, bt) = ((at * bp) - (bt * ap)) ./ ((ap * ap) + (bp .* bp))
        let inline r_d_d(a, b) = Atan2_D_DV(a, b)
        let inline r_d_c(a, b) = Atan2_D_DVCons(a, b)
        let inline r_c_d(a, b) = Atan2_DCons_DV(a, b)
        DV.Op_D_DV_DV(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Add scalar `b` to vector `a` at index `i`
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

    /// Add subvector `b` to vector `a`, starting from index `i`
    static member AddSubVector (a:DV, i:int, b:DV) =
        let inline ff(a:_[], b:_[]) =
            let aa = Array.copy a
            for j = 0 to b.Length - 1 do
                aa.[i + j] <- aa.[i + j] + b.[j]
            aa
        let inline fd(a, b) = DV.AddSubVector(a, i, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DV.AddSubVector(DV.ZeroN a.Length, i, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DV.AddSubVector(at, i, bt)
        let inline r_d_d(a, b) = AddSubVector_DV_DV(a, i, b)
        let inline r_d_c(a, b) = AddSubVector_DV_DVCons(a)
        let inline r_c_d(a, b) = AddSubVector_DVCons_DV(i, b)
        DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // DV - number binary operations
    static member (+) (a:DV, b:number) = a + D b
    static member (-) (a:DV, b:number) = a - D b
    static member (*) (a:DV, b:number) = a * D b
    static member (/) (a:DV, b:number) = a / D b
    static member Pow (a:DV, b:number) = a ** D b
    static member Atan2 (a:DV, b:number) = DV.Atan2(a, D b)

    // number - DV binary operations
    static member (+) (a:number, b:DV) = (D a) + b
    static member (-) (a:number, b:DV) = (D a) - b
    static member (*) (a:number, b:DV) = (D a) * b
    static member (/) (a:number, b:DV) = (D a) / b
    static member Pow (a:number, b:DV) = DV.Pow(D a, b)
    static member Atan2 (a:number, b:DV) = DV.Atan2(D a, b)

    // DV - int binary operations
    static member (+) (a:DV, b:int) = a + D (N.toNumber b)
    static member (-) (a:DV, b:int) = a - D (N.toNumber b)
    static member (*) (a:DV, b:int) = a * D (N.toNumber b)
    static member (/) (a:DV, b:int) = a / D (N.toNumber b)
    static member Pow (a:DV, b:int) = a ** D (N.toNumber b)
    static member Atan2 (a:DV, b: int) = DV.Atan2(a, D (N.toNumber b))

    // int - DV binary operations
    static member (+) (a:int, b:DV) = (D (N.toNumber a)) + b
    static member (-) (a:int, b:DV) = (D (N.toNumber a)) - b
    static member (*) (a:int, b:DV) = (D (N.toNumber a)) * b
    static member (/) (a:int, b:DV) = (D (N.toNumber a)) / b
    static member Pow (a:int, b:DV) = DV.Pow(D (N.toNumber a), b)
    static member Atan2 (a:int, b:DV) = DV.Atan2(D (N.toNumber a), b)

    static member Log (a:DV) =
        let inline ff(a) = Backend.Map_F_V(log, a)
        let inline fd(a) = log a
        let inline df(cp, ap, at) = at ./ ap
        let inline r(a) = Log_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Log10 (a:DV) =
        let inline ff(a) = Backend.Map_F_V(log10, a)
        let inline fd(a) = log10 a
        let inline df(cp, ap:DV, at:DV) = at ./ (ap * N.log10Val)
        let inline r(a) = Log10_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Exp (a:DV) =
        let inline ff(a) = Backend.Map_F_V(exp, a)
        let inline fd(a) = exp a
        let inline df(cp, ap, at) = at .* cp // cp = exp ap
        let inline r(a) = Exp_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Sin (a:DV) =
        let inline ff(a) = Backend.Map_F_V(sin, a)
        let inline fd(a) = sin a
        let inline df(cp, ap:DV, at:DV) = at .* cos ap
        let inline r(a) = Sin_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Cos (a:DV) =
        let inline ff(a) = Backend.Map_F_V(cos, a)
        let inline fd(a) = cos a
        let inline df(cp, ap:DV, at:DV) = -at .* sin ap
        let inline r(a) = Cos_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Tan (a:DV) =
        let inline ff(a) = Backend.Map_F_V(tan, a)
        let inline fd(a) = tan a
        let inline df(cp, ap:DV, at:DV) = let cosa = cos ap in at ./ (cosa .* cosa)
        let inline r(a) = Tan_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member (~-) (a:DV) =
        let inline ff(a) = Backend.Mul_S_V(N.minus1, a)
        let inline fd(a) = -a
        let inline df(cp, ap, at) = -at
        let inline r(a) = Neg_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Sqrt (a:DV) =
        let inline ff(a) = Backend.Map_F_V(sqrt, a)
        let inline fd(a) = sqrt a
        let inline df(cp:DV, ap:DV, at:DV) = at ./ (D N.two * cp) // cp = sqrt ap
        let inline r(a) = Sqrt_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Sinh (a:DV) =
        let inline ff(a) = Backend.Map_F_V(sinh, a)
        let inline fd(a) = sinh a
        let inline df(cp:DV, ap:DV, at:DV) = at .* cosh ap
        let inline r(a) = Sinh_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Cosh (a:DV) =
        let inline ff(a) = Backend.Map_F_V(cosh, a)
        let inline fd(a) = cosh a
        let inline df(cp:DV, ap:DV, at:DV) = at .* sinh ap
        let inline r(a) = Cosh_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Tanh (a:DV) =
        let inline ff(a) = Backend.Map_F_V(tanh, a)
        let inline fd(a) = tanh a
        let inline df(cp:DV, ap:DV, at:DV) = let cosha = cosh ap in at ./ (cosha .* cosha)
        let inline r(a) = Tanh_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Asin (a:DV) =
        let inline ff(a) = Backend.Map_F_V(asin, a)
        let inline fd(a) = asin a
        let inline df(cp:DV, ap:DV, at:DV) = at ./ sqrt (D.One - (ap .* ap))
        let inline r(a) = Asin_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Acos (a:DV) =
        let inline ff(a) = Backend.Map_F_V(acos, a)
        let inline fd(a) = acos a
        let inline df(cp:DV, ap:DV, at:DV) = -at ./ sqrt (D.One - (ap .* ap))
        let inline r(a) = Acos_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Atan (a:DV) =
        let inline ff(a) = Backend.Map_F_V(atan, a)
        let inline fd(a) = atan a
        let inline df(cp:DV, ap:DV, at:DV) = at ./ sqrt (D.One + (ap .* ap))
        let inline r(a) = Atan_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Abs (a:DV) =
        let inline ff(a) = Backend.Map_F_V(abs, a)
        let inline fd(a) = abs a
        let inline df(cp, ap, at) = at .* (DV.Sign ap)
        let inline r(a) = Abs_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Sign (a:DV) =
        let inline ff(a) = Backend.Map_F_V(signummod, a)
        let inline fd(a) = DV.Sign a
        let inline df(cp, ap, at) = DV.ZeroN a.Length
        let inline r(a) = Sign_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Floor (a:DV) =
        let inline ff(a) = Backend.Map_F_V(floor, a)
        let inline fd(a) = floor a
        let inline df(cp, ap, at) = DV.ZeroN a.Length
        let inline r(a) = Floor_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Ceiling (a:DV) =
        let inline ff(a) = Backend.Map_F_V(ceil, a)
        let inline fd(a) = ceil a
        let inline df(cp, ap, at) = DV.ZeroN a.Length
        let inline r(a) = Ceil_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Round (a:DV) =
        let inline ff(a) = Backend.Map_F_V(round, a)
        let inline fd(a) = round a
        let inline df(cp, ap, at) = DV.ZeroN a.Length
        let inline r(a) = Round_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    /// L1 norm of vector `a`
    static member L1Norm (a:DV) =
        let inline ff(a) = Backend.L1Norm_V(a)
        let inline fd(a) = DV.L1Norm(a)
        let inline df(cp, ap, at) = at * DV.Sign(ap)
        let inline r(a) = L1Norm_DV(a)
        DV.Op_DV_D (a, ff, fd, df, r)

    /// Squared L2 norm of vector `a`
    static member L2NormSq (a:DV) =
        let inline ff(a) = let l2norm = Backend.L2Norm_V(a) in l2norm * l2norm
        let inline fd(a) = DV.L2NormSq(a)
        let inline df(cp, ap, at) = (D N.two) * (ap * at)
        let inline r(a) = L2NormSq_DV(a)
        DV.Op_DV_D (a, ff, fd, df, r)

    /// L2 norm of vector `a`
    static member L2Norm (a:DV) =
        let inline ff(a) = Backend.L2Norm_V(a)
        let inline fd(a) = DV.L2Norm(a)
        let inline df(cp, ap, at) = (ap * at) / cp // cp = DV.L2Norm(ap)
        let inline r(a) = L2Norm_DV(a)
        DV.Op_DV_D (a, ff, fd, df, r)

    /// Sum of the elements of vector `a`
    static member Sum (a:DV) =
        let inline ff(a) = Backend.Sum_V(a)
        let inline fd(a) = DV.Sum(a)
        let inline df(cp, ap, at) = DV.Sum(at)
        let inline r(a) = Sum_DV(a)
        DV.Op_DV_D (a, ff, fd, df, r)

    /// Append vector `b` to vector `a`
    static member Append (a:DV, b:DV) =
        if a.Length = 0 then
            b
        elif b.Length = 0 then
            a
        else
            let inline ff(a, b) = Array.append a b
            let inline fd(a, b) = DV.Append(a, b)
            let inline df_da(cp, ap, at) = DV.Append(at, DV.ZeroN b.Length)
            let inline df_db(cp, bp, bt) = DV.Append(DV.ZeroN a.Length, bt)
            let inline df_dab(cp, ap, at, bp, bt) = DV.Append(at, bt)
            let inline r_d_d(a, b) = Append_DV_DV(a, b)
            let inline r_d_c(a, b) = Append_DV_DVCons(a)
            let inline r_c_d(a, b) = Append_DVCons_DV(b)
            DV.Op_DV_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member ReshapeToDM (m:int, a:DV) =
        let inline ff(a) = Backend.ReshapeCopy_V_MRows(m, a)
        let inline fd(a) = DV.ReshapeToDM(m, a)
        let inline df(cp, ap, at) = DV.ReshapeToDM(m, at)
        let inline r(a) = ReshapeCopy_DV_DM(a)
        DV.Op_DV_DM (a, ff, fd, df, r)

    static member ReLU (a:DV) =
        let inline ff(a) = Backend.Map_F_V(max N.zero, a)
        let inline fd(a) = DV.ReLU(a)
        let inline df(cp, ap, at) = at .* ((N.one + DV.Sign(ap)) / N.two)
        let inline r(a) = ReLU_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member Sigmoid (a:DV) =
        let inline ff(a) = Backend.Map_F_V((fun v -> N.one / (N.one + exp -v)), a)
        let inline fd(a) = DV.Sigmoid(a)
        let inline df(cp:DV, ap, at) = at .* cp .* (N.one - cp)
        let inline r(a) = Sigmoid_DV(a)
        DV.Op_DV_DV (a, ff, fd, df, r)

    static member SoftPlus (a:DV) = log (N.one + exp a)

    static member SoftSign (a:DV) = a ./ (N.one + abs a)

    static member LogSumExp (a:DV) =
        let inline ff(a) =
            let m = Array.max a
            let aa = Backend.Sub_V_S(a, m)
            m + log (Backend.Map_F_V(exp, aa) |> Array.sum)
        let inline fd(a) = DV.LogSumExp(a)
        let inline df(cp:D, ap:DV, at:DV) = (at * (exp ap)) / exp cp // cp = DV.LogSumExp(ap)
        let inline r(a) = LogSumExp_DV(a)
        DV.Op_DV_D (a, ff, fd, df, r)

    static member Mean (a:DV) =
        DV.Sum(a) / a.Length

    static member Variance (a:DV) =
        let a' = a - DV.Mean(a)
        DV.Sum(a' .* a') / (a.Length - 1)

    static member StandardDev (a:DV) =
        DV.Variance(a) |> sqrt

    static member Standardize (a:DV) =
        let sd = DV.StandardDev(a)
        if sd = D.Zero then
            a * (D.Zero)
        else
            (a - DV.Mean(a)) / DV.StandardDev(a)

    static member Normalize (a:DV) =
        let min = DV.Min(a)
        let range = DV.Max(a) - min
        if range = D.Zero then
            a * (D.Zero)
        else
            (a - min) / range

    static member Max (a:DV, b:DV) = ((a + b) + abs (b - a)) / N.two
    static member Max (a:DV, b:D) = ((a + b) + abs (b - a)) / N.two
    static member Max (a:D, b:DV) = ((a + b) + abs (b - a)) / N.two
    static member Min (a:DV, b:DV) = ((a + b) - abs (a - b)) / N.two
    static member Min (a:DV, b:D) = ((a + b) - abs (a - b)) / N.two
    static member Min (a:D, b:DV) = ((a + b) - abs (a - b)) / N.two

    /// Index of the maximum element of vector `a`
    static member MaxIndex (a:DV) =
        let a' = DV.op_Explicit(a)
        let mutable maxi = 0
        let mutable maxv = a'.[0]
        for i = 0 to a'.Length - 1 do
            if a'.[i] > maxv then maxi <- i; maxv <- a'.[i]
        maxi
    static member Max (a:DV) = a.[DV.MaxIndex(a)]

    /// Index of the minimum element of vector `b`
    static member MinIndex (a:DV) =
        let a' = DV.op_Explicit(a)
        let mutable mini = 0
        let mutable minv = a'.[0]
        for i = 0 to a'.Length - 1 do
            if a'.[i] < minv then mini <- i; minv <- a'.[i]
        mini
    static member Min (a:DV) = a.[DV.MinIndex(a)]

    static member SoftMax (a:DV) =
        let a' = a - DV.Max(a)
        let e = exp a'
        e / DV.Sum(e)

    member d.Visualize() =
        let (d':number[]) = ((VisualizationContrast * (DV.Normalize(d.P) - N.half)) + N.half) |> DV.op_Explicit
        let sb = System.Text.StringBuilder()
        match d with
        | DV _ -> sb.AppendLine(sprintf "DV : %i" d.Length) |> ignore
        | DVF(_) -> sb.AppendLine(sprintf "DVF: %i" d.Length) |> ignore
        | DVR(_) -> sb.AppendLine(sprintf "DVR: %i" d.Length) |> ignore
        let palette = GlobalConfig.GrayscalePalette
        let palettel = palette.Length
        let palettelf = N.toNumber palettel
        for i = 0 to d.Length - 1 do
            let c = int (d'.[i] * palettelf) - 1
            let c = max 0 c
            let c = min (palettel - 1) c
            sb.Append(palette.[c]) |> ignore
        sb.ToString()


/// Matrix numeric type keeping dual numbers for forward mode and adjoints and tapes for reverse mode AD, with nesting
/// capability, using tags to avoid perturbation confusion
and DM =
    /// Primal
    | DM of number[, ]
    /// Primal, tangent, layer tag (for forward mode)
    | DMF of primal: DM * tanget: DM * tag: uint32
    /// Primal, parent, layer tag (for reverse mode)
    | DMR of primal: DM * parentOperation: TraceOp * tag: uint32 * uniq: int32

    interface dobj

    /// Primal value of this DM
    member d.P =
        match d with
        | DM(_) -> d
        | DMF(ap, _, _) -> ap
        | DMR(ap, _, _, _) -> ap

    /// Deepest primal value of this DM
    member d.PD =
        let rec prec x =
            match x with
            | DM(_) -> x
            | DMF(xp, _, _) -> prec xp
            | DMR(xp, _, _, _) -> prec xp
        prec d

    /// Tangent value of this DM
    member d.T =
        match d with
        | DM(_) -> DM.ZeroMN d.Rows d.Cols
        | DMF(_, at, _) -> at
        | DMR _ -> failwith "Cannot get tangent value of DMR."

    member d.GetForward(t:DM, i:uint32) = DMF(d, t, i)

    member d.GetReverse(i:uint32) = DM.R(d, Noop, i)

    /// Make a reverse node
    static member R(cp, op, ai) = DMR(cp, op, ai, UniqueTagger.Next())

    member d.Length =
        match d with
        | DM(ap) -> ap.Length
        | DMF(ap, _, _) -> ap.Length
        | DMR(ap, _, _, _) -> ap.Length

    member d.Rows =
        match d with
        | DM(ap) -> Array2D.length1 ap
        | DMF(ap, _, _) -> ap.Rows
        | DMR(ap, _, _, _) -> ap.Rows

    member d.Cols =
        match d with
        | DM(ap) -> Array2D.length2 ap
        | DMF(ap, _, _) -> ap.Cols
        | DMR(ap, _, _, _) -> ap.Cols

    member d.Item
        with get (i, j) =
            match d with
            | DM(ap) -> D(ap.[i, j])
            | DMF(ap, at, ai) -> DF(ap.[i, j], at.[i, j], ai)
            | DMR(ap, _, ai, _) -> D.R(ap.[i, j], Item_DM(d, i, j), ai)

    member d.GetSlice(rowStart, rowFinish, colStart, colFinish) =
        let rowStart = defaultArg rowStart 0
        let rowFinish = defaultArg rowFinish (d.Rows - 1)
        let colStart = defaultArg colStart 0
        let colFinish = defaultArg colFinish (d.Cols - 1)
        match d with
        | DM(ap) -> DM(ap.[rowStart..rowFinish, colStart..colFinish])
        | DMF(ap, at, ai) -> DMF(ap.[rowStart..rowFinish, colStart..colFinish], at.[rowStart..rowFinish, colStart..colFinish], ai)
        | DMR(ap, _, ai, _) -> let cp = ap.[rowStart..rowFinish, colStart..colFinish] in DM.R(cp, Slice_DM(d, rowStart, colStart), ai)

    member d.GetSlice(row, colStart, colFinish) =
        let colStart = defaultArg colStart 0
        let colFinish = defaultArg colFinish (d.Cols - 1)
        match d with
        | DM(ap) -> DV(ap.[row, colStart..colFinish])
        | DMF(ap, at, ai) -> DVF(ap.[row, colStart..colFinish], at.[row, colStart..colFinish], ai)
        | DMR(ap, _, ai, _) -> let cp = ap.[row, colStart..colFinish] in DV.R(cp, SliceRow_DM(d, row, colStart), ai)

    member d.GetSlice(rowStart, rowFinish, col) =
        let rowStart = defaultArg rowStart 0
        let rowFinish = defaultArg rowFinish (d.Rows - 1)
        match d with
        | DM(ap) -> DV(ap.[rowStart..rowFinish, col])
        | DMF(ap, at, ai) -> DVF(ap.[rowStart..rowFinish, col], at.[rowStart..rowFinish, col], ai)
        | DMR(ap, _, ai, _) -> let cp = ap.[rowStart..rowFinish, col] in DV.R(cp, SliceCol_DM(d, rowStart, col), ai)

    member d.GetRows() =
        seq {for i = 0 to d.Rows - 1 do yield d.[i, *]}

    member d.GetCols() =
        seq {for j = 0 to d.Cols - 1 do yield d.[*, j]}

    override d.ToString() =
        let (d':number[, ]) = DM.op_Explicit(d)
        let sb = System.Text.StringBuilder()
        match d with
        | DM(_) -> sb.AppendLine(sprintf "DM : %i x %i" d.Rows d.Cols) |> ignore
        | DMF(_) -> sb.AppendLine(sprintf "DMF: %i x %i" d.Rows d.Cols) |> ignore
        | DMR(_) -> sb.AppendLine(sprintf "DMR: %i x %i" d.Rows d.Cols) |> ignore
        for i = 0 to d.Rows - 1 do
            for j = 0 to d.Cols - 1 do
                sb.Append(sprintf "% 9.3g " d'.[i, j]) |> ignore
            if i < d.Rows - 1 then sb.AppendLine() |> ignore
        sb.ToString()

    member d.ToMathematicaString() =
        let (d':number[, ]) = DM.op_Explicit(d)
        let sb = System.Text.StringBuilder()
        sb.Append("{") |> ignore
        for i = 0 to d.Rows - 1 do
            sb.Append("{") |> ignore
            for j = 0 to d.Cols - 1 do
                sb.Append(sprintf "%.2f" d'.[i, j]) |> ignore
                if j <> d.Cols - 1 then sb.Append(", ") |> ignore
            sb.Append("}") |> ignore
            if i <> d.Rows - 1 then sb.Append(", ") |> ignore
        sb.Append("}") |> ignore
        sb.ToString()

    member d.ToMatlabString() =
        let (d':number[, ]) = DM.op_Explicit(d)
        let sb = System.Text.StringBuilder()
        sb.Append("[") |> ignore
        for i = 0 to d.Rows - 1 do
            for j = 0 to d.Cols - 1 do
                sb.Append(sprintf "%.2f" d'.[i, j]) |> ignore
                if j < d.Cols - 1 then sb.Append(" ") |> ignore
            if i < d.Rows - 1 then sb.Append("; ") |> ignore
        sb.Append("]") |> ignore
        sb.ToString()

    static member Zero = DM Array2D.empty

    static member ZeroMN m n = DM (Array2D.zeroCreate m n)

    static member op_Explicit(d:DM):number[, ] =
        let rec prec x =
            match x with
            | DM(p) -> p
            | DMF(xp, _, _) -> prec xp
            | DMR(xp, _, _, _) -> prec xp
        prec d

    static member op_Explicit(d:number[, ]) = DM(d)

    static member OfArray2D (a:D[, ]) =
        // TODO: check to ensure that all elements in the array are of the same type (D, DF, or DR) and have the same nesting tag
        match a.[0, 0] with
        | D _ -> DM (a |> Array2D.map N.toNumber)
        | DF(_, _, ai) ->
            let ap = a |> Array2D.map (fun x -> x.P)
            let at = a |> Array2D.map (fun x -> x.T)
            DMF(DM.OfArray2D(ap), DM.OfArray2D(at), ai)
        | DR(_, _, ai, _) ->
            let ap = a |> Array2D.map (fun x -> x.P)
            let cp = DM.OfArray2D(ap) in DM.R(cp, Make_DM_ofDs(a), ai)

    // Creates a matrix with `m` rows from array `a`, filling columns from left to right and rows from top to bottom. The number of columns will be deduced from `m` and the length of `a`. The length of `a` must be an integer multiple of `m`.
    static member OfArray (m:int, a:D[]) =
        let n = a.Length / m
        Array2D.init m n (fun i j -> a.[i * n + j]) |> DM.OfArray2D

    static member OfRows (s:seq<DV>) =
        // TODO: check to ensure that all elements in the array are of the same type (D, DF, or DR) and have the same nesting tag
        match Seq.head s with
        | DV _ ->
            s |> Seq.map DV.op_Explicit |> array2D |> DM
        | DVF(_, _, ai) ->
            let ap = s |> Seq.map (fun x -> x.P)
            let at = s |> Seq.map (fun x -> x.T)
            DMF(DM.OfRows(ap), DM.OfRows(at), ai)
        | DVR(_, _, ai, _) ->
            let ap = s |> Seq.map (fun x -> x.P)
            let cp = DM.OfRows(ap) in DM.R(cp, Make_DMRows_ofDVs(s |> Seq.toArray), ai)

    static member OfRows (m:int, a:DV) =
        match a with
        | DV(ap) -> DM(Backend.RepeatReshapeCopy_V_MRows(m, ap))
        | DVF(ap, at, ai) -> DMF(DM.OfRows(m, ap), DM.OfRows(m, at), ai)
        | DVR(ap, _, ai, _) ->
            let cp = DM.OfRows(m, ap) in DM.R(cp, Make_DMRows_ofDV(a), ai)

    static member OfCols (n:int, a:DV) =
        match a with
        | DV(ap) -> DM(Backend.RepeatReshapeCopy_V_MCols(n, ap))
        | DVF(ap, at, ai) -> DMF(DM.OfCols(n, ap), DM.OfCols(n, at), ai)
        | DVR(ap, _, ai, _) ->
            let cp = DM.OfCols(n, ap) in DM.R(cp, Make_DMCols_ofDV(a), ai)

    static member inline Op_DM_DM (a, ff, fd, df, r) =
        match a with
        | DM(ap)                      -> DM(ff(ap))
        | DMF(ap, at, ai)             -> let cp = fd(ap) in DMF(cp, df(cp, ap, at), ai)
        | DMR(ap, _, ai, _)            -> let cp = fd(ap) in DM.R(cp, r(a), ai)

    static member inline Op_DM_DV (a, ff, fd, df, r) =
        match a with
        | DM(ap)                      -> DV(ff(ap))
        | DMF(ap, at, ai)             -> let cp = fd(ap) in DVF(cp, df(cp, ap, at), ai)
        | DMR(ap, _, ai, _)            -> let cp = fd(ap) in DV.R(cp, r(a), ai)

    static member inline Op_DM_D (a, ff, fd, df, r) =
        match a with
        | DM(ap)                      -> D(ff(ap))
        | DMF(ap, at, ai)             -> let cp = fd(ap) in DF(cp, df(cp, ap, at), ai)
        | DMR(ap, _, ai, _)            -> let cp = fd(ap) in D.R(cp, r(a), ai)

    static member inline Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | DM(bp)                  -> DM(ff(ap, bp))
            | DMF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DMR(bp, _, bi, _) -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi)
        | DMF(ap, at, ai) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap, _, ai, _) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DM.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | D(bp)                   -> DM(ff(ap, bp))
            | DF(bp, bt, bi)          -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DR(bp, _, bi, _)  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi)
        | DMF(ap, at, ai) ->
            match b with
            | D _                    -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap, _, ai, _) ->
            match b with
            | D _                    -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai)
            | DF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DM.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_D_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | D(ap) ->
            match b with
            | DM(bp)                  -> DM(ff(ap, bp))
            | DMF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DMR(bp, _, bi, _) -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi)
        | DF(ap, at, ai) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DR(ap, _, ai, _) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DM.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | DV(bp)                  -> DV(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp, _, bi, _) -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi)
        | DMF(ap, at, ai) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap, _, ai, _) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DV.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DV_DM_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DM(bp)                  -> DV(ff(ap, bp))
            | DMF(bp, bt, bi)         -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi)
            | DMR(bp, _, bi, _) -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi)
        | DVF(ap, at, ai) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DVF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DVF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap, _, ai, _) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DVF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DV.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DV.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DV.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DM_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DM(ap) ->
            match b with
            | DV(bp)                  -> DM(ff(ap, bp))
            | DVF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DVR(bp, _, bi, _) -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi)
        | DMF(ap, at, ai) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DMR(ap, _, ai, _) ->
            match b with
            | DV _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai)
            | DVF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DVR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DM.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi

    static member inline Op_DV_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d) =
        match a with
        | DV(ap) ->
            match b with
            | DM(bp)                  -> DM(ff(ap, bp))
            | DMF(bp, bt, bi)         -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi)
            | DMR(bp, _, bi, _) -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi)
        | DVF(ap, at, ai) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DMF(cp, df_dab(cp, ap, at, bp, bt), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DMF(cp, df_da(cp, ap, at), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
        | DVR(ap, _, ai, _) ->
            match b with
            | DM(_)                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai)
            | DMF(bp, bt, bi) ->
                match compare ai bi with
                | -1                  -> let cp = fd(a, bp) in DMF(cp, df_db(cp, bp, bt), bi) // ai < bi
                | 1                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi
                | _                   -> failwith "Forward and reverse AD cannot run on the same level."
            | DMR(bp, _, bi, _) ->
                match compare ai bi with
                | 0                   -> let cp = fd(ap, bp) in DM.R(cp, r_d_d(a, b), ai) // ai = bi
                | -1                  -> let cp = fd(a, bp) in DM.R(cp, r_c_d(a, b), bi) // ai < bi
                | _                   -> let cp = fd(ap, b) in DM.R(cp, r_d_c(a, b), ai) // ai > bi

    /// Element-wise addition of `a` and `b`
    static member (+) (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Add_M_M(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DM_DM(a, b)
        let inline r_d_c(a, b) = Add_DM_DMCons(a)
        let inline r_c_d(a, b) = Add_DM_DMCons(b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise addition of `a` and `b`, potentially destructive of the storage of raw matrices in 'b'
    static member internal Add_M_M_Inplace (a:DM, b:DM) =
        let inline ff(a:number[,], b:number[,]) = Backend.AlphaAdd_M_M_Inplace(N.one, a, b); b
        let inline fd(a:DM, b:DM) = a + b
        let inline df_da(cp:DM, ap:DM, at:DM) = at
        let inline df_db(cp:DM, bp:DM, bt:DM) = bt
        let inline df_dab(cp:DM, ap:DM, at:DM, bp:DM, bt:DM) = at + bt
        let inline r_d_d(a:DM, b:DM) = Add_DM_DM(a, b)
        let inline r_d_c(a:DM, b:DM) = Add_DM_DMCons(a)
        let inline r_c_d(a:DM, b:DM) = Add_DM_DMCons(b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise subtraction of `a` and `b`
    static member (-) (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Sub_M_M(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DM_DM(a, b)
        let inline r_d_c(a, b) = Sub_DM_DMCons(a)
        let inline r_c_d(a, b) = Sub_DMCons_DM(b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Matrix product of `a` and `b`
    static member (*) (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Mul_M_M(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_DM(a, b)
        let inline r_d_c(a, b) = Mul_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_DM(a, b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise (Hadamard, Schur) product of `a` and `b`
    static member (.*) (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Mul_Had_M_M(a, b)
        let inline fd(a, b) = a .* b
        let inline df_da(cp, ap, at) = at .* b
        let inline df_db(cp, bp, bt) = a .* bt
        let inline df_dab(cp, ap, at, bp, bt) = (at .* bp) + (ap .* bt)
        let inline r_d_d(a, b) = Mul_Had_DM_DM(a, b)
        let inline r_d_c(a, b) = Mul_Had_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Mul_Had_DM_DMCons(b, a)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Right-multiply matrix `a` by vector `b`
    static member (*) (a:DM, b:DV) =
        let inline ff(a, b) = Backend.Mul_M_V(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_DV(a, b)
        let inline r_d_c(a, b) = Mul_DM_DVCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_DV(a, b)
        DM.Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Left-multiply matrix `b` by vector `a`
    static member (*) (a:DV, b:DM) =
        let inline ff(a, b) = Backend.Mul_V_M(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DV_DM(a, b)
        let inline r_d_c(a, b) = Mul_DV_DMCons(a, b)
        let inline r_c_d(a, b) = Mul_DVCons_DM(a, b)
        DM.Op_DV_DM_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Element-wise (Hadamard, Schur) division `a` and `b`
    static member (./) (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Map2_F_M_M((/), a, b)
        let inline fd(a, b) = a ./ b
        let inline df_da(cp, ap, at) = at ./ b
        let inline df_db(cp, bp, bt) = -bt .* cp ./ bp // cp = ap / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt .* cp) ./ bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_Had_DM_DM(a, b)
        let inline r_d_c(a, b) = Div_Had_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Div_Had_DMCons_DM(a, b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Pow (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Map2_F_M_M((fun x y -> x ** y), a, b)
        let inline fd(a:DM, b:DM) = a ** b
        let inline df_da(cp:DM, ap:DM, at:DM) = at .* (ap ** (b - D.One)) .* b
        let inline df_db(cp, bp, bt) = bt .* cp .* log a // cp = a ** bp
        let inline df_dab(cp:DM, ap:DM, at:DM, bp:DM, bt:DM) = (ap ** (bp - D.One)) .* (at .* bp + ap .* bt .* log ap)
        let inline r_d_d(a, b) = Pow_DM_DM(a, b)
        let inline r_d_c(a, b) = Pow_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Pow_DMCons_DM(a, b)
        DM.Op_DM_DM_DM(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Atan2 (a:DM, b:DM) =
        let inline ff(a, b) = Backend.Map2_F_M_M(atan2, a, b)
        let inline fd(a, b) = atan2 a b
        let inline df_da(cp, ap, at) = (at .* b) ./ ((ap .* ap) + (b .* b))
        let inline df_db(cp, bp, bt) = (-bt .* a) ./ ((a .* a) + (bp .* bp))
        let inline df_dab(cp, ap, at, bp, bt) = ((at .* bp) - (bt .* ap)) ./ ((ap .* ap) + (bp .* bp))
        let inline r_d_d(a, b) = Atan2_DM_DM(a, b)
        let inline r_d_c(a, b) = Atan2_DM_DMCons(a, b)
        let inline r_c_d(a, b) = Atan2_DMCons_DM(a, b)
        DM.Op_DM_DM_DM(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:DM, b:D) =
        let inline ff(a, b) = Backend.Mul_S_M(b, a)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_D(a, b)
        let inline r_d_c(a, b) = Mul_DM_DCons(a, b)
        let inline r_c_d(a, b) = Mul_DMCons_D(a, b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (*) (a:D, b:DM) =
        let inline ff(a, b) = Backend.Mul_S_M(a, b)
        let inline fd(a, b) = a * b
        let inline df_da(cp, ap, at) = at * b
        let inline df_db(cp, bp, bt) = a * bt
        let inline df_dab(cp, ap, at, bp, bt) = (at * bp) + (ap * bt)
        let inline r_d_d(a, b) = Mul_DM_D(b, a)
        let inline r_d_c(a, b) = Mul_DM_DCons(b, a)
        let inline r_c_d(a, b) = Mul_DMCons_D(b, a)
        DM.Op_D_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (/) (a:DM, b:D) =
        let inline ff(a, b) = Backend.Mul_S_M(N.one / b, a)
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt * cp / bp // cp = a / bp
        let inline df_dab(cp, ap, at, bp, bt) = (at - bt * cp) / bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_DM_D(a, b)
        let inline r_d_c(a, b) = Div_DM_DCons(a, b)
        let inline r_c_d(a, b) = Div_DMCons_D(a, b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (/) (a:D, b:DM) =
        let inline ff(a, b) = Backend.Map_F_M((fun v -> a / v), b)
        let inline fd(a, b) = a / b
        let inline df_da(cp, ap, at) = at / b
        let inline df_db(cp, bp, bt) = -bt .* (cp ./ bp) // cp = a / bp
        let inline df_dab(cp:DM, ap:D, at:D, bp:DM, bt:DM) = (at - bt .* cp) ./ bp // cp = ap / bp
        let inline r_d_d(a, b) = Div_D_DM(a, b)
        let inline r_d_c(a, b) = Div_D_DMCons(a, b)
        let inline r_c_d(a, b) = Div_DCons_DM(a, b)
        DM.Op_D_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (+) (a:DM, b:D) =
        let inline ff(a, b) = Backend.Add_S_M(b, a)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.OfArray2D(Array2D.create a.Rows a.Cols bt)
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DM_D(a, b)
        let inline r_d_c(a, b) = Add_DM_DCons(a)
        let inline r_c_d(a, b) = Add_DMCons_D(b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (+) (a:D, b:DM) =
        let inline ff(a, b) = Backend.Add_S_M(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = DM.OfArray2D(Array2D.create b.Rows b.Cols at)
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DM_D(b, a)
        let inline r_d_c(a, b) = Add_DMCons_D(a)
        let inline r_c_d(a, b) = Add_DM_DCons(b)
        DM.Op_D_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:DM, b:D) =
        let inline ff(a, b) = Backend.Sub_M_S(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.OfArray2D(Array2D.create a.Rows a.Cols -bt)
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_DM_D(a, b)
        let inline r_d_c(a, b) = Sub_DM_DCons(a)
        let inline r_c_d(a, b) = Sub_DMCons_D(b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (+) (a:DV, b:DM) =
        let inline ff(a, b) = Backend.Add_V_MCols(a, b)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = DM.OfCols(b.Cols, at)
        let inline df_db(cp, bp, bt) = bt
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DMCols_DV(b, a)
        let inline r_d_c(a, b) = Add_DMColsCons_DV(a)
        let inline r_c_d(a, b) = Add_DMCols_DVCons(b)
        DM.Op_DV_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (+) (a:DM, b:DV) =
        let inline ff(a, b) = Backend.Add_V_MCols(b, a)
        let inline fd(a, b) = a + b
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.OfCols(a.Cols, bt)
        let inline df_dab(cp, ap, at, bp, bt) = at + bt
        let inline r_d_d(a, b) = Add_DMCols_DV(a, b)
        let inline r_d_c(a, b) = Add_DMCols_DVCons(a)
        let inline r_c_d(a, b) = Add_DMColsCons_DV(b)
        DM.Op_DM_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:D, b:DM) =
        let inline ff(a, b) = Backend.Sub_S_M(a, b)
        let inline fd(a, b) = a - b
        let inline df_da(cp, ap, at) = DM.OfArray2D(Array2D.create b.Rows b.Cols at)
        let inline df_db(cp, bp, bt) = -bt
        let inline df_dab(cp, ap, at, bp, bt) = at - bt
        let inline r_d_d(a, b) = Sub_D_DM(a, b)
        let inline r_d_c(a, b) = Sub_D_DMCons(a)
        let inline r_c_d(a, b) = Sub_DCons_DM(b)
        DM.Op_D_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member (-) (a:DV, b:DM) = a + -b
    static member (-) (a:DM, b:DV) = a + -b

    static member Pow (a:DM, b:D) =
        let inline ff(a, b) = Backend.Map_F_M((fun v -> v ** b), a)
        let inline fd(a:DM, b:D) = a ** b
        let inline df_da(cp, ap:DM, at:DM) = at .* (ap ** (b - D.One)) * b
        let inline df_db(cp, bp, bt) = bt * cp .* log a // cp = a ** bp
        let inline df_dab(cp, ap:DM, at:DM, bp:D, bt:D) = (ap ** (bp - D.One)) .* ((at * bp) + (ap * bt .* log ap))
        let inline r_d_d(a, b) = Pow_DM_D(a, b)
        let inline r_d_c(a, b) = Pow_DM_DCons(a, b)
        let inline r_c_d(a, b) = Pow_DMCons_D(a, b)
        DM.Op_DM_D_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Pow (a:D, b:DM) =
        let inline ff(a, b) = Backend.Map_F_M((fun v -> a ** v), b)
        let inline fd(a:D, b:DM) = DM.Pow(a, b)
        let inline df_da(cp, ap:D, at:D) = at * (DM.Pow(ap, b - D.One)) .* b
        let inline df_db(cp, bp, bt) = bt .* cp * log a // cp = a ** bp
        let inline df_dab(cp, ap:D, at:D, bp:DM, bt:DM) = (DM.Pow(ap, bp - D.One)) .* ((at * bp) + (ap * bt * log ap))
        let inline r_d_d(a, b) = Pow_D_DM(a, b)
        let inline r_d_c(a, b) = Pow_D_DMCons(a, b)
        let inline r_c_d(a, b) = Pow_DCons_DM(a, b)
        DM.Op_D_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Atan2 (a:DM, b:D) =
        let inline ff(a, b) = Backend.Map_F_M((fun v -> atan2 v b), a)
        let inline fd(a:DM, b:D) = DM.Atan2(a, b)
        let inline df_da(cp, ap, at) = (at * b) ./ ((ap .* ap) + (b * b))
        let inline df_db(cp, bp, bt) = (-bt * a) ./ ((a .* a) + (bp * bp))
        let inline df_dab(cp, ap, at, bp, bt) = ((at * bp) - (bt * ap)) ./ ((ap .* ap) + (bp * bp))
        let inline r_d_d(a, b) = Atan2_DM_D(a, b)
        let inline r_d_c(a, b) = Atan2_DM_DCons(a, b)
        let inline r_c_d(a, b) = Atan2_DMCons_D(a, b)
        DM.Op_DM_D_DM(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member Atan2 (a:D, b:DM) =
        let inline ff(a, b) = Backend.Map_F_M((fun v -> atan2 a v), b)
        let inline fd(a:D, b:DM) = DM.Atan2(a, b)
        let inline df_da(cp, ap, at) = (at * b) ./ ((ap * ap) + (b .* b))
        let inline df_db(cp, bp, bt) = (-bt * a) ./ ((a * a) + (bp .* bp))
        let inline df_dab(cp, ap, at, bp, bt) = ((at * bp) - (bt * ap)) ./ ((ap * ap) + (bp .* bp))
        let inline r_d_d(a, b) = Atan2_D_DM(a, b)
        let inline r_d_c(a, b) = Atan2_D_DMCons(a, b)
        let inline r_c_d(a, b) = Atan2_DCons_DM(a, b)
        DM.Op_D_DM_DM(a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    // DM - number binary operations
    static member (+) (a:DM, b:number) = a + D b
    static member (-) (a:DM, b:number) = a - D b
    static member (*) (a:DM, b:number) = a * D b
    static member (/) (a:DM, b:number) = a / D b
    static member Pow (a:DM, b:number) = a ** D b
    static member Atan2 (a:DM, b:number) = DM.Atan2(a, D b)

    // number - DM binary operations
    static member (+) (a:number, b:DM) = (D a) + b
    static member (-) (a:number, b:DM) = (D a) - b
    static member (*) (a:number, b:DM) = (D a) * b
    static member (/) (a:number, b:DM) = (D a) / b
    static member Pow (a:number, b:DM) = DM.Pow(D a, b)
    static member Atan2 (a:number, b:DM) = DM.Atan2(D a, b)

    // DM - int binary operations
    static member (+) (a:DM, b:int) = a + D (N.toNumber b)
    static member (-) (a:DM, b:int) = a - D (N.toNumber b)
    static member (*) (a:DM, b:int) = a * D (N.toNumber b)
    static member (/) (a:DM, b:int) = a / D (N.toNumber b)
    static member Pow (a:DM, b:int) = a ** D (N.toNumber b)
    static member Atan2 (a:DM, b: int) = DM.Atan2(a, D (N.toNumber b))

    // int - DM binary operations
    static member (+) (a:int, b:DM) = (D (N.toNumber a)) + b
    static member (-) (a:int, b:DM) = (D (N.toNumber a)) - b
    static member (*) (a:int, b:DM) = (D (N.toNumber a)) * b
    static member (/) (a:int, b:DM) = (D (N.toNumber a)) / b
    static member Pow (a:int, b:DM) = DM.Pow(D (N.toNumber a), b)
    static member Atan2 (a:int, b:DM) = DM.Atan2(D (N.toNumber a), b)

    static member Log (a:DM) =
        let inline ff(a) = Backend.Map_F_M(log, a)
        let inline fd(a) = log a
        let inline df(cp, ap, at) = at ./ ap
        let inline r(a) = Log_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Log10 (a:DM) =
        let inline ff(a) = Backend.Map_F_M(log10, a)
        let inline fd(a) = log10 a
        let inline df(cp, ap:DM, at:DM) = at ./ (ap * N.log10Val)
        let inline r(a) = Log10_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Exp (a:DM) =
        let inline ff(a) = Backend.Map_F_M(exp, a)
        let inline fd(a) = exp a
        let inline df(cp, ap, at) = at .* cp // cp = exp ap
        let inline r(a) = Exp_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Sin (a:DM) =
        let inline ff(a) = Backend.Map_F_M(sin, a)
        let inline fd(a) = sin a
        let inline df(cp, ap:DM, at:DM) = at .* cos ap
        let inline r(a) = Sin_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Cos (a:DM) =
        let inline ff(a) = Backend.Map_F_M(cos, a)
        let inline fd(a) = cos a
        let inline df(cp, ap:DM, at:DM) = -at .* sin ap
        let inline r(a) = Cos_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Tan (a:DM) =
        let inline ff(a) = Backend.Map_F_M(tan, a)
        let inline fd(a) = tan a
        let inline df(cp, ap:DM, at:DM) = let cosa = cos ap in at ./ (cosa .* cosa)
        let inline r(a) = Tan_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member (~-) (a:DM) =
        let inline ff(a) = Backend.Mul_S_M(N.minus1, a)
        let inline fd(a) = -a
        let inline df(cp, ap, at) = -at
        let inline r(a) = Neg_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Sqrt (a:DM) =
        let inline ff(a) = Backend.Map_F_M(sqrt, a)
        let inline fd(a) = sqrt a
        let inline df(cp:DM, ap:DM, at:DM) = at ./ (D N.two * cp) // cp = sqrt ap
        let inline r(a) = Sqrt_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Sinh (a:DM) =
        let inline ff(a) = Backend.Map_F_M(sinh, a)
        let inline fd(a) = sinh a
        let inline df(cp:DM, ap:DM, at:DM) = at .* cosh ap
        let inline r(a) = Sinh_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Cosh (a:DM) =
        let inline ff(a) = Backend.Map_F_M(cosh, a)
        let inline fd(a) = cosh a
        let inline df(cp:DM, ap:DM, at:DM) = at .* sinh ap
        let inline r(a) = Cosh_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Tanh (a:DM) =
        let inline ff(a) = Backend.Map_F_M(tanh, a)
        let inline fd(a) = tanh a
        let inline df(cp:DM, ap:DM, at:DM) = let cosha = cosh ap in at ./ (cosha .* cosha)
        let inline r(a) = Tanh_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Asin (a:DM) =
        let inline ff(a) = Backend.Map_F_M(asin, a)
        let inline fd(a) = asin a
        let inline df(cp:DM, ap:DM, at:DM) = at ./ sqrt (D.One - (ap .* ap))
        let inline r(a) = Asin_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Acos (a:DM) =
        let inline ff(a) = Backend.Map_F_M(acos, a)
        let inline fd(a) = acos a
        let inline df(cp:DM, ap:DM, at:DM) = -at ./ sqrt (D.One - (ap .* ap))
        let inline r(a) = Acos_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Atan (a:DM) =
        let inline ff(a) = Backend.Map_F_M(atan, a)
        let inline fd(a) = atan a
        let inline df(cp:DM, ap:DM, at:DM) = at ./ sqrt (D.One + (ap .* ap))
        let inline r(a) = Atan_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Abs (a:DM) =
        let inline ff(a) = Backend.Map_F_M(abs, a)
        let inline fd(a) = abs a
        let inline df(cp, ap, at) = at .* (DM.Sign ap)
        let inline r(a) = Abs_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Sign (a:DM) =
        let inline ff(a) = Backend.Map_F_M(signummod, a)
        let inline fd(a) = DM.Sign a
        let inline df(cp, ap, at) = DM.ZeroMN a.Rows a.Cols
        let inline r(a) = Sign_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Floor (a:DM) =
        let inline ff(a) = Backend.Map_F_M(floor, a)
        let inline fd(a) = floor a
        let inline df(cp, ap, at) = DM.ZeroMN a.Rows a.Cols
        let inline r(a) = Floor_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Ceiling (a:DM) =
        let inline ff(a) = Backend.Map_F_M(ceil, a)
        let inline fd(a) = ceil a
        let inline df(cp, ap, at) = DM.ZeroMN a.Rows a.Cols
        let inline r(a) = Ceil_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Round (a:DM) =
        let inline ff(a) = Backend.Map_F_M(round, a)
        let inline fd(a) = round a
        let inline df(cp, ap, at) = DM.ZeroMN a.Rows a.Cols
        let inline r(a) = Round_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    /// Transpose of matrix `a`
    static member Transpose(a:DM) =
        let inline ff(a) = Backend.Transpose_M(a)
        let inline fd(a) = DM.Transpose(a)
        let inline df(cp, ap, at) = DM.Transpose(at)
        let inline r(a) = Transpose_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    /// Diagonal of matrix `a`
    static member Diagonal(a:DM) =
        let inline ff(a) = Backend.Diagonal_M(a)
        let inline fd(a) = DM.Diagonal(a)
        let inline df(cp, ap, at) = DM.Diagonal(at)
        let inline r(a) = Diagonal_DM(a)
        DM.Op_DM_DV (a, ff, fd, df, r)

    /// Trace of matrix `a`
    static member Trace(a:DM) =
        DV.Sum(DM.Diagonal(a))

    /// Sum of the entries of matrix `a`
    static member Sum(a:DM) =
        let inline ff(a) = Backend.Sum_M(a)
        let inline fd(a) = DM.Sum(a)
        let inline df(cp, ap, at) = DM.Sum(at)
        let inline r(a) = Sum_DM(a)
        DM.Op_DM_D (a, ff, fd, df, r)

    /// Solve a system of linear equations Ax = b, where the coefficient matrix `a` has general form
    static member Solve (a:DM, b:DV) =
        let inline ff(a, b) = match Backend.Solve_M_V(a, b) with Some(x) -> x | _ -> ErrorMessages.InvalidArgSolve()
        let inline fd(a, b) = DM.Solve(a, b)
        let inline df_da(cp, ap, at) = DM.Solve(ap, -at * cp) // cp = DM.Solve(ap, b)
        let inline df_db(cp, bp, bt) = DM.Solve(a, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.Solve(ap, bt - at * cp) // cp = DM.Solve(ap, bp)
        let inline r_d_d(a, b) = Solve_DM_DV(a, b)
        let inline r_d_c(a, b) = Solve_DM_DVCons(a, b)
        let inline r_c_d(a, b) = Solve_DMCons_DV(a, b)
        DM.Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Solve a system of linear equations Ax = b, where the coefficient matrix `a` is symmetric
    static member SolveSymmetric (a:DM, b:DV) =
        let inline ff(a, b) = match Backend.SolveSymmetric_M_V(a, b) with Some(x) -> x | _ -> ErrorMessages.InvalidArgSolve()
        let inline fd(a, b) = DM.SolveSymmetric(a, b)
        let inline df_da(cp, ap, at) = DM.SolveSymmetric(ap, -at * cp) // cp = DM.Solve(ap, b)
        let inline df_db(cp, bp, bt) = DM.SolveSymmetric(a, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.SolveSymmetric(ap, bt - at * cp) // cp = DM.Solve(ap, bp)
        let inline r_d_d(a, b) = Solve_DM_DV(a, b)
        let inline r_d_c(a, b) = Solve_DM_DVCons(a, b)
        let inline r_c_d(a, b) = Solve_DMCons_DV(a, b)
        DM.Op_DM_DV_DV (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Add scalar `b` to matrix `a` at row `i` and column `j`
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

    /// Add submatrix `b` to matrix `a`, where the upper left corner of `b` is positioned at row `i` and column `j`
    static member AddSubMatrix (a:DM, i:int, j:int, b:DM) =
        let inline ff(a:number[, ], bb:number[, ]) =
            let aa = Array2D.copy a
//            Parallel.For(0, b.Rows, fun ii ->
//                Parallel.For(0, b.Cols, fun jj ->
//                    aa.[i + ii, j + jj] <- aa.[i + ii, j + jj] + bb.[ii, jj]) |> ignore) |> ignore
            for ii = 0 to b.Rows - 1 do
                for jj = 0 to b.Cols - 1 do
                    aa.[i + ii, j + jj] <- aa.[i + ii, j + jj] + bb.[ii, jj]
            aa
        let inline fd(a, b) = DM.AddSubMatrix(a, i, j, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.AddSubMatrix(DM.ZeroMN a.Rows a.Cols, i, j, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.AddSubMatrix(at, i, j, bt)
        let inline r_d_d(a, b) = AddSubMatrix_DM_DM(a, i, j, b)
        let inline r_d_c(a, b) = AddSubMatrix_DM_DMCons(a)
        let inline r_c_d(a, b) = AddSubMatrix_DMCons_DM(i, j, b)
        DM.Op_DM_DM_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    /// Add the elements of vector `b` to the diagonal elements of matrix `a`
    static member AddDiagonal (a:DM, b:DV) =
        let inline ff(a:number[, ], b:number[]) =
            let aa = Array2D.copy a
            let n = min (Array2D.length1 a) (Array2D.length2 a) |> min b.Length
            for i = 0 to n - 1 do
                aa.[i, i] <- aa.[i, i] + b.[i]
            aa
        let inline fd(a, b) = DM.AddDiagonal(a, b)
        let inline df_da(cp, ap, at) = at
        let inline df_db(cp, bp, bt) = DM.AddDiagonal(DM.ZeroMN a.Rows a.Cols, bt)
        let inline df_dab(cp, ap, at, bp, bt) = DM.AddDiagonal(at, bt)
        let inline r_d_d(a, b) = AddDiagonal_DM_DV(a, b)
        let inline r_d_c(a, b) = AddDiagonal_DM_DVCons(a)
        let inline r_c_d(a, b) = AddDiagonal_DMCons_DV(b)
        DM.Op_DM_DV_DM (a, b, ff, fd, df_da, df_db, df_dab, r_d_d, r_d_c, r_c_d)

    static member ReshapeToDV(a:DM) =
        let inline ff(a) = Backend.ReshapeCopy_MRows_V(a)
        let inline fd(a) = DM.ReshapeToDV(a)
        let inline df(cp, ap, at) = DM.ReshapeToDV(at)
        let inline r(a) = ReshapeCopy_DM_DV(a)
        DM.Op_DM_DV (a, ff, fd, df, r)

    /// Matrix inverse of `a`
    static member Inverse(a:DM) =
        let inline ff(a) = match Backend.Inverse_M(a) with Some(x) -> x | _ -> ErrorMessages.InvalidArgInverse()
        let inline fd(a) = DM.Inverse(a)
        let inline df(cp, ap, at) = -cp * at * cp
        let inline r(a) = Inverse_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    /// Determinant of matrix `a`
    static member Det(a:DM) =
        let inline ff(a) = match Backend.Det_M(a) with Some(x) -> x | _ -> ErrorMessages.InvalidArgDet()
        let inline fd(a) = DM.Det(a)
        let inline df(cp, ap, at) = cp * DM.Trace(DM.Inverse(ap) * at)
        let inline r(a) = Det_DM(a)
        DM.Op_DM_D (a, ff, fd, df, r)

    static member ReLU (a:DM) =
        let inline ff(a) = Backend.Map_F_M(max N.zero, a)
        let inline fd(a) = DM.ReLU(a)
        let inline df(cp, ap, at) = at .* ((N.one + DM.Sign(ap)) / N.two)
        let inline r(a) = ReLU_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member Sigmoid (a:DM) =
        let inline ff(a) = Backend.Map_F_M((fun v -> N.one / (N.one + exp -v)), a)
        let inline fd(a) = DM.Sigmoid(a)
        let inline df(cp:DM, ap, at) = at .* cp .* (N.one - cp)
        let inline r(a) = Sigmoid_DM(a)
        DM.Op_DM_DM (a, ff, fd, df, r)

    static member SoftPlus (a:DM) = log (N.one + exp a)
    static member SoftSign (a:DM) = a ./ (N.one + abs a)

    static member Mean (a:DM) =
        DM.Sum(a) / a.Length
    static member Variance (a:DM) =
        let a' = a - DM.Mean(a)
        DM.Sum(a' .* a') / (a.Length - 1)
    static member StandardDev (a:DM) =
        DM.Variance(a) |> sqrt
    static member Standardize (a:DM) =
        let sd = DM.StandardDev(a)
        if sd = D.Zero then
            a * (D.Zero)
        else
            (a - DM.Mean(a)) / DM.StandardDev(a)
    static member Normalize (a:DM) =
        let min = DM.Min(a)
        let range = DM.Max(a) - min
        if range = D.Zero then
            a * (D.Zero)
        else
            (a - min) / range

    static member Max (a:DM, b:DM) = ((a + b) + abs (b - a)) / N.two
    static member Max (a:DM, b:D) = ((a + b) + abs (b - a)) / N.two
    static member Max (a:D, b:DM) = ((a + b) + abs (b - a)) / N.two
    static member Min (a:DM, b:DM) = ((a + b) - abs (a - b)) / N.two
    static member Min (a:DM, b:D) = ((a + b) - abs (a - b)) / N.two
    static member Min (a:D, b:DM) = ((a + b) - abs (a - b)) / N.two

    /// Index of the maximum element of matrix `a`
    static member MaxIndex (a:DM) =
        let a' = DM.op_Explicit(a)
        let mutable maxij = 0, 0
        let mutable maxv = a'.[0, 0]
        for i = 0 to a.Rows - 1 do
            for j = 0 to a.Cols - 1 do
                if a'.[i, j] > maxv then maxij <- (i, j); maxv <- a'.[i, j]
        maxij
    static member Max (a:DM) = let maxij = DM.MaxIndex(a) in a.[fst maxij, snd maxij]

    /// Index of the minimum element of matrix `a`
    static member MinIndex (a:DM) =
        let a' = DM.op_Explicit(a)
        let mutable minij = 0, 0
        let mutable minv = a'.[0, 0]
        for i = 0 to a.Rows - 1 do
            for j = 0 to a.Cols - 1 do
                if a'.[i, j] < minv then minij <- (i, j); minv <- a'.[i, j]
        minij
    static member Min (a:DM) = let minij = DM.MinIndex(a) in a.[fst minij, snd minij]

    member d.Visualize() =
        let (d':number[, ]) = ((VisualizationContrast * (DM.Normalize(d.P) - N.half)) + N.half) |> DM.op_Explicit
        let sb = System.Text.StringBuilder()
        match d with
        | DM(_) -> sb.AppendLine(sprintf "DM : %i x %i" d.Rows d.Cols) |> ignore
        | DMF(_) -> sb.AppendLine(sprintf "DMF: %i x %i" d.Rows d.Cols) |> ignore
        | DMR(_) -> sb.AppendLine(sprintf "DMR: %i x %i" d.Rows d.Cols) |> ignore
        let palette = GlobalConfig.GrayscalePalette
        let palettel = palette.Length
        let palettelf = N.toNumber palettel
        for i = 0 to d.Rows - 1 do
            for j = 0 to d.Cols - 1 do
                let c = int (d'.[i, j] * palettelf) - 1
                let c = max 0 c
                let c = min (palettel - 1) c
                sb.Append(palette.[c]) |> ignore
            if i < d.Rows - 1 then sb.AppendLine() |> ignore
        sb.ToString()


/// Operation types recorded in the evaluation trace
and TraceOp =
    // Scalar-valued operations
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
    | ReLU_D                 of D
    | Sigmoid_D              of D
    | LogSumExp_DV           of DV
    | FixedPoint_D           of D * D * D * D

    // Vector-valued operations
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
    | Mul_DV_DM              of DV * DM
    | Mul_DV_DMCons          of DV * DM
    | Mul_DVCons_DM          of DV * DM
    | Div_Had_DV_DV          of DV * DV
    | Div_Had_DV_DVCons      of DV * DV
    | Div_Had_DVCons_DV      of DV * DV
    | Div_DV_D               of DV * D
    | Div_DV_DCons           of DV * D
    | Div_DVCons_D           of DV * D
    | Div_D_DV               of D * DV
    | Div_D_DVCons           of D * DV
    | Div_DCons_DV           of D * DV
    | Pow_DV_DV              of DV * DV
    | Pow_DV_DVCons          of DV * DV
    | Pow_DVCons_DV          of DV * DV
    | Atan2_DV_DV            of DV * DV
    | Atan2_DV_DVCons        of DV * DV
    | Atan2_DVCons_DV        of DV * DV
    | Pow_DV_D               of DV * D
    | Pow_DV_DCons           of DV * D
    | Pow_DVCons_D           of DV * D
    | Pow_D_DV               of D * DV
    | Pow_D_DVCons           of D * DV
    | Pow_DCons_DV           of D * DV
    | Atan2_DV_D             of DV * D
    | Atan2_DV_DCons         of DV * D
    | Atan2_DVCons_D         of DV * D
    | Atan2_D_DV             of D * DV
    | Atan2_D_DVCons         of D * DV
    | Atan2_DCons_DV         of D * DV
    | Exp_DV                 of DV
    | Log_DV                 of DV
    | Log10_DV               of DV
    | Sin_DV                 of DV
    | Cos_DV                 of DV
    | Tan_DV                 of DV
    | Neg_DV                 of DV
    | Sqrt_DV                of DV
    | Sinh_DV                of DV
    | Cosh_DV                of DV
    | Tanh_DV                of DV
    | Asin_DV                of DV
    | Acos_DV                of DV
    | Atan_DV                of DV
    | Abs_DV                 of DV
    | Sign_DV                of DV
    | Floor_DV               of DV
    | Ceil_DV                of DV
    | Round_DV               of DV
    | Make_DV_ofDs            of D[]
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
    | ReshapeCopy_DM_DV      of DM
    | Slice_DV               of DV * int
    | Diagonal_DM            of DM
    | ReLU_DV                of DV
    | Sigmoid_DV             of DV

    // Matrix-valued operations
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
    | Div_Had_DM_DM          of DM * DM
    | Div_Had_DM_DMCons      of DM * DM
    | Div_Had_DMCons_DM      of DM * DM
    | Pow_DM_DM              of DM * DM
    | Pow_DM_DMCons          of DM * DM
    | Pow_DMCons_DM          of DM * DM
    | Atan2_DM_DM            of DM * DM
    | Atan2_DM_DMCons        of DM * DM
    | Atan2_DMCons_DM        of DM * DM
    | Div_DM_D               of DM * D
    | Div_DM_DCons           of DM * D
    | Div_DMCons_D           of DM * D
    | Div_D_DM               of D * DM
    | Div_D_DMCons           of D * DM
    | Div_DCons_DM           of D * DM
    | Add_DM_D               of DM * D
    | Add_DM_DCons           of DM
    | Add_DMCons_D           of D
    | Add_DMCols_DV          of DM * DV
    | Add_DMCols_DVCons      of DM
    | Add_DMColsCons_DV      of DV
    | Sub_DM_D               of DM * D
    | Sub_DM_DCons           of DM
    | Sub_DMCons_D           of D
    | Sub_D_DM               of D * DM
    | Sub_D_DMCons           of D
    | Sub_DCons_DM           of DM
    | Pow_DM_D               of DM * D
    | Pow_DM_DCons           of DM * D
    | Pow_DMCons_D           of DM * D
    | Pow_D_DM               of D * DM
    | Pow_D_DMCons           of D * DM
    | Pow_DCons_DM           of D * DM
    | Atan2_DM_D             of DM * D
    | Atan2_DM_DCons         of DM * D
    | Atan2_DMCons_D         of DM * D
    | Atan2_D_DM             of D * DM
    | Atan2_D_DMCons         of D * DM
    | Atan2_DCons_DM         of D * DM
    | Exp_DM                 of DM
    | Log_DM                 of DM
    | Log10_DM               of DM
    | Sin_DM                 of DM
    | Cos_DM                 of DM
    | Tan_DM                 of DM
    | Neg_DM                 of DM
    | Sqrt_DM                of DM
    | Sinh_DM                of DM
    | Cosh_DM                of DM
    | Tanh_DM                of DM
    | Asin_DM                of DM
    | Acos_DM                of DM
    | Atan_DM                of DM
    | Abs_DM                 of DM
    | Sign_DM                of DM
    | Floor_DM               of DM
    | Ceil_DM                of DM
    | Round_DM               of DM
    | Transpose_DM           of DM
    | Make_DM_ofDs           of D[, ]
    | Make_DMRows_ofDV       of DV
    | Make_DMCols_ofDV       of DV
    | Make_DMRows_ofDVs      of DV[]
    | AddItem_DM_D           of DM * int * int * D
    | AddItem_DM_DCons       of DM
    | AddItem_DMCons_D       of int * int * D
    | AddSubMatrix_DM_DM     of DM * int * int * DM
    | AddSubMatrix_DM_DMCons of DM
    | AddSubMatrix_DMCons_DM of int * int * DM
    | Slice_DM               of DM * int * int
    | RowMatrix_DV           of DV
    | AddDiagonal_DM_DV      of DM * DV
    | AddDiagonal_DM_DVCons  of DM
    | AddDiagonal_DMCons_DV  of DV
    | ReshapeCopy_DV_DM      of DV
    | Inverse_DM             of DM
    | Det_DM                 of DM
    | ReLU_DM                of DM
    | Sigmoid_DM             of DM

    | Noop


/// A constraint used to ensure the evaluation stack is only over D, DV or DM
and dobj = interface end


/// Functional-oriented operations on vectors. Implementing functionality similar to FSharp.Collections.Array.
[<RequireQualifiedAccess>]
[<CompilationRepresentation (CompilationRepresentationFlags.ModuleSuffix)>]
module DV =

    // Note: map operations are not implemented on purpose. To benefit from the performance of BLAS ops, supplied element-wise operations are used. For example: "exp v" instead of "DV.map exp v"
    /// Creates a vector from array `a`
    let inline ofArray a = DV.OfArray(a)

    /// Converts vector `v` into an array
    let inline toArray (v:DV) = v.ToArray()

    /// Converts vector `v` into a row matrix
    let inline toRowDM (v:DV) = v.ToRowDM()

    /// Converts vector `v` into a column matrix
    let inline toColDM (v:DV) = v.ToColDM()

    /// Creates a vector with `n` elements, each with value `v`
    let inline create n (v:'a) =
        let at = typeof<'a>
        if at.Equals(typeof<D>) then DV.OfArray(Array.create n (unbox<D>(box v)))
        elif at.Equals(typeof<number>) then DV (Array.create n (unbox<number>(box v)))
        elif at.Equals(typeof<int>) then DV (Array.create n (unbox<int>(box v) |> N.toNumber))
        else N.failWithInvalidTypeMessage ()

    /// Creates a vector with `n` zero elements
    let inline zeroCreate n = DV.ZeroN n

    /// Empty vector
    let empty = DV.Zero

    /// Creates a vector of `n` elements, where each element is defined by function `f`
    let inline init n (f:int->'a) =
        let at = typeof<'a>
        if at.Equals(typeof<D>) then DV.OfArray(Array.init n (unbox<int->D>(box f)))
        elif at.Equals(typeof<number>) then DV (Array.init n (unbox<int->number>(box f)))
        elif at.Equals(typeof<int>) then DV ((Array.init n (unbox<int->int>(box f))) |> Array.map N.toNumber)
        else N.failWithInvalidTypeMessage ()

    /// Returns true if vector `v` is empty, otherwise returns false
    let isEmpty (v:DV) = v.Length = 0

    /// Iterates function `f` over the elements of vector `v`
    let inline iter (f:D->unit) (v:DV) = v |> toArray |> Array.iter f

    /// Iterates function `f` over the elements of vector `v`. An element index is also supplied to `f`.
    let inline iteri (f:int->D->unit) (v:DV) = v |> toArray |> Array.iteri f

    /// Iterates function `f` over the elements of vectors `v1` and `v2`
    let inline iter2 (f:D->D->unit) (v1:DV) (v2:DV) = Array.iter2 f (v1 |> toArray) (v2 |> toArray)

    /// Iterates function `f` over the elements of vectors `v1` and `v2`. An element index is also supplied to `f`.
    let inline iteri2 (f:int->D->D->unit) (v1:DV) (v2:DV) = Array.iteri2 f (v1 |> toArray) (v2 |> toArray)

    /// Length of vector `v`
    let inline length (v:DV) = v.Length

    /// L1 norm of vector `v`
    let inline l1norm (v:DV) = DV.L1Norm(v)

    /// L2 norm of vector `v`
    let inline l2norm (v:DV) = DV.L2Norm(v)

    /// Squared L2 norm of vector `v`
    let inline l2normSq (v:DV) = DV.L2NormSq(v)

    /// Maximum of the elements of vector `v`
    let inline max (v:DV) = DV.Max(v)

    /// Index of the maximum element of vector `v`
    let inline maxIndex (v:DV) = DV.MaxIndex(v)

    /// Minimum of the elements of vector `v`
    let inline min (v:DV) = DV.Min(v)

    /// Index of the minimum element of vector `v`
    let inline minIndex (v:DV) = DV.MinIndex(v)

    /// Mean of vector `v`
    let inline mean (v:DV) = DV.Mean(v)

    /// Average of vector `v`. Same with mean.
    let average = mean

    /// Standard deviation of vector `v`
    let inline standardDev (v:DV) = DV.StandardDev(v)

    /// Variance of vector `v`
    let inline variance (v:DV) = DV.Variance(v)

    /// Shift and scale the elements of vector `v` to have zero mean and unit variance
    let inline standardize (v:DV) = DV.Standardize(v)

    /// Shift and scale the elements of vector `v` to be in the range [0, 1]
    let inline normalize (v:DV) = DV.Normalize(v)

    /// L2 norm of vector `v`. Same with DV.l2norm.
    let inline norm (v:DV) = DV.L2Norm(v)

    /// Squared L2 norm of vector `v`. Same with DV.l2normSq.
    let inline normSq(v:DV) = DV.L2NormSq(v)

    // TODO: implement supNorm (infinity norm, with BLAS IDAMAX)
    /// Creates a vector where elements of `v1` are followed by elements of `v2`
    let inline append (v1:DV) (v2:DV) = DV.Append(v1, v2)

    /// Creates a vector where elements of `v2` are followed by elements of `v1`
    let inline prepend (v1:DV) (v2:DV) = DV.Append(v2, v1)

    /// Concatenates the given sequence of vectors `v` into one vector
    let inline concat (v:seq<DV>) = Seq.fold append DV.Zero v

    /// Splits vector `v` into a sequence of subvectors whose lengths are given in sequence `n`
    let inline split (n:seq<int>) (v:DV) = DV.Split(v, n)

    /// Splits vector `v` into `n` subvectors of equal length. The length of vector `v` must be an integer multiple of `n`.
    let inline splitEqual (n:int) (v:DV) = DV.Split(v, Array.create n (v.Length / n))

    /// Sums the elements of vector `v`
    let inline sum (v:DV) = DV.Sum(v)

    /// Creates a vector with `n` elements where the `i`-th element is one and the rest of the elements are zero
    let inline standardBasis (n:int) (i:int) = DV(standardBasis n i)

    /// Creates a vector with `n` elements where the `i`-th element has value `v` and the rest of the elements are zero
    let inline standardBasisVal (n:int) (i:int) (v:number) = DV(standardBasisVal n i v)

    /// Gets the unit vector codirectional with vector `v`
    let inline unitDV (v:DV) = v / DV.L2Norm(v)

    /// Converts matrix `m` into a vector by stacking its rows
    let inline ofDM (m:DM) = DM.ReshapeToDV(m)

    /// Creates a matrix with `m` rows from vector `v`
    let inline toDM (m:int) (v:DV) = DV.ReshapeToDM(m, v)

    // Experimental
    let inline toString (v:DV) = v.ToString()
    let inline visualize (v:DV) = v.Visualize()
    let inline visualizeAsDM (m:int) (v:DV) = DV.ReshapeToDM(m, v).Visualize()


/// Functional-oriented operations on matrices. Implementing functionality similar to FSharp.Collections.Array2D.
[<RequireQualifiedAccess>]
[<CompilationRepresentation (CompilationRepresentationFlags.ModuleSuffix)>]
module DM =

    /// Creates a matrix from 2D array `a`
    let inline ofArray2D a = DM.OfArray2D(a)

    /// Converts matrix `m` into a 2D array
    let inline toArray2D (m:DM) = m.GetRows() |> Seq.map DV.toArray |> array2D

    /// Creates a matrix with `m` rows from array `a`
    let inline ofArray m a = DM.OfArray(m, a)

    /// Converts matrix `m` into an array by stacking its rows
    let inline toArray (m:DM) = DM.ReshapeToDV(m) |> DV.toArray

    /// Transpose of matrix `m`
    let inline transpose (m:DM) = DM.Transpose(m)

    /// Creates a matrix from a sequence of row vectors `s`
    let inline ofRows s = DM.OfRows(s)

    /// Creates a matrix from a sequence of column vectors `s`
    let inline ofCols (s:seq<DV>) = s |> ofRows |> transpose

    /// Gets the sequence of row vectors in matrix `m`
    let inline toRows (m:DM) = m.GetRows()

    /// Gets the sequence of column vectors in matrix `m`
    let inline toCols (m:DM) = m.GetCols()

    /// Converts matrix `m` into a vector by stacking its rows
    let inline toDV (m:DM) = DM.ReshapeToDV(m)

    /// Creates a matrix with `m` rows from vector `v`
    let inline ofDV (m:int) (v:DV) = DV.ReshapeToDM(m, v)

    /// Gets the column with index `j` of matrix `m`
    let inline col (j:int) (m:DM) = m.[*, j]

    /// Gets the row with index `i` of matrix `m`
    let inline row (i:int) (m:DM) = m.[i, *]

    /// Number of columns in matrix `m`
    let inline cols (m:DM) = m.Cols

    /// Number of rows in matrix `m`
    let inline rows (m:DM) = m.Rows

    /// Creates a matrix with `m` rows and `n` columns, where all entries have value `v`
    let inline create m n (v:'a) =
        let at = typeof<'a>
        if at.Equals(typeof<D>) then DM.OfArray2D(Array2D.create m n (unbox<D>(box v)))
        elif at.Equals(typeof<number>) then DM (Array2D.create m n (unbox<number>(box v)))
        elif at.Equals(typeof<int>) then DM (Array2D.create m n (unbox<int>(box v) |> N.toNumber))
        else N.failWithInvalidTypeMessage ()

    /// Creates a matrix with `m` rows, where all rows are equal to `v`
    let inline createRows (m:int) (v:DV) = DM.OfRows(m, v)

    /// Creates a matrix with `n` columns, where all columns are equal to `v`
    let inline createCols (n:int) (v:DV) = DM.OfCols(n, v)

    /// Creates a matrix with `m` rows and `n` columns, where all entries are zero
    let inline zeroCreate m n = DM.ZeroMN m n

    /// Gets the diagonal of matrix `m`
    let inline diagonal (m:DM) = DM.Diagonal(m)

    /// Zero matrix
    let empty = DM.Zero

    /// Returns true if matrix `m` is empty, otherwise returns false
    let isEmpty (m:DM) = m.Length = 0

    /// Creates a matrix with `m` rows and `n` columns, where each element is given by function `f`
    let inline init m n (f:int->int->'a) =
        let at = typeof<'a>
        if at.Equals(typeof<D>) then DM.OfArray2D(Array2D.init m n (unbox<int->int->D>(box f)))
        elif at.Equals(typeof<number>) then DM (Array2D.init m n (unbox<int->int->number>(box f)))
        elif at.Equals(typeof<int>) then DM ((Array2D.init m n (unbox<int->int->int>(box f))) |> Array2D.map N.toNumber)
        else N.failWithInvalidTypeMessage ()

    /// Creates a matrix with `m` rows, where each row is given by `f` as a vector
    let inline initRows (m:int) (f:int->DV) = Seq.init m f |> ofRows

    /// Creates a matrix with `n` columns, where each column is given by `f` as a vector
    let inline initCols (n:int) (f:int->DV) = Seq.init n f |> ofCols

    /// Inverse of matrix `m`
    let inline inverse (m:DM) = DM.Inverse(m)

    /// Iterates function `f` over the entries of matrix `m`
    let inline iter (f:D->unit) (m:DM) = m |> toDV |> DV.iter f

    /// Iterates function `f` over the entries of matrices `m1` and `m2`
    let inline iter2 (f:D->D->unit) (m1:DM) (m2:DM) = DV.iter2 f (m1 |> toDV) (m2 |> toDV)

    /// Iterates function `f` over the entries of matrix `m`. Indices are also supplied to `f`.
    let inline iteri (f:int->int->D->unit) (m:DM) = m |> toArray2D |> Array2D.iteri f

    /// Iterates function `f` over the columns of matrix `m`
    let inline iterCols (f:DV->unit) (m:DM) = m |> toCols |> Seq.iter f

    /// Iterates function `f` over the rows of matrix `m`
    let inline iterRows (f:DV->unit) (m:DM) = m |> toRows |> Seq.iter f

    /// Iterates function `f` over the columns of matrix `m`. Column indices are also supplied to `f`.
    let inline iteriCols (f:int->DV->unit) (m:DM) = m |> toCols |> Seq.iteri f

    /// Iterates function `f` over the rows of matrix `m`. Row indices are also supplied to `f`.
    let inline iteriRows (f:int->DV->unit) (m:DM) = m |> toRows |> Seq.iteri f

    /// Iterates function `f` over the columns of matrices `m1` and `m2`
    let inline iter2Cols (f:DV->DV->unit) (m1:DM) (m2:DM) = Seq.iter2 f (m1 |> toCols) (m2 |> toCols)

    /// Iterates function `f` over the rows of matrices `m1` and `m2
    let inline iter2Rows (f:DV->DV->unit) (m1:DM) (m2:DM) = Seq.iter2 f (m1 |> toRows) (m2 |> toRows)

    /// Iterates function `f` over the columns of matrices `m1` and `m2`. Column indices are also supplied to `f`.
    let inline iteri2Cols (f:int->DV->DV->unit) (m1:DM) (m2:DM) = Seq.iteri2 f (m1 |> toCols) (m2 |> toCols)

    /// Iterates function `f` over the rows of matrices `m1` and `m2`. Row indices are also supplied to `f`.
    let inline iteri2Rows (f:int->DV->DV->unit) (m1:DM) (m2:DM) = Seq.iteri2 f (m1 |> toRows) (m2 |> toRows)

    /// Total number of elements in matrix `m`
    let inline length (m:DM) = m.Length

    /// Number of rows in matrix `m`. Same with DM.rows.
    let inline length1 (m:DM) = m.Rows

    /// Number of columns in matrix `m`. Same with DM.cols.
    let inline length2 (m:DM) = m.Cols

    /// Determinant of matrix `m`
    let inline det (m:DM) = DM.Det(m)

    /// Maps function `f` to the columns of matrix `m`
    let inline mapCols (f:DV->DV) (m:DM) = m |> toCols |> Seq.map f |> ofCols

    /// Maps function `f` to the rows of matrix `m`
    let inline mapRows (f:DV->DV) (m:DM) = m |> toRows |> Seq.map f |> ofRows

    /// Maps function `f` to the columns of matrix `m`. Column indices are also supplied to `f`.
    let inline mapiCols (f:int->DV->DV) (m:DM) = m |> toCols |> Seq.mapi f |> ofCols

    /// Maps function `f` to the rows of matrix `m`. Row indices are also supplied to `f`.
    let inline mapiRows (f:int->DV->DV) (m:DM) = m |> toRows |> Seq.mapi f |> ofRows

    /// Maps function `f` to the columns of matrices `m1` and `m2`
    let inline map2Cols (f:DV->DV->DV) (m1:DM) (m2:DM) = Seq.map2 f (m1 |> toCols) (m2 |> toCols) |> ofCols

    /// Maps function `f` to the rows of matrices `m1` and `m2`
    let inline map2Rows (f:DV->DV->DV) (m1:DM) (m2:DM) = Seq.map2 f (m1 |> toRows) (m2 |> toRows) |> ofRows

    /// Maps function `f` to the columns of matrices `m1` and `m2`. Column indices are also supplied to `f`.
    let inline mapi2Cols (f:int->DV->DV->DV) (m1:DM) (m2:DM) = Seq.mapi2 f (m1 |> toCols) (m2 |> toCols) |> ofCols

    /// Maps function `f` to the rows of matrices `m1` and `m2`. Row indices are also supplied to `f`.
    let inline mapi2Rows (f:int->DV->DV->DV) (m1:DM) (m2:DM) = Seq.mapi2 f (m1 |> toRows) (m2 |> toRows) |> ofRows

    /// Maximum of the entries of matrix `m`
    let inline max (m:DM) = DM.Max(m)

    /// Index of the maximum entry of matrix `m`
    let inline maxIndex (m:DM) = DM.MaxIndex(m)

    /// Minimum of the entries of matrix `m`
    let inline min (m:DM) = DM.Min(m)

    /// Index of the minimum entry of matrix `m`
    let inline minIndex (m:DM) = DM.MinIndex(m)

    /// Mean of matrix `m`
    let inline mean (m:DM) = DM.Mean(m)

    /// Average of matrix `m`. Same with mean.
    let average = mean

    /// Standard deviation of matrix `m`
    let inline standardDev (m:DM) = DM.StandardDev(m)

    /// Variance of matrix `m`
    let inline variance (m:DM) = DM.Variance(m)

    /// Shift and scale the elements of matrix `m` to have zero mean and unit variance
    let inline standardize (m:DM) = DM.Standardize(m)

    /// Shift and scale the elements of matrix `m` to be in the range [0, 1]
    let inline normalize (m:DM) = DM.Normalize(m)

    /// Solve a system of linear equations Ax = b, where the coefficient matrix `m` has general form
    let inline solve (m:DM) (v:DV) = DM.Solve(m, v)

    /// Solve a system of linear equations Ax = b, where the coefficient matrix `m` is symmetric
    let inline solveSymmetric (m:DM) (v:DV) = DM.SolveSymmetric(m, v)

    /// Sums the elements of matrix `m`
    let inline sum (m:DM) = DM.Sum(m)

    /// Trace of matrix `m`
    let inline trace (m:DM) = DM.Trace(m)

    /// Append row `v` to matrix `m`
    let inline appendRow (v:DV) (m:DM) = let rows = m |> toRows in Seq.append rows (seq [v]) |> ofRows

    /// Prepend row `v` to matrix `m`
    let inline prependRow (v:DV) (m:DM) = let rows = m |> toRows in Seq.append (seq [v]) rows |> ofRows

    /// Append column `v` to matrix `m`
    let inline appendCol (v:DV) (m:DM) = let cols = m |> toCols in Seq.append cols (seq [v]) |> ofCols

    /// Prepend column `v` to matrix `m`
    let inline prependCol (v:DV) (m:DM) = let cols = m |> toCols in Seq.append (seq [v]) cols |> ofCols

    /// Experimental
    let inline toString (m:DM) = m.ToString()
    let inline visualize (m:DM) = m.Visualize()
    let inline visualizeAsDV (m:DM) = DM.ReshapeToDV(m).Visualize()


// Scripts for adjusting the adjoint
type Delta =
    | X of D
    //| XNeg of DeltaV
    interface delta
and DeltaV =
    | XV of DV
    //| XNegV of DeltaV
    interface delta
and DeltaM =
    | XM of DM
    | XNegM of DeltaM
    interface delta
and delta = interface end

/// Represents the computed adjoints for reverse AD.  This is a table indexed by node ID.
/// The table is destructively updated as the adjoints are accumulated.
type Adjoints() =
    let dict = Dictionary<int, dobj>()

    let rec eval d =
        match d with
        | X d -> d
    and evalV d =
        match d with
        | XV d -> d
    and evalM d =
        match d with
        | XM v -> v
        | XNegM v -> -(evalM v)

    member internal __.GetD(uniq: int) = dict.[uniq] :?> D
    member internal __.SetD(uniq:int, v:D) = dict.[uniq] <- v
    member internal __.GetDV(uniq: int) = dict.[uniq] :?> DV
    member internal __.SetDV(uniq:int, v:DV) = dict.[uniq] <- v
    member internal __.GetDM(uniq: int) = dict.[uniq] :?> DM
    member internal __.SetDM(uniq:int, v:DM) = dict.[uniq] <- v

    // adj <- adj + interp(delta)
    member internal __.ApplyDelta(uniq: int, x:Delta) =
        let adj = dict.[uniq] :?> D
        let res = eval x + adj
        dict.[uniq] <- res
        res

    // adj <- adj + interp(delta)
    member internal __.ApplyDeltaV(uniq: int, x:DeltaV) =
        let adj = dict.[uniq] :?> DV
        match adj,x with
        | DV adjv, XV (DV xv) ->
            Backend.Add_V_V_Inplace(xv, adjv)
            adj
        | _ ->
            let res = DV.Add_V_V_Inplace(evalV x,adj)
            dict.[uniq] <- res
            res

    // adj <- adj + interp(delta)
    member internal __.ApplyDeltaM(uniq: int, x:DeltaM) =
        let adj = dict.[uniq] :?> DM
        match adj,x with
        | DM adjm, XM (DM xm) ->
            Backend.AlphaAdd_M_M_Inplace(N.one, xm, adjm)
            adj
        | DM adjm, XNegM (XM (DM xm)) ->
            // TODO: also perform the inplace update in the case where adj is not "DM adj"
            // However this needs care.
            Backend.AlphaAdd_M_M_Inplace(N.minus1, xm, adjm)
            adj
        | _ ->
            let adj = DM.Add_M_M_Inplace(evalM x,adj)
            dict.[uniq] <- adj
            adj

    /// Lookup the adjoint for a value
    member this.Item
      with get (d:D) : D =
        match d with
        | D _ -> D.Zero
        | DF _ -> failwith "Cannot get adjoint value of DF. Use makeReverse on this node when composing the computation."
        | DR (_, _, _, uniq) -> this.GetD(uniq)
      and set (d:D) (v : D) =
        match d with
        | D _ -> ()
        | DF _ -> failwith "Cannot set adjoint value of DF. Use makeReverse on this node when composing the computation."
        | DR (_, _, _, uniq) -> this.SetD(uniq, v)

    /// Lookup the adjoint for a vector
    member this.Item
      with get (d:DV) : DV =
        match d with
        | DV _ -> DV.ZeroN d.Length
        | DVF _ -> failwith "Cannot get adjoint value of DVF. Use makeReverse on this node when composing the computation."
        | DVR (_, _, _, uniq) -> this.GetDV(uniq)
      and set (d:DV) (v : DV) =
        match d with
        | DV _ -> ()
        | DVF _ -> failwith "Cannot set adjoint value of DVF. Use makeReverse on this node when composing the computation."
        | DVR (_, _, _, uniq) -> this.SetDV(uniq, v)

    /// Lookup the adjoint for a matrix
    member this.Item
      with get (d:DM) : DM =
        match d with
        | DM(_) -> DM.ZeroMN d.Rows d.Cols
        | DMF _ -> failwith "Cannot get adjoint value of DMF. Use makeReverse on this node when composing the computation."
        | DMR (_, _, _, uniq) -> this.GetDM(uniq)
      and set (d:DM) (v : DM) =
        match d with
        | DM _ -> ()
        | DMF _ -> failwith "Cannot set adjoint value of DMF. Use makeReverse on this node when composing the computation."
        | DMR (_, _, _, uniq) -> this.SetDM(uniq, v)

    override __.ToString() = sprintf "(%d computed adjoints)" dict.Count


/// D, DV, DM operations (automatically opened)
[<AutoOpen>]
module DOps =

    /// Explicit conversion between types where it is permitted. For example: DV -> number[], number[, ] -> DM
    let inline convert (v:^a) : ^b = ((^a or ^b) : (static member op_Explicit: ^a -> ^b) v)

    /// Create a vector from sequence `v`
    let inline toDV (v:seq<_>) =
        match v with
        | :? seq<D> as v ->
            v |> Seq.toArray |> DV.ofArray
        | _ ->
            v |> Seq.toArray |> Array.map N.toNumber |> DV

    /// Create a matrix form sequence of sequences `m`
    let inline toDM (m:seq<seq<_>>) =
        match m with
        | :? seq<seq<D>> as m ->
            m |> array2D |> DM.ofArray2D
        | _ ->
            m |> array2D |> Array2D.map N.toNumber |> DM

    /// Make forward AD type, with tag `i`, primal `p` and tangent `t`
    let inline makeForward i (t:^a) (p:^a) =
        (^a : (member GetForward : ^a -> uint32 -> ^a) p, t, i)

    /// Make reverse AD type, with tag `i` and primal `p`
    let inline makeReverse i (p:^a) =
        (^a : (member GetReverse : uint32 -> ^a) p, i)

    /// Get the primal value of `d`
    let inline primal (d:^a when ^a : (member P : ^a)) = (^a : (member P : ^a) d)

    /// Get the deepest primal value of `d`
    let inline primalDeep (d:^a when ^a : (member PD: ^a)) = (^a :(member PD :^a) d)

    /// Get the tangent value of `d`
    let inline tangent (d:^a when ^a : (member T : ^a)) = (^a : (member T : ^a) d)

    /// Get the adjoint value of `d`
    let adjoint (adjoints: Adjoints) (d : 'T :> dobj) : 'T =
         match box d with
         | :? D as d -> adjoints.[d] |> box :?> 'T
         | :? DV as d -> adjoints.[d] |> box :?> 'T
         | :? DM as d -> adjoints.[d] |> box :?> 'T
         | _ -> failwith "invalid dobj type"

    /// Get the primal and tangent values of `d`, as a tuple
    let inline primalTangent d = d |> primal, d |> tangent


    type Fanouts = Dictionary<int32,uint32>
    let incrementFanout (fanouts: Fanouts) d =
        match fanouts.TryGetValue(d) with
        | true,fanout ->
            fanouts.[d] <- fanout + 1u
            fanout + 1u
        | _ ->
            fanouts.[d] <- 1u
            1u

    /// Resets the adjoints of all the values in the evaluation trace of `d`, preparing for a new reverse propagation
    let reverseReset (adjoints: Adjoints) (d:dobj) =
        let bxd (x : dobj) = x
        let fanouts = Fanouts()
        // Note, this uses an explicit worklist over (D|DV|DM) to make it tail-recursive
        let rec resetRec (ds:dobj list) =
            match ds with
            | [] -> ()
            | d :: t ->
                match d with
                | :? D as d ->
                    match d with
                    | DR(_, o, _, uniq) ->
                        let fanout = incrementFanout fanouts uniq
                        if fanout = 1u then
                            adjoints.SetD(uniq,D.Zero)
                            match o with
                            | Add_D_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Add_D_DCons(a) -> resetRec (bxd a :: t)
                            | Sub_D_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_D_DCons(a) -> resetRec (bxd a :: t)
                            | Sub_DCons_D(b) -> resetRec (bxd b :: t)
                            | Mul_D_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_D_DCons(a, _) -> resetRec (bxd a :: t)
                            | Div_D_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_D_DCons(a, _) -> resetRec (bxd a :: t)
                            | Div_DCons_D(_, b) -> resetRec (bxd b :: t)
                            | Pow_D_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_D_DCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DCons_D(_, b) -> resetRec (bxd b :: t)
                            | Atan2_D_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_D_DCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DCons_D(_, b) -> resetRec (bxd b :: t)
                            | Log_D(a) -> resetRec (bxd a :: t)
                            | Log10_D(a) -> resetRec (bxd a :: t)
                            | Exp_D(a) -> resetRec (bxd a :: t)
                            | Sin_D(a) -> resetRec (bxd a :: t)
                            | Cos_D(a) -> resetRec (bxd a :: t)
                            | Tan_D(a) -> resetRec (bxd a :: t)
                            | Neg_D(a) -> resetRec (bxd a :: t)
                            | Sqrt_D(a) -> resetRec (bxd a :: t)
                            | Sinh_D(a) -> resetRec (bxd a :: t)
                            | Cosh_D(a) -> resetRec (bxd a :: t)
                            | Tanh_D(a) -> resetRec (bxd a :: t)
                            | Asin_D(a) -> resetRec (bxd a :: t)
                            | Acos_D(a) -> resetRec (bxd a :: t)
                            | Atan_D(a) -> resetRec (bxd a :: t)
                            | Abs_D(a) -> resetRec (bxd a :: t)
                            | Sign_D(a) -> resetRec (bxd a :: t)
                            | Floor_D(a) -> resetRec (bxd a :: t)
                            | Ceil_D(a) -> resetRec (bxd a :: t)
                            | Round_D(a) -> resetRec (bxd a :: t)
                            | Mul_Dot_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_Dot_DV_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Sum_DV(a) -> resetRec (bxd a :: t)
                            | L1Norm_DV(a) -> resetRec (bxd a :: t)
                            | L2NormSq_DV(a) -> resetRec (bxd a :: t)
                            | L2Norm_DV(a) -> resetRec (bxd a :: t)
                            | Item_DV(a, _) -> resetRec (bxd a :: t)
                            | Sum_DM(a) -> resetRec (bxd a :: t)
                            | Item_DM(a, _, _) -> resetRec (bxd a :: t)
                            | Det_DM(a) -> resetRec (bxd a :: t)
                            | ReLU_D(a) -> resetRec (bxd a :: t)
                            | Sigmoid_D(a) -> resetRec (bxd a :: t)
                            | LogSumExp_DV(a) -> resetRec (bxd a :: t)
                            | FixedPoint_D(b, _, _, _) -> resetRec (bxd b :: t)
                            | _ -> resetRec t
                        else resetRec t
                    | _ -> resetRec t
                | :? DV as d ->
                    match d with
                    | DVR(_, o, _, uniq) ->
                        let fanout = incrementFanout fanouts uniq
                        if fanout = 1u then
                            adjoints.SetDV(uniq,DV.ZeroN d.Length)
                            match o with
                            | Add_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Add_DV_DVCons(a) -> resetRec (bxd a :: t)
                            | Add_DV_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Add_DV_DCons(a) -> resetRec (bxd a :: t)
                            | Add_DVCons_D(b) -> resetRec (bxd b :: t)
                            | Sub_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_DV_DVCons(a) -> resetRec (bxd a :: t)
                            | Sub_DVCons_DV(a) -> resetRec (bxd a :: t)
                            | Sub_DV_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_DV_DCons(a) -> resetRec (bxd a :: t)
                            | Sub_DVCons_D(b) -> resetRec (bxd b :: t)
                            | Sub_D_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_D_DVCons(a) -> resetRec (bxd a :: t)
                            | Sub_DCons_DV(b) -> resetRec (bxd b :: t)
                            | Mul_Had_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_Had_DV_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DV_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_DV_DCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DVCons_D(_, b) -> resetRec (bxd b :: t)
                            | Mul_DM_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_DM_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DMCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Mul_DV_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_DV_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DVCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Div_Had_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_Had_DV_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Div_Had_DVCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Div_DV_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_DV_DCons(a, _) -> resetRec (bxd a :: t)
                            | Div_DVCons_D(_, b) -> resetRec (bxd b :: t)
                            | Div_D_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_D_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Div_DCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Pow_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_DV_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DVCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Atan2_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_DV_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DVCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Pow_DV_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_DV_DCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DVCons_D(_, b) -> resetRec (bxd b :: t)
                            | Pow_D_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_D_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Atan2_DV_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_DV_DCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DVCons_D(_, b) -> resetRec (bxd b :: t)
                            | Atan2_D_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_D_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Log_DV(a) -> resetRec (bxd a :: t)
                            | Log10_DV(a) -> resetRec (bxd a :: t)
                            | Exp_DV(a) -> resetRec (bxd a :: t)
                            | Sin_DV(a) -> resetRec (bxd a :: t)
                            | Cos_DV(a) -> resetRec (bxd a :: t)
                            | Tan_DV(a) -> resetRec (bxd a :: t)
                            | Neg_DV(a) -> resetRec (bxd a :: t)
                            | Sqrt_DV(a) -> resetRec (bxd a :: t)
                            | Sinh_DV(a) -> resetRec (bxd a :: t)
                            | Cosh_DV(a) -> resetRec (bxd a :: t)
                            | Tanh_DV(a) -> resetRec (bxd a :: t)
                            | Asin_DV(a) -> resetRec (bxd a :: t)
                            | Acos_DV(a) -> resetRec (bxd a :: t)
                            | Atan_DV(a) -> resetRec (bxd a :: t)
                            | Abs_DV(a) -> resetRec (bxd a :: t)
                            | Sign_DV(a) -> resetRec (bxd a :: t)
                            | Floor_DV(a) -> resetRec (bxd a :: t)
                            | Ceil_DV(a) -> resetRec (bxd a :: t)
                            | Round_DV(a) -> resetRec (bxd a :: t)
                            | Make_DV_ofDs(a) -> resetRec (List.append (a |> Array.map bxd |> List.ofArray) t)
                            | SliceRow_DM(a, _, _) -> resetRec (bxd a :: t)
                            | SliceCol_DM(a, _, _) -> resetRec (bxd a :: t)
                            | Solve_DM_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Solve_DM_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Solve_DMCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Append_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Append_DV_DVCons(a) -> resetRec (bxd a :: t)
                            | Append_DVCons_DV(b) -> resetRec (bxd b :: t)
                            | Split_DV(a, _) -> resetRec (bxd a :: t)
                            | AddItem_DV_D(a, _, b) -> resetRec (bxd a :: bxd b :: t)
                            | AddItem_DV_DCons(a) -> resetRec (bxd a :: t)
                            | AddItem_DVCons_D(_, b) -> resetRec (bxd b :: t)
                            | AddSubVector_DV_DV(a, _, b) -> resetRec (bxd a :: bxd b :: t)
                            | AddSubVector_DV_DVCons(a) -> resetRec (bxd a :: t)
                            | AddSubVector_DVCons_DV(_, b) -> resetRec (bxd b :: t)
                            | ReshapeCopy_DM_DV(a) -> resetRec (bxd a :: t)
                            | Slice_DV(a, _) -> resetRec (bxd a :: t)
                            | Diagonal_DM(a) -> resetRec (bxd a :: t)
                            | ReLU_DV(a) -> resetRec (bxd a :: t)
                            | Sigmoid_DV(a) -> resetRec (bxd a :: t)
                            | _ -> resetRec t
                        else resetRec t
                    | _ -> resetRec t
                | :? DM as d ->
                    match d with
                    | DMR(_, o, _, uniq) ->
                        let fanout = incrementFanout fanouts uniq
                        if fanout = 1u then
                            adjoints.SetDM(uniq,DM.ZeroMN d.Rows d.Cols)
                            match o with
                            | Add_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Add_DM_DMCons(a) -> resetRec (bxd a :: t)
                            | Sub_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_DM_DMCons(a) -> resetRec (bxd a :: t)
                            | Sub_DMCons_DM(a) -> resetRec (bxd a :: t)
                            | Mul_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_DM_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DMCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Mul_Had_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_Had_DM_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DM_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_DM_DCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_DMCons_D(_, b) -> resetRec (bxd b :: t)
                            | Mul_Out_DV_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Mul_Out_DV_DVCons(a, _) -> resetRec (bxd a :: t)
                            | Mul_Out_DVCons_DV(_, b) -> resetRec (bxd b :: t)
                            | Div_Had_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_Had_DM_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Div_Had_DMCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Pow_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_DM_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DMCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Atan2_DM_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_DM_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DMCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Div_DM_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_DM_DCons(a, _) -> resetRec (bxd a :: t)
                            | Div_DMCons_D(_, b) -> resetRec (bxd b :: t)
                            | Div_D_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Div_D_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Div_DCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Add_DM_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Add_DM_DCons(a) -> resetRec (bxd a :: t)
                            | Add_DMCons_D(b) -> resetRec (bxd b :: t)
                            | Add_DMCols_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Add_DMCols_DVCons(a) -> resetRec (bxd a :: t)
                            | Add_DMColsCons_DV(b) -> resetRec (bxd b :: t)
                            | Sub_DM_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_DM_DCons(a) -> resetRec (bxd a :: t)
                            | Sub_DMCons_D(b) -> resetRec (bxd b :: t)
                            | Sub_D_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Sub_D_DMCons(a) -> resetRec (bxd a :: t)
                            | Sub_DCons_DM(b) -> resetRec (bxd b :: t)
                            | Pow_DM_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_DM_DCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DMCons_D(_, b) -> resetRec (bxd b :: t)
                            | Pow_D_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Pow_D_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Pow_DCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Atan2_DM_D(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_DM_DCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DMCons_D(_, b) -> resetRec (bxd b :: t)
                            | Atan2_D_DM(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | Atan2_D_DMCons(a, _) -> resetRec (bxd a :: t)
                            | Atan2_DCons_DM(_, b) -> resetRec (bxd b :: t)
                            | Log_DM(a) -> resetRec (bxd a :: t)
                            | Log10_DM(a) -> resetRec (bxd a :: t)
                            | Exp_DM(a) -> resetRec (bxd a :: t)
                            | Sin_DM(a) -> resetRec (bxd a :: t)
                            | Cos_DM(a) -> resetRec (bxd a :: t)
                            | Tan_DM(a) -> resetRec (bxd a :: t)
                            | Neg_DM(a) -> resetRec (bxd a :: t)
                            | Sqrt_DM(a) -> resetRec (bxd a :: t)
                            | Sinh_DM(a) -> resetRec (bxd a :: t)
                            | Cosh_DM(a) -> resetRec (bxd a :: t)
                            | Tanh_DM(a) -> resetRec (bxd a :: t)
                            | Asin_DM(a) -> resetRec (bxd a :: t)
                            | Acos_DM(a) -> resetRec (bxd a :: t)
                            | Atan_DM(a) -> resetRec (bxd a :: t)
                            | Abs_DM(a) -> resetRec (bxd a :: t)
                            | Sign_DM(a) -> resetRec (bxd a :: t)
                            | Floor_DM(a) -> resetRec (bxd a :: t)
                            | Ceil_DM(a) -> resetRec (bxd a :: t)
                            | Round_DM(a) -> resetRec (bxd a :: t)
                            | Transpose_DM(a) -> resetRec (bxd a :: t)
                            | Make_DM_ofDs(a) -> resetRec (List.append (a |> Array2D.toArray |> Array.map bxd |> List.ofArray) t)
                            | Make_DMRows_ofDV(a) -> resetRec (bxd a :: t)
                            | Make_DMCols_ofDV(a) -> resetRec (bxd a :: t)
                            | Make_DMRows_ofDVs(a) -> resetRec (List.append (a |> Array.map bxd |> List.ofArray) t)
                            | AddItem_DM_D(a, _, _, b) -> resetRec (bxd a :: bxd b :: t)
                            | AddItem_DM_DCons(a) -> resetRec (bxd a :: t)
                            | AddItem_DMCons_D(_, _, b) -> resetRec (bxd b :: t)
                            | AddSubMatrix_DM_DM(a, _, _, b) -> resetRec (bxd a :: bxd b :: t)
                            | AddSubMatrix_DM_DMCons(a) -> resetRec (bxd a :: t)
                            | AddSubMatrix_DMCons_DM(_, _, b) -> resetRec (bxd b :: t)
                            | Slice_DM(a, _, _) -> resetRec (bxd a :: t)
                            | RowMatrix_DV(a) -> resetRec (bxd a :: t)
                            | AddDiagonal_DM_DV(a, b) -> resetRec (bxd a :: bxd b :: t)
                            | AddDiagonal_DM_DVCons(a) -> resetRec (bxd a :: t)
                            | AddDiagonal_DMCons_DV(b) -> resetRec (bxd b :: t)
                            | ReshapeCopy_DV_DM(a) -> resetRec (bxd a :: t)
                            | Inverse_DM(a) -> resetRec (bxd a :: t)
                            | ReLU_DM(a) -> resetRec (bxd a :: t)
                            | Sigmoid_DM(a) -> resetRec (bxd a :: t)
                            | _ -> resetRec t
                        else resetRec t
                    | _ -> resetRec t
                | _ -> resetRec t
        resetRec [d]
        fanouts

    /// Propagates the adjoint `v` backwards through the evaluation trace of `d`. The adjoints in the trace are reset before the push.
    let rec reverseProp (adjoints: Adjoints) (v:dobj) (d:dobj) =
        let fanouts = reverseReset adjoints d
        let inline bxd (x : dobj) = x
        let inline bxdelta (x : delta) = x

        let inline bd (v: Delta) (d:D) = bxdelta v, bxd d
        let inline bdv (v: DeltaV) (d:DV) = bxdelta v, bxd d
        let inline bdm (v: DeltaM) (d:DM) = bxdelta v, bxd d

        let inline bx (v: D) (d:D) = bd (X v) d
        let inline bxv (v: DV) (d:DV) = bdv (XV v) d
        let inline bxm (v: DM) (d:DM) = bdm (XM v) d

        // Note, this uses an explicit worklist over (D*D|DV*DV|DM*DM) to make it tail-recursive
        let rec pushRec (ds:(delta*dobj) list) =
            match ds with
            | [] -> ()
            | (v, d) :: t ->
                match v, d with
                | (:? Delta as delta),  (:? D as d) ->
                    match d with
                    | DR(_, o, _, uniq) ->
                        let dA = adjoints.ApplyDelta(uniq, delta)
                        let fanout = fanouts.[uniq] - 1u
                        fanouts.[uniq] <- fanout
                        // If all incoming parts of the adjoint have been received, then proceed to the parent
                        if fanout = 0u then
                            match o with
                            | Add_D_D(a, b) -> pushRec ((bx dA a) :: (bx dA b) :: t)
                            | Add_D_DCons(a) -> pushRec ((bx dA a) :: t)
                            | Sub_D_D(a, b) -> pushRec ((bx dA a) :: (bx -dA b) :: t)
                            | Sub_D_DCons(a) -> pushRec ((bx dA a) :: t)
                            | Sub_DCons_D(b) -> pushRec ((bx -dA b) :: t)
                            | Mul_D_D(a, b) -> pushRec ((bx (dA * b.P) a) :: (bx (dA * a.P) b) :: t)
                            | Mul_D_DCons(a, cons) -> pushRec ((bx (dA * cons) a) :: t)
                            | Div_D_D(a, b) -> pushRec ((bx (dA / b.P) a) :: (bx (dA * (-a.P / (b.P * b.P))) b) :: t)
                            | Div_D_DCons(a, cons) -> pushRec ((bx (dA / cons) a) :: t)
                            | Div_DCons_D(cons, b) -> pushRec ((bx (dA * (-cons / (b.P * b.P))) b) :: t)
                            | Pow_D_D(a, b) -> pushRec ((bx (dA * (a.P ** (b.P - D.One)) * b.P) a) :: (bx (dA * (a.P ** b.P) * log a.P) b) :: t)
                            | Pow_D_DCons(a, cons) -> pushRec ((bx (dA * (a.P ** (cons - D.One)) * cons) a) :: t)
                            | Pow_DCons_D(cons, b) -> pushRec ((bx (dA * (cons ** b.P) * log cons) b) :: t)
                            | Atan2_D_D(a, b) -> let denom = a.P * a.P + b.P * b.P in pushRec ((bx (dA * b.P / denom) a) :: (bx (dA * (-a.P) / denom) b) :: t)
                            | Atan2_D_DCons(a, cons) -> pushRec ((bx (dA * cons / (a.P * a.P + cons * cons)) a) :: t)
                            | Atan2_DCons_D(cons, b) -> pushRec ((bx (dA * (-cons) / (cons * cons + b.P * b.P)) b) :: t)
                            | Log_D(a) -> pushRec ((bx (dA / a.P) a) :: t)
                            | Log10_D(a) -> pushRec ((bx (dA / (a.P * N.log10Val)) a) :: t)
                            | Exp_D(a) -> pushRec ((bx (dA * d.P) a) :: t) // d.P = exp a.P
                            | Sin_D(a) -> pushRec ((bx (dA * cos a.P) a) :: t)
                            | Cos_D(a) -> pushRec ((bx (dA * (-sin a.P)) a) :: t)
                            | Tan_D(a) -> let seca = D.One / cos a.P in pushRec ((bx (dA * seca * seca) a) :: t)
                            | Neg_D(a) -> pushRec ((bx -dA a) :: t)
                            | Sqrt_D(a) -> pushRec ((bx (dA / (D N.two * d.P)) a) :: t) // d.P = sqrt a.P
                            | Sinh_D(a) -> pushRec ((bx (dA * cosh a.P) a) :: t)
                            | Cosh_D(a) -> pushRec ((bx (dA * sinh a.P) a) :: t)
                            | Tanh_D(a) -> let secha = D.One / cosh a.P in pushRec ((bx (dA * secha * secha) a) :: t)
                            | Asin_D(a) -> pushRec ((bx (dA / sqrt (D.One - a.P * a.P)) a) :: t)
                            | Acos_D(a) -> pushRec ((bx (-dA / sqrt (D.One - a.P * a.P)) a) :: t)
                            | Atan_D(a) -> pushRec ((bx (dA / (D.One + a.P * a.P)) a) :: t)
                            | Abs_D(a) -> pushRec ((bx (dA * D.Sign(a.P)) a) :: t)
                            | Sign_D(a) -> pushRec ((bx D.Zero a) :: t)
                            | Floor_D(a) -> pushRec ((bx D.Zero a) :: t)
                            | Ceil_D(a) -> pushRec ((bx D.Zero a) :: t)
                            | Round_D(a) -> pushRec ((bx D.Zero a) :: t)
                            | Mul_Dot_DV_DV(a, b) -> pushRec ((bxv (dA * b.P) a) :: (bxv (dA * a.P) b) :: t)
                            | Mul_Dot_DV_DVCons(a, cons) -> pushRec ((bxv (dA * cons) a) :: t)
                            | Sum_DV(a) -> pushRec ((bxv (DV.create a.Length dA) a) :: t)
                            | L1Norm_DV(a) -> pushRec ((bxv (dA * DV.Sign a.P) a) :: t)
                            | L2NormSq_DV(a) -> pushRec ((bxv (dA * (D N.two) * a.P) a) :: t)
                            | L2Norm_DV(a) -> pushRec ((bxv ((dA / d.P) * a.P) a) :: t)
                            | Item_DV(a, i) ->
                                adjoints.[a] <- DV.AddItem(adjoints.[a], i, dA);
                                pushRec ((bxv DV.Zero a) :: t)
                            | Sum_DM(a) -> pushRec ((bxm (DM.create a.Rows a.Cols dA) a) :: t)
                            | Item_DM(a, i, j) ->
                                adjoints.[a] <- DM.AddItem(adjoints.[a], i, j, dA);
                                pushRec ((bxm DM.Zero a) :: t)
                            | Det_DM(a) -> pushRec ((bxm (d.T * d.P * DM.Transpose(DM.Inverse(a))) a) :: t) // Check this
                            | ReLU_D(a) -> pushRec ((bx (dA * ((D.Sign(a.P) + N.one) / N.two)) a) :: t)
                            | Sigmoid_D(a) -> pushRec ((bx (dA * d.P * (N.one - d.P)) a) :: t) // d.P = D.Sigmoid(a.P)
                            | LogSumExp_DV(a) -> pushRec ((bxv ((dA / exp d.P) * exp a.P) a) :: t) // d.P = DV.LogSumExp(a.P)
                            | FixedPoint_D(b, bfirst, aprev, alast) ->
                                // Christianson (1994)
                                let imax = DiffSharp.Config.GlobalConfig.FixedPointMaxIterations
                                let eps = D (FixedPointEpsilon)

                                let mutable i = 0

                                let r = dA
                                reverseProp adjoints r alast

                                while i < imax do
                                    i <- i + 1
                                    if i >= imax then
                                        //printfn "Fixed point reverse iteration timeout, i = %i" i
                                        ignore()
                                    else
                                        if abs (adjoints.[aprev] + r - adjoints.[alast]) <= eps then
                                            //printfn "Fixed point reverse iteration converged, i = %i" i
                                            i <- imax
                                        else
                                            reverseProp adjoints (r + adjoints.[aprev]) alast

                                pushRec ((bx (adjoints.[bfirst]) b) :: t) // Propogate converged adjoint back towards the original b at the beginning of the fixed point iteration
                            | _ -> pushRec t
                        else pushRec t
                    | _ -> pushRec t

                | (:? DeltaV as delta), (:? DV as d) ->
                    match d with
                    | DVR(_, o, _, uniq) ->
                        let dA = adjoints.ApplyDeltaV(uniq,delta)
                        let fanout = fanouts.[uniq] - 1u
                        fanouts.[uniq] <- fanout
                        // If all incoming parts of the adjoint have been received, then proceed to the parent
                        if fanout = 0u then
                            match o with
                            | Add_DV_DV(a, b) -> pushRec ((bxv dA a) :: (bxv dA b) :: t)
                            | Add_DV_DVCons(a) -> pushRec ((bxv dA a) :: t)
                            | Add_DV_D(a, b) -> pushRec ((bxv dA a) :: (bx (DV.Sum(dA)) b) :: t)
                            | Add_DV_DCons(a) -> pushRec ((bxv dA a) :: t)
                            | Add_DVCons_D(b) -> pushRec ((bx (DV.Sum(dA)) b) :: t)
                            | Sub_DV_DV(a, b) -> pushRec ((bxv dA a) :: (bxv -dA b) :: t)
                            | Sub_DV_DVCons(a) -> pushRec ((bxv dA a) :: t)
                            | Sub_DVCons_DV(a) -> pushRec ((bxv -dA a) :: t)
                            | Sub_DV_D(a, b) -> pushRec ((bxv dA a) :: (bx -(DV.Sum(dA)) b) :: t)
                            | Sub_DV_DCons(a) -> pushRec ((bxv dA a) :: t)
                            | Sub_DVCons_D(b) -> pushRec ((bx -(DV.Sum(dA)) b) :: t)
                            | Sub_D_DV(a, b) -> pushRec ((bx (DV.Sum(dA)) a) :: (bxv -dA b) :: t)
                            | Sub_D_DVCons(a) -> pushRec ((bx (DV.Sum(dA)) a) :: t)
                            | Sub_DCons_DV(b) -> pushRec ((bxv -dA b) :: t)
                            | Mul_Had_DV_DV(a, b) -> pushRec ((bxv (dA .* b.P) a) :: (bxv (dA .* a.P) b) :: t)
                            | Mul_Had_DV_DVCons(a, cons) -> pushRec ((bxv (dA .* cons) a) :: t)
                            | Mul_DV_D(a, b) -> pushRec ((bxv (dA * b.P) a) :: (bx (dA * a.P) b) :: t)
                            | Mul_DV_DCons(a, cons) -> pushRec ((bxv (dA * cons) a) :: t)
                            | Mul_DVCons_D(cons, b) -> pushRec ((bx (dA * cons) b) :: t)
                            | Mul_DM_DV(a, b) -> pushRec ((bxm (dA &* b.P) a) :: (bxv (DM.Transpose(a.P) * dA) b) :: t)
                            | Mul_DM_DVCons(a, cons) -> pushRec ((bxm (dA &* cons) a) :: t)
                            | Mul_DMCons_DV(cons, b) -> pushRec ((bxv (DM.Transpose(cons) * dA) b) :: t)
                            | Mul_DV_DM(a, b) -> pushRec ((bxv (dA * DM.Transpose(b.P)) a) :: (bxm (a.P &* dA) b) :: t)
                            | Mul_DV_DMCons(a, cons) -> pushRec ((bxv (dA * DM.Transpose(cons)) a) :: t)
                            | Mul_DVCons_DM(cons, b) -> pushRec ((bxm (cons &* dA) b) :: t)
                            | Div_Had_DV_DV(a, b) -> pushRec ((bxv (dA ./ b.P) a) :: (bxv (dA .* (-a.P ./ (b.P .* b.P))) b) :: t)
                            | Div_Had_DV_DVCons(a, cons) -> pushRec ((bxv (dA ./ cons) a) :: t)
                            | Div_Had_DVCons_DV(cons, b) -> pushRec ((bxv (dA .* (-cons ./ (b.P .* b.P))) b) :: t)
                            | Div_DV_D(a, b) -> pushRec ((bxv (dA / b.P) a) :: (bx (dA * (-a.P / (b.P * b.P))) b) :: t)
                            | Div_DV_DCons(a, cons) -> pushRec ((bxv (dA / cons) a) :: t)
                            | Div_DVCons_D(cons, b) -> pushRec ((bx (dA * (-cons / (b.P * b.P))) b) :: t)
                            | Div_D_DV(a, b) -> pushRec ((bx (DV.Sum(dA ./ b.P)) a) :: (bxv (dA .* (-a.P / (b.P .* b.P))) b) :: t)
                            | Div_D_DVCons(a, cons) -> pushRec ((bx (DV.Sum(dA ./ cons)) a) :: t)
                            | Div_DCons_DV(cons, b) -> pushRec ((bxv (dA .* (-cons / (b.P .* b.P))) b) :: t)
                            | Pow_DV_DV(a, b) -> pushRec ((bxv (dA .* (a.P ** (b.P - D.One)) .* b.P) a) :: (bxv (dA .* (a.P ** b.P) .* log a.P) b) :: t)
                            | Pow_DV_DVCons(a, cons) -> pushRec ((bxv (dA .* (a.P ** (cons - D.One)) .* cons) a) :: t)
                            | Pow_DVCons_DV(cons, b) -> pushRec ((bxv (dA .* (cons ** b.P) .* log cons) b) :: t)
                            | Atan2_DV_DV(a, b) -> let denom = (a.P .* a.P) + (b.P .* b.P) in pushRec ((bxv (dA .* b.P ./ denom) a) :: (bxv (dA .* (-a.P) ./ denom) b) :: t)
                            | Atan2_DV_DVCons(a, cons) -> pushRec ((bxv (dA .* cons ./ ((a.P .* a.P) + (cons .* cons))) a) :: t)
                            | Atan2_DVCons_DV(cons, b) -> pushRec ((bxv (dA .* (-cons) ./ ((cons .* cons) + (b.P .* b.P))) b) :: t)
                            | Pow_DV_D(a, b) -> pushRec ((bxv (dA .* (a.P ** (b.P - D.One)) * b.P) a) :: (bx (DV.Sum(dA .* (a.P ** b.P) .* log a.P)) b) :: t)
                            | Pow_DV_DCons(a, cons) -> pushRec ((bxv (dA .* (a.P ** (cons - D.One)) * cons) a) :: t)
                            | Pow_DVCons_D(cons, b) -> pushRec ((bx (DV.Sum(dA .* (cons ** b.P) .* log cons)) b) :: t)
                            | Pow_D_DV(a, b) -> pushRec ((bx (DV.Sum(dA .* (DV.Pow(a.P, b.P - D.One)) .* b.P)) a) :: (bxv (dA .* (DV.Pow(a.P, b.P)) * log a.P) b) :: t)
                            | Pow_D_DVCons(a, cons) -> pushRec ((bx (DV.Sum(dA .* (DV.Pow(a.P, cons - D.One)) .* cons)) a) :: t)
                            | Pow_DCons_DV(cons, b) -> pushRec ((bxv (dA .* (DV.Pow(cons, b.P)) * log cons) b) :: t)
                            | Atan2_DV_D(a, b) -> let denom = (a.P .* a.P) + (b.P * b.P) in pushRec ((bxv (dA * b.P ./ denom) a) :: (bx (DV.Sum(dA .* (-a.P) ./ denom)) b) :: t)
                            | Atan2_DV_DCons(a, cons) -> pushRec ((bxv (dA * cons ./ ((a.P .* a.P) + (cons * cons))) a) :: t)
                            | Atan2_DVCons_D(cons, b) -> pushRec ((bx (DV.Sum(dA .* (-cons) ./ ((cons .* cons) + (b.P * b.P)))) b) :: t)
                            | Atan2_D_DV(a, b) -> let denom = (a.P * a.P) + (b.P .* b.P) in pushRec ((bx (DV.Sum(dA .* b.P ./ denom)) a) :: (bxv (dA * (-a.P) ./ denom) b) :: t)
                            | Atan2_D_DVCons(a, cons) -> pushRec ((bx (DV.Sum(dA .* cons ./ ((a.P * a.P) + (cons .* cons)))) a) :: t)
                            | Atan2_DCons_DV(cons, b) -> pushRec ((bxv (dA * (-cons) ./ ((cons * cons) + (b.P .* b.P))) b) :: t)
                            | Log_DV(a) -> pushRec ((bxv (dA ./ a.P) a) :: t)
                            | Log10_DV(a) -> pushRec ((bxv (dA ./ (a.P * N.log10Val)) a) :: t)
                            | Exp_DV(a) -> pushRec ((bxv (dA .* d.P) a) :: t) // d.P = exp a.P
                            | Sin_DV(a) -> pushRec ((bxv (dA .* cos a.P) a) :: t)
                            | Cos_DV(a) -> pushRec ((bxv (-dA .* sin a.P) a) :: t)
                            | Tan_DV(a) -> let seca = D.One / cos a.P in pushRec ((bxv (dA .* seca .* seca) a) :: t)
                            | Neg_DV(a) -> pushRec ((bxv -dA a) :: t)
                            | Sqrt_DV(a) -> pushRec ((bxv (dA ./ (N.two * d.P)) a) :: t) // d.P = sqrt a.P
                            | Sinh_DV(a) -> pushRec ((bxv (dA .* cosh a.P) a) :: t)
                            | Cosh_DV(a) -> pushRec ((bxv (dA .* sinh a.P) a) :: t)
                            | Tanh_DV(a) -> let secha = D.One / cosh a.P in pushRec ((bxv (dA .* secha .* secha) a) :: t)
                            | Asin_DV(a) -> pushRec ((bxv (dA ./ sqrt (D.One - (a.P .* a.P))) a) :: t)
                            | Acos_DV(a) -> pushRec ((bxv (-dA ./ sqrt (D.One - (a.P .* a.P))) a) :: t)
                            | Atan_DV(a) -> pushRec ((bxv (dA ./ (D.One + (a.P .* a.P))) a) :: t)
                            | Abs_DV(a) -> pushRec ((bxv (dA .* DV.Sign a.P) a) :: t)
                            | Sign_DV(a) -> pushRec ((bxv DV.Zero a) :: t)
                            | Floor_DV(a) -> pushRec ((bxv DV.Zero a) :: t)
                            | Ceil_DV(a) -> pushRec ((bxv DV.Zero a) :: t)
                            | Round_DV(a) -> pushRec ((bxv DV.Zero a) :: t)
                            | Make_DV_ofDs(a) -> pushRec (t |> List.append (a |> Array.mapi (fun i v -> (bx dA.[i] v)) |> List.ofArray))
                            | SliceRow_DM(a, i, j) ->
                                adjoints.[a] <- DM.AddSubMatrix(adjoints.[a], i, j, dA.ToRowDM())
                                pushRec ((bxm DM.Zero a) :: t)
                            | SliceCol_DM(a, i, j) ->
                                adjoints.[a] <- DM.AddSubMatrix(adjoints.[a], i, j, dA.ToColDM())
                                pushRec ((bxm DM.Zero a) :: t)
                            | Solve_DM_DV(a, b) -> let ba = DM.Solve(DM.Transpose(a), dA) in pushRec ((bxm (-ba &* dA) a) :: (bxv (ba) b) :: t)
                            | Solve_DM_DVCons(a, cons) -> let ba = DM.Solve(DM.Transpose(a), dA) in pushRec ((bxm (-ba &* dA) a) :: t)
                            | Solve_DMCons_DV(cons, b) -> let ba = DM.Solve(DM.Transpose(cons), dA) in pushRec ((bxv ba b) :: t)
                            | Append_DV_DV(a, b) ->
                                adjoints.[a] <- adjoints.[a] + dA.[..(a.Length - 1)]
                                adjoints.[b] <- adjoints.[b] + dA.[a.Length..]
                                pushRec ((bxv DV.Zero a) :: (bxv DV.Zero b) :: t)
                            | Append_DV_DVCons(a) ->
                                adjoints.[a] <- adjoints.[a] + dA.[..(a.Length - 1)]
                                pushRec ((bxv DV.Zero a) :: t)
                            | Append_DVCons_DV(b) ->
                                adjoints.[b] <- adjoints.[b] + dA.[(d.Length - b.Length)..]
                                pushRec ((bxv DV.Zero b) :: t)
                            | Split_DV(a, i) ->
                                adjoints.[a] <- DV.AddSubVector(adjoints.[a], i, dA)
                                pushRec ((bxv DV.Zero a) :: t)
                            | AddItem_DV_D(a, i, b) -> pushRec ((bxv dA a) :: (bx (dA.[i]) b) :: t)
                            | AddItem_DV_DCons(a) -> pushRec ((bxv dA a) :: t)
                            | AddItem_DVCons_D(i, b) -> pushRec ((bx dA.[i] b) :: t)
                            | AddSubVector_DV_DV(a, i, b) -> pushRec ((bxv dA a) :: (bxv (dA.[i..(i + b.Length - 1)]) b) :: t)
                            | AddSubVector_DV_DVCons(a) -> pushRec ((bxv dA a) :: t)
                            | AddSubVector_DVCons_DV(i, b) -> pushRec ((bxv (dA.[i..(i + b.Length - 1)]) b) :: t)
                            | ReshapeCopy_DM_DV(a) -> pushRec ((bxm (DV.ReshapeToDM(a.Rows, dA)) a) :: t)
                            | Slice_DV(a, i) ->
                                adjoints.[a] <- DV.AddSubVector(adjoints.[a], i, dA)
                                pushRec ((bxv DV.Zero a) :: t)
                            | Diagonal_DM(a) ->
                                adjoints.[a] <- DM.AddDiagonal(adjoints.[a], dA)
                                pushRec ((bxm DM.Zero a) :: t)
                            | ReLU_DV(a) -> pushRec ((bxv (dA .* ((DV.Sign(a.P) + N.one) / N.two)) a) :: t)
                            | Sigmoid_DV(a) -> pushRec ((bxv (dA .* d.P .* (N.one - d.P)) a) :: t) // d.P = DV.Sigmoid(a.P)
                            | _ -> pushRec t
                        else pushRec t
                    | _ -> pushRec t

                | (:? DeltaM as delta), (:? DM as d) ->
                    match d with
                    | DMR(_, o, _, uniq) ->
                        let dA = adjoints.ApplyDeltaM(uniq,delta)
                        let fanout = fanouts.[uniq] - 1u
                        fanouts.[uniq] <- fanout
                        // If all incoming parts of the adjoint have been received, then proceed to the parent
                        if fanout = 0u then
                            match o with
                            | Add_DM_DM(a, b) -> pushRec ((bxm dA a) :: (bxm dA b) :: t)
                            | Add_DM_DMCons(a) -> pushRec ((bxm dA a) :: t)

                            // When pushing "-dA" as adjoint increment for b, the operation
                            //    "b.Adjoint <- -1.0 * dA + b.Adjoint"
                            // can be performed directly in-place. Instead of pushing a D|DV|DM we should a
                            // structured expression about how to compute the D|DV|DM which can be interpreted
                            // to do an in-place update
                            | Sub_DM_DM(a, b) -> pushRec ((bxm dA a) :: (bdm (XNegM (XM dA)) b) :: t)

                            // TODO: also avoid the inplace operations in most of the below.
                            | Sub_DM_DMCons(a) -> pushRec ((bxm dA a) :: t)
                            | Sub_DMCons_DM(a) -> pushRec ((bxm -dA a) :: t)
                            | Mul_DM_DM(a, b) -> pushRec ((bxm (dA * DM.Transpose(b.P)) a) :: (bxm (DM.Transpose(a.P) * dA) b) :: t)
                            | Mul_DM_DMCons(a, cons) -> pushRec ((bxm (dA * DM.Transpose(cons)) a) :: t)
                            | Mul_DMCons_DM(cons, b) -> pushRec ((bxm (DM.Transpose(cons) * dA) b) :: t)
                            | Mul_Had_DM_DM(a, b) -> pushRec ((bxm (dA .* b.P) a) :: (bxm (dA .* a.P) b) :: t)
                            | Mul_Had_DM_DMCons(a, cons) -> pushRec ((bxm (dA .* cons) a) :: t)
                            | Mul_DM_D(a, b) -> pushRec ((bxm (dA * b.P) a) :: (bx (DM.Sum(dA .* a.P)) b) :: t)
                            | Mul_DM_DCons(a, cons) -> pushRec ((bxm (dA * cons) a) :: t)
                            | Mul_DMCons_D(cons, b) -> pushRec ((bx (DM.Sum(dA .* cons)) b) :: t)
                            | Mul_Out_DV_DV(a, b) -> pushRec ((bxv (dA * b.P) a) :: (bxv (DM.Transpose(dA) * a.P) b) :: t)
                            | Mul_Out_DV_DVCons(a, cons) -> pushRec ((bxv (dA * cons) a) :: t)
                            | Mul_Out_DVCons_DV(cons, b) -> pushRec ((bxv (DM.Transpose(dA) * cons) b) :: t)
                            | Div_Had_DM_DM(a, b) -> pushRec ((bxm (dA ./ b.P) a) :: (bxm (dA .* (-a.P ./ (b.P .* b.P))) b) :: t)
                            | Div_Had_DM_DMCons(a, cons) -> pushRec ((bxm (dA ./ cons) a) :: t)
                            | Div_Had_DMCons_DM(cons, b) -> pushRec ((bxm (dA .* (-cons ./ (b.P .* b.P))) b) :: t)
                            | Pow_DM_DM(a, b) -> pushRec ((bxm (dA .* (a.P ** (b.P - D.One)) .* b.P) a) :: (bxm (dA .* (a.P ** b.P) .* log a.P) b) :: t)
                            | Pow_DM_DMCons(a, cons) -> pushRec ((bxm (dA .* (a.P ** (cons - D.One)) .* cons) a) :: t)
                            | Pow_DMCons_DM(cons, b) -> pushRec ((bxm (dA .* (cons ** b.P) .* log cons) b) :: t)
                            | Atan2_DM_DM(a, b) -> let denom = (a.P .* a.P) + (b.P .* b.P) in pushRec ((bxm (dA .* b.P ./ denom) a) :: (bxm (dA .* (-a.P) ./ denom) b) :: t)
                            | Atan2_DM_DMCons(a, cons) -> pushRec ((bxm (dA .* cons ./ ((a.P .* a.P) + (cons .* cons))) a) :: t)
                            | Atan2_DMCons_DM(cons, b) -> pushRec ((bxm (dA .* (-cons) ./ ((cons .* cons) + (b.P .* b.P))) b) :: t)
                            | Add_DM_D(a, b) -> pushRec ((bxm dA a) :: (bx (DM.Sum(dA)) b) :: t)
                            | Add_DM_DCons(a) -> pushRec ((bxm dA a) :: t)
                            | Add_DMCons_D(b) -> pushRec ((bx (DM.Sum(dA)) b) :: t)
                            | Add_DMCols_DV(a, b) ->
                                dA.GetCols() |> Seq.iter (fun v -> adjoints.[b] <- adjoints.[b] + v)
                                pushRec ((bxm dA a) :: (bxv DV.Zero b) :: t)
                            | Add_DMCols_DVCons(a) ->
                                pushRec ((bxm dA a) :: t)
                            | Add_DMColsCons_DV(b) ->
                                dA.GetCols() |> Seq.iter (fun v -> adjoints.[b] <- adjoints.[b] + v)
                                pushRec ((bxv DV.Zero b) :: t)
                            | Sub_DM_D(a, b) -> pushRec ((bxm dA a) :: (bx -(DM.Sum(dA)) b) :: t)
                            | Sub_DM_DCons(a) -> pushRec ((bxm dA a) :: t)
                            | Sub_DMCons_D(b) -> pushRec ((bx -(DM.Sum(dA)) b) :: t)
                            | Sub_D_DM(a, b) -> pushRec ((bx (DM.Sum(dA)) a) :: (bxm -dA b) :: t)
                            | Sub_D_DMCons(a) -> pushRec ((bx (DM.Sum(dA)) a) :: t)
                            | Sub_DCons_DM(b) -> pushRec ((bxm -dA b) :: t)
                            | Div_DM_D(a, b) -> pushRec ((bxm (dA / b.P) a) :: (bx (DM.Sum (dA .* (-a.P / b.P * b.P))) b) :: t)
                            | Div_DM_DCons(a, cons) -> pushRec ((bxm (dA / cons) a) :: t)
                            | Div_DMCons_D(cons, b) -> pushRec ((bx (DM.Sum (dA .* (-cons / (b.P * b.P)))) b) :: t)
                            | Div_D_DM(a, b) -> pushRec ((bx (DM.Sum(dA ./ b.P)) a) :: (bxm (dA .* (-a.P / (b.P .* b.P))) b) :: t)
                            | Div_D_DMCons(a, cons) -> pushRec ((bx (DM.Sum(dA ./ cons)) a) :: t)
                            | Div_DCons_DM(cons, b) -> pushRec ((bxm (dA .* (-cons / (b.P .* b.P))) b) :: t)
                            | Pow_DM_D(a, b) -> pushRec ((bxm (dA .* (a.P ** (b.P - D.One)) * b.P) a) :: (bx (DM.Sum(dA .* (a.P ** b.P) .* log a.P)) b) :: t)
                            | Pow_DM_DCons(a, cons) -> pushRec ((bxm (dA .* (a.P ** (cons - D.One)) * cons) a) :: t)
                            | Pow_DMCons_D(cons, b) -> pushRec ((bx (DM.Sum(dA .* (cons ** b.P) .* log cons)) b) :: t)
                            | Pow_D_DM(a, b) -> pushRec ((bx (DM.Sum(dA .* (DM.Pow(a.P, b.P - D.One)) .* b.P)) a) :: (bxm (dA .* (DM.Pow(a.P, b.P)) * log a.P) b) :: t)
                            | Pow_D_DMCons(a, cons) -> pushRec ((bx (DM.Sum(dA .* (DM.Pow(a.P, cons - D.One)) .* cons)) a) :: t)
                            | Pow_DCons_DM(cons, b) -> pushRec ((bxm (dA .* (DM.Pow(cons, b.P)) * log cons) b) :: t)
                            | Atan2_DM_D(a, b) -> let denom = (a.P .* a.P) + (b.P * b.P) in pushRec ((bxm (dA * b.P ./ denom) a) :: (bx (DM.Sum(dA .* (-a.P) ./ denom)) b) :: t)
                            | Atan2_DM_DCons(a, cons) -> pushRec ((bxm (dA * cons ./ ((a.P .* a.P) + (cons * cons))) a) :: t)
                            | Atan2_DMCons_D(cons, b) ->pushRec ((bx (DM.Sum(dA .* (-cons) ./ ((cons .* cons) + (b.P * b.P)))) b) :: t)
                            | Atan2_D_DM(a, b) -> let denom = (a.P * a.P) + (b.P .* b.P) in pushRec ((bx (DM.Sum(dA .* b.P ./ denom)) a) :: (bxm (dA * (-a.P) ./ denom) b) :: t)
                            | Atan2_D_DMCons(a, cons) -> pushRec ((bx (DM.Sum(dA .* cons ./ ((a.P * a.P) + (cons .* cons)))) a) :: t)
                            | Atan2_DCons_DM(cons, b) -> pushRec ((bxm (dA * (-cons) ./ ((cons * cons) + (b.P .* b.P))) b) :: t)
                            | Log_DM(a) -> pushRec ((bxm (dA ./ a.P) a) :: t)
                            | Log10_DM(a) -> pushRec ((bxm (dA ./ (a.P * N.log10Val)) a) :: t)
                            | Exp_DM(a) -> pushRec ((bxm (dA .* d.P) a) :: t) // d.P = exp a.P
                            | Sin_DM(a) -> pushRec ((bxm (dA .* cos a.P) a) :: t)
                            | Cos_DM(a) -> pushRec ((bxm (-dA .* sin a.P) a) :: t)
                            | Tan_DM(a) -> let seca = D.One / cos a.P in pushRec ((bxm (dA .* seca .* seca) a) :: t)
                            | Neg_DM(a) -> pushRec ((bxm -dA a) :: t)
                            | Sqrt_DM(a) -> pushRec ((bxm (dA ./ (N.two * d.P)) a) :: t) // d.P = sqrt a.P
                            | Sinh_DM(a) -> pushRec ((bxm (dA .* cosh a.P) a) :: t)
                            | Cosh_DM(a) -> pushRec ((bxm (dA .* sinh a.P) a) :: t)
                            | Tanh_DM(a) -> let secha = D.One / cosh a.P in pushRec ((bxm (dA .* secha .* secha) a) :: t)
                            | Asin_DM(a) -> pushRec ((bxm (dA ./ sqrt (D.One - (a.P .* a.P))) a) :: t)
                            | Acos_DM(a) -> pushRec ((bxm (-dA ./ sqrt (D.One - (a.P .* a.P))) a) :: t)
                            | Atan_DM(a) -> pushRec ((bxm (dA ./ (D.One + (a.P .* a.P))) a) :: t)
                            | Abs_DM(a) -> pushRec ((bxm (dA .* DM.Sign a.P) a) :: t)
                            | Sign_DM(a) -> pushRec ((bxm DM.Zero a) :: t)
                            | Floor_DM(a) -> pushRec ((bxm DM.Zero a) :: t)
                            | Ceil_DM(a) -> pushRec ((bxm DM.Zero a) :: t)
                            | Round_DM(a) -> pushRec ((bxm DM.Zero a) :: t)
                            | Transpose_DM(a) -> pushRec ((bxm (DM.Transpose(dA)) a) :: t)
                            | Make_DM_ofDs(a) -> pushRec (t |> List.append (List.map2 (fun v dd -> (bx v dd)) (dA |> DM.toDV |> DV.toArray |> Array.toList) (a |> Array2D.toArray |> List.ofArray)))
                            | Make_DMRows_ofDV(a) ->
                                dA.GetRows() |> Seq.iter (fun v -> adjoints.[a] <- adjoints.[a] + v)
                                pushRec ((bxv DV.Zero a) :: t)
                            | Make_DMCols_ofDV(a) ->
                                dA.GetCols() |> Seq.iter (fun v -> adjoints.[a] <- adjoints.[a] + v)
                                pushRec ((bxv DV.Zero a) :: t)
                            | Make_DMRows_ofDVs(a) -> pushRec (t |> List.append (a |> List.ofArray |> List.mapi (fun i v -> (bxv dA.[i, *] v))))
                            | AddItem_DM_D(a, i, j, b) -> pushRec ((bxm dA a) :: (bx (dA.[i, j]) b) :: t)
                            | AddItem_DM_DCons(a) -> pushRec ((bxm dA a) :: t)
                            | AddItem_DMCons_D(i, j, b) -> pushRec ((bx dA.[i, j] b) :: t)
                            | AddSubMatrix_DM_DM(a, i, j, b) -> pushRec ((bxm dA a) :: (bxm (dA.[i..(i + b.Rows - 1), j..(j + b.Cols - 1)]) b) :: t)
                            | AddSubMatrix_DM_DMCons(a) -> pushRec ((bxm dA a) :: t)
                            | AddSubMatrix_DMCons_DM(i, j, b) -> pushRec ((bxm (dA.[i..(i + b.Rows - 1), j..(j + b.Cols - 1)]) b) :: t)
                            | Slice_DM(a, i, j) ->
                                adjoints.[a] <- DM.AddSubMatrix(adjoints.[a], i, j, dA)
                                pushRec ((bxm DM.Zero a) :: t)
                            | RowMatrix_DV(a) -> pushRec ((bxv (dA.[0, *]) a) :: t)
                            | AddDiagonal_DM_DV(a, b) -> pushRec ((bxm dA a) :: (bxv (DM.Diagonal(dA)) b) :: t)
                            | AddDiagonal_DM_DVCons(a) -> pushRec ((bxm dA a) :: t)
                            | AddDiagonal_DMCons_DV(b) -> pushRec ((bxv (DM.Diagonal(dA)) b) :: t)
                            | ReshapeCopy_DV_DM(a) -> pushRec ((bxv (DM.ReshapeToDV(dA)) a) :: t)
                            | Inverse_DM(a) -> let dpt = DM.Transpose(d.P) in pushRec ((bxm (-dpt * dA * dpt) a) :: t) // d.P = DM.Inverse(a.P)
                            | ReLU_DM(a) -> pushRec ((bxm (dA .* ((DM.Sign(a.P) + N.one) / N.two)) a) :: t)
                            | Sigmoid_DM(a) -> pushRec ((bxm (dA .* d.P .* (N.one - d.P)) a) :: t) // d.P = DM.Sigmoid(a.P)
                            | _ -> pushRec t
                        else pushRec t
                    | _ -> pushRec t
                | _ -> pushRec t
        let initialv =
            match v with
            | :? D as v -> bxdelta (X v)
            | :? DV as v -> bxdelta (XV v)
            | :? DM as v -> bxdelta (XM v)
            | _ -> failwith "invalid dobj"
        pushRec [(initialv, d)]

/// Forward and reverse differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =

    /// Compute the adjoints of a differentiable value
    let inline computeAdjoints (d: 'T :> dobj) =
        let adjoints = Adjoints()
        let one = LanguagePrimitives.GenericOne<'T>
        reverseProp adjoints one d
        adjoints

    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff' (f: D -> D) x =
        let dx = makeForward GlobalTagger.Next (D.One) x
        dx |> f |> primalTangent

    /// First derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff (f: D -> D) x = diff' f x |> snd

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff2 (f: D -> D) x  : D =
        diff (diff f) x

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff2'' (f: D -> D) x : D * D * D =
        let v, d = diff' f x
        let d2 = diff2 f x
        (v, d, d2)

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff2' (f: D -> D) x  : D * D =
        diff2'' f x |> drop2Of3

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diffn n (f: D -> D) x  : D =
        if n < 0 then ErrorMessages.InvalidArgDiffn()
        elif n = 0 then x |> f
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diffn' n (f: D -> D) x  : D * D =
        (x |> f, diffn n f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let grad' (f: DV -> D) x : D * DV =
        let xa = x |> makeReverse GlobalTagger.Next
        let z:D = f xa
        let adjoints = computeAdjoints z
        (z |> primal, xa |> adjoint adjoints )

    /// Gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let grad (f: DV -> D) x : DV =
        grad' f x |> snd

    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Forward AD.
    let gradv' (f: DV -> D) (x: DV) (v: DV) : D * D =
        let dvx = makeForward GlobalTagger.Next v x
        dvx |> f |> primalTangent

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Forward AD.
    let gradv (f: DV -> D) x v : D =
        gradv' f x v |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Forward AD.
    let jacobianv' (f: DV -> DV) x v : DV * DV =
        x |> makeForward GlobalTagger.Next v |> f |> primalTangent

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Forward AD.
    let jacobianv (f: DV -> DV) x v : DV =
        jacobianv' f x v |> snd

    /// Original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of reverse mode AD, with the given vector, without repeating the forward pass). Reverse AD.
    let jacobianTv'' (f: DV -> DV) (x:DV) =
        let xa = x |> makeReverse GlobalTagger.Next
        let z = f xa
        let r1 = z |> primal
        let r2 =
            fun (v:DV) ->
                let adjoints = Adjoints()
                z |> reverseProp adjoints v
                xa |> adjoint adjoints
        (r1, r2)

    /// Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Reverse AD.
    let jacobianTv' (f: DV -> DV) x v =
        let r1, r2 = jacobianTv'' f x
        (r1, r2 v)

    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Reverse AD.
    let jacobianTv (f: DV -> DV) x v =
        jacobianTv' f x v |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let jacobian' (f: DV -> DV) (x:DV) : DV * DM =
        let o:DV = x |> f |> primal
        if x.Length > o.Length then
            let r = jacobianTv f x
            (o, Array.init o.Length (fun j -> r (DV.standardBasis o.Length j)) |> DM.ofRows)
        else
            (o, Array.init x.Length (fun i -> jacobianv f x (DV.standardBasis x.Length i)) |> DM.ofCols)

    /// Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let jacobian (f: DV -> DV) x : DM =
        jacobian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let jacobianT' (f: DV -> DV) x =
        jacobian' f x |> fun (r, j) -> (r, DM.transpose j)

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let jacobianT (f: DV -> DV) x : DM =
        jacobianT' f x |> snd

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let gradhessian (f: DV -> D) x : DV * DM =
        jacobian' (grad f) x

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let gradhessian' (f: DV -> D) x : D * DV * DM =
        let g, h = gradhessian f x
        (x |> f , g, h)

    /// Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let hessian (f: DV -> D) x : DM =
        jacobian (grad f) x

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let hessian' (f: DV -> D) x : D * DM =
        (x |> f, hessian f x)

    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let gradhessianv' (f: DV -> D) x v =
        let gv, hv = grad' (fun xx -> gradv f xx v) x
        (x |> f, gv, hv)

    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let gradhessianv (f: DV -> D) x v : D * DV =
        gradhessianv' f x v |> drop1Of3

    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let hessianv' (f: DV -> D) x v =
        gradhessianv' f x v |> drop2Of3

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let hessianv (f: DV -> D) x v : DV =
        hessianv' f x v |> snd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let laplacian' (f: DV -> D) x : D * D = // TODO: reimplement faster
        let v, h = hessian' f x
        (v, DM.trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let laplacian (f: DV -> D) x : D =
        laplacian' f x |> snd

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let curl' (f: DV -> DV) x =
        let v, j = jacobianT' f x
        if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, toDV [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let curl (f: DV -> DV) x : DV =
        curl' f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix. Forward AD.
    let div' (f: DV -> DV) x =
        let v, j = jacobianT' f x
        if j.Rows <> j.Cols then ErrorMessages.InvalidArgDiv()
        v, DM.trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix. Forward AD.
    let div (f: DV -> DV) x : D =
        div' f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let curldiv' (f: DV -> DV) x =
        let v, j = jacobianT' f x
        if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, toDV [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], DM.trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let curldiv (f: DV -> DV) x : DV * D =
        curldiv' f x |> drop1Of3
