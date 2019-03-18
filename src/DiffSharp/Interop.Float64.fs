// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

/// Interoperability layer, for C# and other CLR languages
namespace DiffSharp.Interop.Float64

open DiffSharp.Util

module AD = DiffSharp.AD.Float64
module Numerical = DiffSharp.Numerical.Float64
type number = AD.number

type internal ADD = AD.D
type internal ADDV = AD.DV
type internal ADDM = AD.DM

type D(x:ADD) =
    new(x:AD.number) = D(ADD.D(x))
    member internal this.toADD() : ADD = x
    static member internal ADDtoD (x:ADD) = new D(x)
    static member internal DtoADD (x:D) = x.toADD()

    member d.P = d.toADD().P |> D.ADDtoD
    member d.T = d.toADD().T |> D.ADDtoD

    override d.ToString() =
        let rec s (d:ADD) =
            match d with
            | AD.D(p) -> sprintf "D %A" p
            | AD.DF(p, t, _) -> sprintf "DF (%A, %A)" (s p) (s t)
            | AD.DR(p, op, _, _) -> sprintf "DR (%A, %A)"  (s p) (op.ToString())
        s (d.toADD())
    static member op_Explicit(d:D):AD.number = ADD.op_Explicit (d.toADD())
    static member op_Implicit(a:AD.number):D = D(a)
    static member Zero = ADD.Zero |> D.ADDtoD
    static member One = ADD.One  |> D.ADDtoD
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> compare (d.toADD()) (d2.toADD())
            | _ -> invalidArg "" "Cannot compare this D with another type of object."
    override d.Equals(other) =
        match other with
        | :? D as d2 -> compare (d.toADD()) (d2.toADD()) = 0
        | _ -> false
    override d.GetHashCode() = d.toADD().GetHashCode()
    // D - D binary operations
    static member (+) (a:D, b:D) = D(a.toADD() + b.toADD())
    static member (-) (a:D, b:D) = D(a.toADD() - b.toADD())
    static member (*) (a:D, b:D) = D(a.toADD() * b.toADD())
    static member (/) (a:D, b:D) = D(a.toADD() / b.toADD())
    static member Pow (a:D, b:D) = D(a.toADD() ** b.toADD())
    static member Atan2 (a:D, b:D) = D(atan2 (a.toADD()) (b.toADD()))
    // D -  binary operations
    static member (+) (a:D, b:AD.number) = a + (D b)
    static member (-) (a:D, b:AD.number) = a - (D b)
    static member (*) (a:D, b:AD.number) = a * (D b)
    static member (/) (a:D, b:AD.number) = a / (D b)
    static member Pow (a:D, b:AD.number) = a ** (D b)
    static member Atan2 (a:D, b:AD.number) = atan2 a (D b)
    // AD.number - D binary operations
    static member (+) (a:AD.number, b:D) = (D a) + b
    static member (-) (a:AD.number, b:D) = (D a) - b
    static member (*) (a:AD.number, b:D) = (D a) * b
    static member (/) (a:AD.number, b:D) = (D a) / b
    static member Pow (a:AD.number, b:D) = (D a) ** b
    static member Atan2 (a:AD.number, b:D) = atan2 (D a) b
    // D - int binary operations
    static member (+) (a:D, b:int) = a + (D (AD.N.toNumber b))
    static member (-) (a:D, b:int) = a - (D (AD.N.toNumber b))
    static member (*) (a:D, b:int) = a * (D (AD.N.toNumber b))
    static member (/) (a:D, b:int) = a / (D (AD.N.toNumber b))
    static member Pow (a:D, b:int) = D.Pow(a, (D (AD.N.toNumber b)))
    static member Atan2 (a:D, b:int) = D.Atan2(a, (D (AD.N.toNumber b)))
    // int - D binary operations
    static member (+) (a:int, b:D) = (D (AD.N.toNumber a)) + b
    static member (-) (a:int, b:D) = (D (AD.N.toNumber a)) - b
    static member (*) (a:int, b:D) = (D (AD.N.toNumber a)) * b
    static member (/) (a:int, b:D) = (D (AD.N.toNumber a)) / b
    static member Pow (a:int, b:D) = D.Pow((D (AD.N.toNumber a)), b)
    static member Atan2 (a:int, b:D) = D.Atan2((D (AD.N.toNumber a)), b)
    // D unary operations
    static member Log (a:D) = D(log (a.toADD()))
    static member Log10 (a:D) = D(log10 (a.toADD()))
    static member Exp (a:D) = D(exp (a.toADD()))
    static member Sin (a:D) = D(sin (a.toADD()))
    static member Cos (a:D) = D(cos (a.toADD()))
    static member Tan (a:D) = D(tan (a.toADD()))
    static member Neg (a:D) = D(-(a.toADD()))
    static member Sqrt (a:D) = D(sqrt (a.toADD()))
    static member Sinh (a:D) = D(sinh (a.toADD()))
    static member Cosh (a:D) = D(cosh (a.toADD()))
    static member Tanh (a:D) = D(tanh (a.toADD()))
    static member Asin (a:D) = D(asin (a.toADD()))
    static member Acos (a:D) = D(acos (a.toADD()))
    static member Atan (a:D) = D(atan (a.toADD()))
    static member Abs (a:D) = D(abs (a.toADD()))
    static member Floor (a:D) = D(floor (a.toADD()))
    static member Ceiling (a:D) = D(ceil (a.toADD()))
    static member Round (a:D) = D(round (a.toADD()))
    static member Sign (a:D) = D(ADD.Sign(a.toADD()))
    static member SoftPlus (a:D) = D(ADD.SoftPlus(a.toADD()))
    static member SoftSign (a:D) = D(ADD.SoftSign(a.toADD()))
    static member Max (a:D, b:D) = D(ADD.Max(a.toADD(), b.toADD()))
    static member Min (a:D, b:D) = D(ADD.Min(a.toADD(), b.toADD()))

and DV(v:ADDV) =
    new(v:AD.number[]) = DV(ADDV.DV(v))
    new(v:D[]) = DV(AD.DOps.toDV(v |> Array.map D.DtoADD))
    member internal this.toADDV() = v
    static member internal ADDVtoDV (v:ADDV) = new DV(v)
    static member internal DVtoADDV (v:DV) = v.toADDV()

    member d.P = d.toADDV().P |> DV.ADDVtoDV
    member d.T = d.toADDV().T |> DV.ADDVtoDV

    member d.Item
        with get i = d.toADDV().[i] |> D.ADDtoD

    override d.ToString() =
        let rec s (d:ADDV) =
            match d with
            | AD.DV(p) -> sprintf "DV %A" p
            | AD.DVF(p, t, _) -> sprintf "DVF (%A, %A)" (s p) (s t)
            | AD.DVR(p, op, _, _) -> sprintf "DVR (%A, %A)" (s p) (op.ToString())
        s (d.toADDV())
    member d.Visualize() = d.toADDV().Visualize()
    static member op_Explicit(d:DV):AD.number[] = ADDV.op_Explicit(d.toADDV())
    static member op_Implicit(a:AD.number[]):DV = DV(a)
    static member Zero = DV(Array.empty<AD.number>)
    // DV - DV binary operations
    static member (+) (a:DV, b:DV) = DV(a.toADDV() + b.toADDV())
    static member (-) (a:DV, b:DV) = DV(a.toADDV() - b.toADDV())
    static member (*) (a:DV, b:DV) = D(a.toADDV() * b.toADDV())
    static member PointwiseMultiply (a:DV, b:DV) = DV(a.toADDV() .* b.toADDV())
    static member TensorMultiply (a:DV, b:DV) = DM(a.toADDV() &* b.toADDV())
    static member PointwiseDivision (a:DV, b:DV) = DV(a.toADDV() ./ b.toADDV())
    static member Pow (a:DV, b:DV) = DV(a.toADDV() ** b.toADDV())
    static member Atan2 (a:DV, b:DV) = DV(atan2 (a.toADDV()) (b.toADDV()))
    // DV - D binary operations
    static member (+) (a:DV, b:D) = DV(a.toADDV() + b.toADD())
    static member (-) (a:DV, b:D) = DV(a.toADDV() - b.toADD())
    static member (*) (a:DV, b:D) = DV(a.toADDV() * b.toADD())
    static member (/) (a:DV, b:D) = DV(a.toADDV() / b.toADD())
    static member Pow (a:DV, b:D) = DV(a.toADDV() ** b.toADD())
    static member Atan2 (a:DV, b:D) = DV(ADDV.Atan2(a.toADDV(), b.toADD()))
    // D - DV binary operations
    static member (+) (a:D, b:DV) = DV(a.toADD() + b.toADDV())
    static member (-) (a:D, b:DV) = DV(a.toADD() - b.toADDV())
    static member (*) (a:D, b:DV) = DV(a.toADD() * b.toADDV())
    static member (/) (a:D, b:DV) = DV(a.toADD() / b.toADDV())
    static member Pow (a:D, b:DV) = DV(ADDV.Pow(a.toADD(), b.toADDV()))
    static member Atan2 (a:D, b:DV) = DV(ADDV.Atan2(a.toADD(), b.toADDV()))
    // DV - AD.number binary operations
    static member (+) (a:DV, b:AD.number) = a + (D b)
    static member (-) (a:DV, b:AD.number) = a - (D b)
    static member (*) (a:DV, b:AD.number) = a * (D b)
    static member (/) (a:DV, b:AD.number) = a / (D b)
    static member Pow (a:DV, b:AD.number) = a ** (D b)
    static member Atan2 (a:DV, b:AD.number) = DV.Atan2(a, D b)
    // AD.number - DV binary operations
    static member (+) (a:AD.number, b:DV) = (D a) + b
    static member (-) (a:AD.number, b:DV) = (D a) - b
    static member (*) (a:AD.number, b:DV) = (D a) * b
    static member (/) (a:AD.number, b:DV) = (D a) / b
    static member Pow (a:AD.number, b:DV) = DV.Pow(D a, b)
    static member Atan2 (a:AD.number, b:DV) = DV.Atan2(D a, b)
    // DV - int binary operations
    static member (+) (a:DV, b:int) = a + (D (AD.N.toNumber b))
    static member (-) (a:DV, b:int) = a - (D (AD.N.toNumber b))
    static member (*) (a:DV, b:int) = a * (D (AD.N.toNumber b))
    static member (/) (a:DV, b:int) = a / (D (AD.N.toNumber b))
    static member Pow (a:DV, b:int) = DV.Pow(a, (D (AD.N.toNumber b)))
    static member Atan2 (a:DV, b:int) = DV.Atan2(a, (D (AD.N.toNumber b)))
    // int - DV binary operations
    static member (+) (a:int, b:DV) = (D (AD.N.toNumber a)) + b
    static member (-) (a:int, b:DV) = (D (AD.N.toNumber a)) - b
    static member (*) (a:int, b:DV) = (D (AD.N.toNumber a)) * b
    static member (/) (a:int, b:DV) = (D (AD.N.toNumber a)) / b
    static member Pow (a:int, b:DV) = DV.Pow((D (AD.N.toNumber a)), b)
    static member Atan2 (a:int, b:DV) = DV.Atan2((D (AD.N.toNumber a)), b)
    // DV unary operations
    static member Log (a:DV) = DV(log (a.toADDV()))
    static member Log10 (a:DV) = DV(log10 (a.toADDV()))
    static member Exp (a:DV) = DV(exp (a.toADDV()))
    static member Sin (a:DV) = DV(sin (a.toADDV()))
    static member Cos (a:DV) = DV(cos (a.toADDV()))
    static member Tan (a:DV) = DV(tan (a.toADDV()))
    static member Neg (a:DV) = DV(-(a.toADDV()))
    static member Sqrt (a:DV) = DV(sqrt (a.toADDV()))
    static member Sinh (a:DV) = DV(sinh (a.toADDV()))
    static member Cosh (a:DV) = DV(cosh (a.toADDV()))
    static member Tanh (a:DV) = DV(tanh (a.toADDV()))
    static member Asin (a:DV) = DV(asin (a.toADDV()))
    static member Acos (a:DV) = DV(acos (a.toADDV()))
    static member Atan (a:DV) = DV(atan (a.toADDV()))
    static member Abs (a:DV) = DV(abs (a.toADDV()))
    static member Floor (a:DV) = DV(floor (a.toADDV()))
    static member Ceiling (a:DV) = DV(ceil (a.toADDV()))
    static member Round (a:DV) = DV(round (a.toADDV()))
    static member Sign (a:DV) = DV(ADDV.Sign(a.toADDV()))
    static member SoftPlus (a:DV) = DV(ADDV.SoftPlus(a.toADDV()))
    static member SoftSign (a:DV) = DV(ADDV.SoftSign(a.toADDV()))
    static member Max (a:DV, b:DV) = DV(ADDV.Max(a.toADDV(), b.toADDV()))
    static member Min (a:DV, b:DV) = DV(ADDV.Min(a.toADDV(), b.toADDV()))
    static member Sum (a:DV) = D(ADDV.Sum(a.toADDV()))
    static member L1Norm (a:DV) = D(ADDV.L1Norm(a.toADDV()))
    static member L2Norm (a:DV) = D(ADDV.L2Norm(a.toADDV()))
    static member L2NormSq (a:DV) = D(ADDV.L2NormSq(a.toADDV()))
    static member Max (a:DV) = D(ADDV.Max(a.toADDV()))
    static member MaxIndex (a:DV) = ADDV.MaxIndex(a.toADDV())
    static member Min (a:DV) = D(ADDV.Min(a.toADDV()))
    static member MinIndex (a:DV) = ADDV.MinIndex(a.toADDV())
    static member SoftMax (a:DV) = DV(ADDV.SoftMax(a.toADDV()))
    static member Mean (a:DV) = D(ADDV.Mean(a.toADDV()))
    static member StandardDev (a:DV) = D(ADDV.StandardDev(a.toADDV()))
    static member Variance (a:DV) = D(ADDV.Variance(a.toADDV()))
    static member Normalize (a:DV) = DV(ADDV.Normalize(a.toADDV()))
    static member Standardize (a:DV) = DV(ADDV.Standardize(a.toADDV()))

and DM(m:ADDM) =
    new(m:AD.number[, ]) = DM(ADDM.DM(m))
    member internal this.toADDM() = m
    static member internal ADDMtoDM (x:ADDM) = new DM(x)
    static member internal DMtoADDM (x:DM) = x.toADDM()

    member d.P = d.toADDM().P |> DM.ADDMtoDM
    member d.T = d.toADDM().T |> DM.ADDMtoDM

    member d.Item
        with get (i, j) = d.toADDM().[i, j] |> D.ADDtoD

    override d.ToString() =
        let rec s (d:ADDM) =
            match d with
            | AD.DM(p) -> sprintf "DM %A" p
            | AD.DMF(p, t, _) -> sprintf "DMF (%A, %A)" (s p) (s t)
            | AD.DMR(p, op, _, _) -> sprintf "DMR (%A, %A)" (s p) (op.ToString())
        s (d.toADDM())
    member d.Visualize() = d.toADDM().Visualize()
    static member op_Explicit(d:DM):AD.number[, ] = ADDM.op_Explicit(d.toADDM())
    static member op_Implicit(a:AD.number[, ]):DM = DM(a)
    static member Zero = DM(Array2D.empty)

    // DM - DM binary operations
    static member (+) (a:DM, b:DM) = DM(a.toADDM() + b.toADDM())
    static member (-) (a:DM, b:DM) = DM(a.toADDM() - b.toADDM())
    static member (*) (a:DM, b:DM) = DM(a.toADDM() * b.toADDM())
    static member PointwiseMultiply (a:DM, b:DM) = DM(a.toADDM() .* b.toADDM())
    static member PointwiseDivision (a:DM, b:DM) = DM(a.toADDM() ./ b.toADDM())
    static member Pow (a:DM, b:DM) = DM(a.toADDM() ** b.toADDM())
    static member Atan2 (a:DM, b:DM) = DM(atan2 (a.toADDM()) (b.toADDM()))

    // DM - DV binary operations
    static member (+) (a:DV, b:DM) = DM(a.toADDV() + b.toADDM())
    static member (+) (a:DM, b:DV) = DM(a.toADDM() + b.toADDV())
    static member (-) (a:DV, b:DM) = DM(a.toADDV() - b.toADDM())
    static member (-) (a:DM, b:DV) = DM(a.toADDM() - b.toADDV())
    static member (*) (a:DM, b:DV) = DV(a.toADDM() * b.toADDV())
    static member (*) (a:DV, b:DM) = DV(a.toADDV() * b.toADDM())

    // DV - D binary operations
    static member (+) (a:DM, b:D) = DM(a.toADDM() + b.toADD())
    static member (-) (a:DM, b:D) = DM(a.toADDM() - b.toADD())
    static member (*) (a:DM, b:D) = DM(a.toADDM() * b.toADD())
    static member (/) (a:DM, b:D) = DM(a.toADDM() / b.toADD())
    static member Pow (a:DM, b:D) = DM(a.toADDM() ** b.toADD())
    static member Atan2 (a:DM, b:D) = DM(ADDM.Atan2(a.toADDM(), b.toADD()))
    // D - DV binary operations
    static member (+) (a:D, b:DM) = DM(a.toADD() + b.toADDM())
    static member (-) (a:D, b:DM) = DM(a.toADD() - b.toADDM())
    static member (*) (a:D, b:DM) = DM(a.toADD() * b.toADDM())
    static member (/) (a:D, b:DM) = DM(a.toADD() / b.toADDM())
    static member Pow (a:D, b:DM) = DM(ADDM.Pow(a.toADD(), b.toADDM()))
    static member Atan2 (a:D, b:DM) = DM(ADDM.Atan2(a.toADD(), b.toADDM()))
    // DV - number binary operations
    static member (+) (a:DM, b:AD.number) = a + (D b)
    static member (-) (a:DM, b:AD.number) = a - (D b)
    static member (*) (a:DM, b:AD.number) = a * (D b)
    static member (/) (a:DM, b:AD.number) = a / (D b)
    static member Pow (a:DM, b:AD.number) = a ** (D b)
    static member Atan2 (a:DM, b:AD.number) = DM.Atan2(a, D b)
    // number - DV binary operations
    static member (+) (a:AD.number, b:DM) = (D a) + b
    static member (-) (a:AD.number, b:DM) = (D a) - b
    static member (*) (a:AD.number, b:DM) = (D a) * b
    static member (/) (a:AD.number, b:DM) = (D a) / b
    static member Pow (a:AD.number, b:DM) = DM.Pow(D a, b)
    static member Atan2 (a:AD.number, b:DM) = DM.Atan2(D a, b)
    // DV - int binary operations
    static member (+) (a:DM, b:int) = a + (D (AD.N.toNumber b))
    static member (-) (a:DM, b:int) = a - (D (AD.N.toNumber b))
    static member (*) (a:DM, b:int) = a * (D (AD.N.toNumber b))
    static member (/) (a:DM, b:int) = a / (D (AD.N.toNumber b))
    static member Pow (a:DM, b:int) = DM.Pow(a, (D (AD.N.toNumber b)))
    static member Atan2 (a:DM, b:int) = DM.Atan2(a, (D (AD.N.toNumber b)))
    // int - DV binary operations
    static member (+) (a:int, b:DM) = (D (AD.N.toNumber a)) + b
    static member (-) (a:int, b:DM) = (D (AD.N.toNumber a)) - b
    static member (*) (a:int, b:DM) = (D (AD.N.toNumber a)) * b
    static member (/) (a:int, b:DM) = (D (AD.N.toNumber a)) / b
    static member Pow (a:int, b:DM) = DM.Pow((D (AD.N.toNumber a)), b)
    static member Atan2 (a:int, b:DM) = DM.Atan2((D (AD.N.toNumber a)), b)
    // DV unary operations
    static member Log (a:DM) = DM(log (a.toADDM()))
    static member Log10 (a:DM) = DM(log10 (a.toADDM()))
    static member Exp (a:DM) = DM(exp (a.toADDM()))
    static member Sin (a:DM) = DM(sin (a.toADDM()))
    static member Cos (a:DM) = DM(cos (a.toADDM()))
    static member Tan (a:DM) = DM(tan (a.toADDM()))
    static member Neg (a:DM) = DM(-(a.toADDM()))
    static member Sqrt (a:DM) = DM(sqrt (a.toADDM()))
    static member Sinh (a:DM) = DM(sinh (a.toADDM()))
    static member Cosh (a:DM) = DM(cosh (a.toADDM()))
    static member Tanh (a:DM) = DM(tanh (a.toADDM()))
    static member Asin (a:DM) = DM(asin (a.toADDM()))
    static member Acos (a:DM) = DM(acos (a.toADDM()))
    static member Atan (a:DM) = DM(atan (a.toADDM()))
    static member Abs (a:DM) = DM(abs (a.toADDM()))
    static member Floor (a:DM) = DM(floor (a.toADDM()))
    static member Ceiling (a:DM) = DM(ceil (a.toADDM()))
    static member Round (a:DM) = DM(round (a.toADDM()))
    static member Sign (a:DM) = DM(ADDM.Sign(a.toADDM()))
    static member Sum (a:DM) = D(ADDM.Sum(a.toADDM()))
    static member Transpose (a:DM) = DM(ADDM.Transpose(a.toADDM()))
    static member Diagonal (a:DM) = DV(ADDM.Diagonal(a.toADDM()))
    static member Trace (a:DM) = D(ADDM.Trace(a.toADDM()))
    static member Solve (a:DM, b:DV) = DV(ADDM.Solve(a.toADDM(), b.toADDV()))
    static member SolveSymmetric (a:DM, b:DV) = DV(ADDM.SolveSymmetric(a.toADDM(), b.toADDV()))
    static member Inverse (a:DM) = DM(ADDM.Inverse(a.toADDM()))
    static member Det (a:DM) = D(ADDM.Det(a.toADDM()))
    static member ReLU (a:DM) = DM(ADDM.ReLU(a.toADDM()))
    static member Sigmoid (a:DM) = DM(ADDM.Sigmoid(a.toADDM()))
    static member SoftPlus (a:DM) = DM(ADDM.SoftPlus(a.toADDM()))
    static member SoftSign (a:DM) = DM(ADDM.SoftSign(a.toADDM()))
    static member Max (a:DM, b:DM) = DM(ADDM.Max(a.toADDM(), b.toADDM()))
    static member Min (a:DM, b:DM) = DM(ADDM.Min(a.toADDM(), b.toADDM()))
    static member Max (a:DM, b:D) = DM(ADDM.Max(a.toADDM(), b.toADD()))
    static member Max (a:D, b:DM) = DM(ADDM.Max(a.toADD(), b.toADDM()))
    static member Min (a:DM, b:D) = DM(ADDM.Min(a.toADDM(), b.toADD()))
    static member Min (a:D, b:DM) = DM(ADDM.Min(a.toADD(), b.toADDM()))
    static member MaxIndex (a:DM) = ADDM.MaxIndex(a.toADDM())
    static member MinIndex (a:DM) = ADDM.MinIndex(a.toADDM())
    static member Mean (a:DM) = D(ADDM.Mean(a.toADDM()))
    static member StandardDev (a:DM) = D(ADDM.StandardDev(a.toADDM()))
    static member Variance (a:DM) = D(ADDM.Variance(a.toADDM()))
    static member Normalize (a:DM) = DM(ADDM.Normalize(a.toADDM()))
    static member Standardize (a:DM) = DM(ADDM.Standardize(a.toADDM()))

and ADAdjoints = AD.Adjoints

and Adjoints() =
    let m = ADAdjoints()
    member internal this.toADAdjoints() = m
    member this.Item with get (d:D) = m.[d.toADD()] |> D.ADDtoD
    member this.Item with get (d:DV) = m.[d.toADDV()] |> DV.ADDVtoDV
    member this.Item with get (d:DM) = m.[d.toADDM()] |> DM.ADDMtoDM


/// Nested forward and reverse mode automatic differentiation module
type AD =

    /// First derivative of a scalar-to-scalar function `f`
    static member Diff(f:System.Func<D, D>) = System.Func<D, D>(D.DtoADD >> (AD.DiffOps.diff (D.ADDtoD >> f.Invoke >> D.DtoADD)) >> D.ADDtoD)

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff(f:System.Func<D, D>, x:D) = D.ADDtoD <| AD.DiffOps.diff (D.ADDtoD >> f.Invoke >> D.DtoADD) (x |> D.DtoADD)

    /// Second derivative of a scalar-to-scalar function `f`
    static member Diff2(f:System.Func<D, D>) = System.Func<D, D>(D.DtoADD >> (AD.DiffOps.diff2 (D.ADDtoD >> f.Invoke >> D.DtoADD)) >> D.ADDtoD)

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff2(f:System.Func<D, D>, x:D) = D.ADDtoD <| AD.DiffOps.diff2 (D.ADDtoD >> f.Invoke >> D.DtoADD) (x |> D.DtoADD)

    /// `n`-th derivative of a scalar-to-scalar function `f`
    static member Diffn(n:int, f:System.Func<D, D>) = System.Func<D, D>(D.DtoADD >> (AD.DiffOps.diffn n (D.ADDtoD >> f.Invoke >> D.DtoADD)) >> D.ADDtoD)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diffn(n:int, f:System.Func<D, D>, x:D) = D.ADDtoD <| AD.DiffOps.diffn n (D.ADDtoD >> f.Invoke >> D.DtoADD) (x |> D.DtoADD)

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    static member Gradv(f:System.Func<DV, D>, x:DV, v:DV) = D.ADDtoD <| AD.DiffOps.gradv (DV.ADDVtoDV >> f.Invoke >> D.DtoADD) (x |> DV.DVtoADDV) (v |> DV.DVtoADDV)

    /// Gradient of a vector-to-scalar function `f`
    static member Grad(f:System.Func<DV, D>) = System.Func<DV, DV>(DV.DVtoADDV >> (AD.DiffOps.grad (DV.ADDVtoDV >> f.Invoke >> D.DtoADD)) >> DV.ADDVtoDV)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    static member Grad(f:System.Func<DV, D>, x:DV) = DV.ADDVtoDV <| AD.DiffOps.grad ((DV.ADDVtoDV) >> f.Invoke >> D.DtoADD) (x |> DV.DVtoADDV)

    /// Laplacian of a vector-to-scalar function `f`
    static member Laplacian(f:System.Func<DV, D>) = System.Func<DV, D>(DV.DVtoADDV >> (AD.DiffOps.laplacian (DV.ADDVtoDV >> f.Invoke >> D.DtoADD)) >> D.ADDtoD)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    static member Laplacian(f:System.Func<DV, D>, x:DV) = D.ADDtoD <| AD.DiffOps.laplacian (DV.ADDVtoDV >> f.Invoke >> D.DtoADD) (x |> DV.DVtoADDV)

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    static member Jacobianv(f:System.Func<DV, DV>, x:DV, v:DV) = DV.ADDVtoDV <| AD.DiffOps.jacobianv (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV) (x |> DV.DVtoADDV) (v |> DV.DVtoADDV)

    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    static member JacobianTv(f:System.Func<DV, DV>, x:DV, v:DV) = DV.ADDVtoDV <| AD.DiffOps.jacobianTv (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV) (x |> DV.DVtoADDV) (v |> DV.DVtoADDV)

    /// Jacobian of a vector-to-vector function `f`
    static member Jacobian(f:System.Func<DV, DV>) = System.Func<DV, DM>(DV.DVtoADDV >> (AD.DiffOps.jacobian (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV)) >> DM.ADDMtoDM)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    static member Jacobian(f:System.Func<DV, DV>, x:DV) = DM.ADDMtoDM <| AD.DiffOps.jacobian (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV) (x |> DV.DVtoADDV)

    /// Transposed Jacobian of a vector-to-vector function `f`
    static member JacobianT(f:System.Func<DV, DV>) = System.Func<DV, DM>(DV.DVtoADDV >> (AD.DiffOps.jacobianT (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV)) >> DM.ADDMtoDM)

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    static member JacobianT(f:System.Func<DV, DV>, x:DV) = DM.ADDMtoDM <| AD.DiffOps.jacobianT (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV) (x |> DV.DVtoADDV)

    /// Hessian of a vector-to-scalar function `f`
    static member Hessian(f:System.Func<DV, D>) = System.Func<DV, DM>(DV.DVtoADDV >> (AD.DiffOps.hessian (DV.ADDVtoDV >> f.Invoke >> D.DtoADD)) >> DM.ADDMtoDM)

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    static member Hessian(f:System.Func<DV, D>, x:DV) = DM.ADDMtoDM <| AD.DiffOps.hessian (DV.ADDVtoDV >> f.Invoke >> D.DtoADD) (x |> DV.DVtoADDV)

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`
    static member Hessianv(f:System.Func<DV, D>, x:DV, v:DV) = DV.ADDVtoDV <| AD.DiffOps.hessianv (DV.ADDVtoDV >> f.Invoke >> D.DtoADD) (x |> DV.DVtoADDV) (v |> DV.DVtoADDV)

    /// Curl of a vector-to-vector function `f`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<DV, DV>) = System.Func<DV, DV>(DV.DVtoADDV >> (AD.DiffOps.curl (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV)) >> DV.ADDVtoDV)

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<DV, DV>, x:DV) = DV.ADDVtoDV <| AD.DiffOps.curl (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV) (x |> DV.DVtoADDV)

    /// Divergence of a vector-to-vector function `f`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<DV, DV>) = System.Func<DV, D>(DV.DVtoADDV >> (AD.DiffOps.div (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV)) >> D.ADDtoD)

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<DV, DV>, x:DV) = D.ADDtoD <| AD.DiffOps.div (DV.ADDVtoDV >> f.Invoke >> DV.DVtoADDV) (x |> DV.DVtoADDV)

    /// Returns a specified number raised to the specified power.
    static member inline Pow(a:'T, b:'U) = a ** b

    /// Returns the angle whose tangent is the quotient of two specified numbers.
    static member inline Atan2(a:'T, b:'T) = atan2 a b

    /// Returns the logarithm of a specified number.
    static member inline Log(a:'T) = log a

    /// Returns the base 10 logarithm of a specified number.
    static member inline Log10(a:'T) = log10 a

    /// Returns e raised to the specified power.
    static member inline Exp(a:'T) = exp a

    /// Returns the sine of the specified angle.
    static member inline Sin(a:'T) = sin a

    /// Returns the cosine of the specified angle.
    static member inline Cos(a:'T) = cos a

    /// Returns the tangent of the specified angle.
    static member inline Tan(a:'T) = tan a

    /// Returns the square root of a specified number.
    static member inline Sqrt(a:'T) = sqrt a

    /// Returns the hyperbolic sine of the specified angle.
    static member inline Sinh(a:'T) = sinh a

    /// Returns the hyperbolic cosine of the specified angle.
    static member inline Cosh(a:'T) = cosh a

    /// Returns the hyperbolic tangent of the specified angle.
    static member inline Tanh(a:'T) = tanh a

    /// Returns the angle whose sine is the specified number.
    static member inline Asin(a:'T) = asin a

    /// Returns the angle whose cosine is the specified number.
    static member inline Acos(a:'T) = acos a

    /// Returns the angle whose tangent is the specified number.
    static member inline Atan(a:'T) = atan a

    /// Returns the absolute value of a specified number.
    static member inline Abs(a:'T) = abs a

    /// Returns the largest integer less than or equal to the specified number.
    static member inline Floor(a:'T) = floor a

    /// Returns the smallest integer greater than or equal to the specified number.
    static member inline Ceiling(a:'T) = ceil a

    /// Rounds a value to the nearest integer or to the specified number of fractional digits.
    static member inline Round(a:'T) = round a

    /// Returns the larger of two specified numbers.
    static member inline Max(a:'T, b:'T) = min a b

    /// Returns the smaller of two numbers.
    static member inline Min(a:'T, b:'T) = min a b
    static member inline LogSumExp(a:'T) = (^T : (static member LogSumExp : ^T -> ^U) a)
    static member inline SoftPlus(a:'T) = (^T : (static member SoftPlus : ^T -> ^T) a)
    static member inline SoftSign(a:'T) = (^T : (static member SoftSign : ^T -> ^T) a)
    static member inline Sigmoid(a:'T) = (^T : (static member Sigmoid : ^T -> ^T) a)
    static member inline ReLU(a:'T) = (^T : (static member ReLU : ^T -> ^T) a)
    static member inline SoftMax(a:'T) = (^T : (static member SoftMax : ^T -> ^T) a)
    static member inline Max(a:'T, b:'U):^V = ((^T or ^U) : (static member Max : ^T * ^U -> ^V) a, b)
    static member inline Min(a:'T, b:'U):^V = ((^T or ^U) : (static member Min : ^T * ^U -> ^V) a, b)
    static member inline Signum(a:'T) = (^T : (static member Sign : ^T -> ^T) a)
    static member inline Mean(a:'T) = (^T : (static member Mean : ^T -> D) a)
    static member inline StandardDev(a:'T) = (^T : (static member StandardDev : ^T -> D) a)
    static member inline Variance(a:'T) = (^T : (static member Variance : ^T -> D) a)
    static member inline Normalize(a:'T) = (^T : (static member Normalize : ^T -> ^T) a)
    static member inline Standardize(a:'T) = (^T : (static member Standardize : ^T -> ^T) a)
    static member L1Norm(a:DV) = D(ADDV.L1Norm(a.toADDV()))
    static member L2Norm(a:DV) = D(ADDV.L2Norm(a.toADDV()))
    static member L2NormSq(a:DV) = D(ADDV.L2NormSq(a.toADDV()))
    static member Sum(a:DV) = D(ADDV.Sum(a.toADDV()))
    static member Sum(a:DM) = D(ADDM.Sum(a.toADDM()))
    static member Transpose (a:DM) = DM(ADDM.Transpose(a.toADDM()))
    static member Diagonal (a:DM) = DV(ADDM.Diagonal(a.toADDM()))
    static member Trace (a:DM) = D(ADDM.Trace(a.toADDM()))
    static member Solve (a:DM, b:DV) = DV(ADDM.Solve(a.toADDM(), b.toADDV()))
    static member SolveSymmetric (a:DM, b:DV) = DV(ADDM.SolveSymmetric(a.toADDM(), b.toADDV()))
    static member Inverse (a:DM) = DM(ADDM.Inverse(a.toADDM()))


/// Numerical differentiation module
type Numerical =

    /// First derivative of a scalar-to-scalar function `f`
    static member Diff(f:System.Func<AD.number, AD.number>) = System.Func<AD.number, AD.number>(Numerical.DiffOps.diff f.Invoke)

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff(f:System.Func<AD.number, AD.number>, x:AD.number) = Numerical.DiffOps.diff f.Invoke x

    /// Second derivative of a scalar-to-scalar function `f`
    static member Diff2(f:System.Func<AD.number, AD.number>) = System.Func<AD.number, AD.number>(Numerical.DiffOps.diff2 f.Invoke)

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff2(f:System.Func<AD.number, AD.number>, x:AD.number) = Numerical.DiffOps.diff2 f.Invoke x

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    static member Gradv(f:System.Func<AD.number[], AD.number>, x:AD.number[], v:AD.number[]) = Numerical.DiffOps.gradv f.Invoke x v

    /// Gradient of a vector-to-scalar function `f`
    static member Grad(f:System.Func<AD.number[], AD.number>) = System.Func<AD.number[], AD.number[]>(Numerical.DiffOps.grad f.Invoke)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    static member Grad(f:System.Func<AD.number[], AD.number>, x:AD.number[]) = Numerical.DiffOps.grad f.Invoke x

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    static member Hessianv(f:System.Func<AD.number[], AD.number>, x:AD.number[], v:AD.number[]) = Numerical.DiffOps.hessianv f.Invoke x v

    /// Hessian of a vector-to-scalar function `f`
    static member Hessian(f:System.Func<AD.number[], AD.number>) = System.Func<AD.number[], AD.number[, ]>(Numerical.DiffOps.hessian f.Invoke)

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    static member Hessian(f:System.Func<AD.number[], AD.number>, x:AD.number[]) = Numerical.DiffOps.hessian f.Invoke x

    /// Laplacian of a vector-to-scalar function `f`
    static member Laplacian(f:System.Func<AD.number[], AD.number>) = System.Func<AD.number[], AD.number>(Numerical.DiffOps.laplacian f.Invoke)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    static member Laplacian(f:System.Func<AD.number[], AD.number>, x:AD.number[]) = Numerical.DiffOps.laplacian f.Invoke x

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    static member Jacobianv(f:System.Func<AD.number[], AD.number[]>, x:AD.number[], v:AD.number[]) = Numerical.DiffOps.jacobianv f.Invoke x v

    /// Jacobian of a vector-to-vector function `f`
    static member Jacobian(f:System.Func<AD.number[], AD.number[]>) = System.Func<AD.number[], AD.number[, ]>(Numerical.DiffOps.jacobian f.Invoke)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    static member Jacobian(f:System.Func<AD.number[], AD.number[]>, x:AD.number[]) = Numerical.DiffOps.jacobian f.Invoke x

    /// Transposed Jacobian of a vector-to-vector function `f`
    static member JacobianT(f:System.Func<AD.number[], AD.number[]>) = System.Func<AD.number[], AD.number[, ]>(Numerical.DiffOps.jacobianT f.Invoke)

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    static member JacobianT(f:System.Func<AD.number[], AD.number[]>, x:AD.number[]) = Numerical.DiffOps.jacobianT f.Invoke x

    /// Curl of a vector-to-vector function `f`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<AD.number[], AD.number[]>) = System.Func<AD.number[], AD.number[]>(Numerical.DiffOps.curl f.Invoke)

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<AD.number[], AD.number[]>, x:AD.number[]) = Numerical.DiffOps.curl f.Invoke x

    /// Divergence of a vector-to-vector function `f`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<AD.number[], AD.number[]>) = System.Func<AD.number[], AD.number>(Numerical.DiffOps.div f.Invoke)

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<AD.number[], AD.number[]>, x:AD.number[]) = Numerical.DiffOps.div f.Invoke x
