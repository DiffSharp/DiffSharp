// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

module DiffSharp.Tests.Backend.OpenBLAS

open NUnit.Framework
open FsCheck.NUnit
open DiffSharp.Util
open DiffSharp.Backend
open DiffSharp.Tests

let Float32Backend = DiffSharp.Backend.OpenBLAS.Float32Backend() :> Backend<float32>
let Float64Backend = DiffSharp.Backend.OpenBLAS.Float64Backend() :> Backend<float>


//
// Random tests
//

// float32
[<Property>]
let ``OpenBLAS.32.Mul_Dot_V_V``(v:float32[]) =
    let r = v |> Array.map (fun x -> x * x) |> Array.sum
    Util.(=~)(Float32Backend.Mul_Dot_V_V(v, v), r)

[<Property>]
let ``OpenBLAS.32.L1Norm_V``(v:float32[]) =
    let r = v |> Array.map abs |> Array.sum
    Util.(=~)(Float32Backend.L1Norm_V(v), r)

[<Property>]
let ``OpenBLAS.32.L2Norm_V``(v:float32[]) =
    let r = v |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
    Util.(=~)(Float32Backend.L2Norm_V(v), r)

[<Property>]
let ``OpenBLAS.32.Sum_V``(v:float32[]) =
    let r = v |> Array.sum
    Util.(=~)(Float32Backend.Sum_V(v), r)

[<Property>]
let ``OpenBLAS.32.Sum_M``(m:float32[,]) =
    let mutable r = 0.f
    for i = 0 to (Array2D.length1 m) - 1 do
        for j = 0 to (Array2D.length2 m) - 1 do
            r <- r + m.[i, j]
    Util.(=~)(Float32Backend.Sum_M(m), r)

[<Property>]
let ``OpenBLAS.32.Add_V_V``(v:float32[]) =
    let r = Array.map2 (+) v v
    Util.(=~)(Float32Backend.Add_V_V(v, v), r)

[<Property>]
let ``OpenBLAS.32.Add_S_V``(s:float32, v:float32[]) =
    let r = v |> Array.map ((+) s)
    Util.(=~)(Float32Backend.Add_S_V(s, v), r)

[<Property>]
let ``OpenBLAS.32.Sub_V_V``(v:float32[]) =
    let r = v |> Array.map (fun x -> x - x)
    Util.(=~)(Float32Backend.Sub_V_V(v, v), r)

[<Property>]
let ``OpenBLAS.32.Sub_S_V``(s:float32, v:float32[]) =
    let r = v |> Array.map (fun x -> s - x)
    Util.(=~)(Float32Backend.Sub_S_V(s, v), r)

[<Property>]
let ``OpenBLAS.32.Sub_V_S``(v:float32[], s:float32) =
    let r = v |> Array.map (fun x -> x - s)
    Util.(=~)(Float32Backend.Sub_V_S(v, s), r)

[<Property>]
let ``OpenBLAS.32.Mul_S_V``(s:float32, v:float32[]) =
    if not (Util.IsNice(s)) then
        true
    else
        let r =
            if (s = 0.f) then
                Array.zeroCreate v.Length
            else
                v |> Array.map (fun x -> s * x)
        Util.(=~)(Float32Backend.Mul_S_V(s, v), r)

[<Property>]
let ``OpenBLAS.32.Diagonal_M``(m:float32[,]) =
    let n = min (Array2D.length1 m) (Array2D.length2 m)
    let r = Array.init n (fun i -> m.[i, i])
    Util.(=~)(Float32Backend.Diagonal_M(m), r)

[<Property>]
let ``OpenBLAS.32.Map_F_V``(v:float32[]) =
    let f (x:float32) = sin (exp x)
    let r = v |> Array.map f
    Util.(=~)(Float32Backend.Map_F_V(f, v), r)

[<Property>]
let ``OpenBLAS.32.Map2_F_V_V``(v:float32[]) =
    let f (x1:float32) (x2:float32) = sin (exp x1) + cos x2
    let r = Array.map2 f v v
    Util.(=~)(Float32Backend.Map2_F_V_V(f, v, v), r)

[<Property>]
let ``OpenBLAS.32.ReshapeCopy_MRows_V``(m:float32[,]) =
    let r = Array.zeroCreate m.Length
    let mutable ri = 0
    for i = 0 to (Array2D.length1 m) - 1 do
        for j = 0 to (Array2D.length2 m) - 1 do
            r.[ri] <- m.[i, j]
            ri <- ri + 1
    Util.(=~)(Float32Backend.ReshapeCopy_MRows_V(m), r)

[<Property>]
let ``OpenBLAS.32.Mul_Out_V_V``(v1:float32[], v2:float32[]) =
    let r =
        if (v1.Length = 0) || (v2.Length = 0) then
            Array2D.empty
        else
            let rr = Array2D.zeroCreate v1.Length v2.Length
            for i = 0 to v1.Length - 1 do
                for j = 0 to v2.Length - 1 do
                    rr.[i, j] <- v1.[i] * v2.[j]
            rr
    Util.(=~)(Float32Backend.Mul_Out_V_V(v1, v2), r)

[<Property>]
let ``OpenBLAS.32.Add_M_M``(m:float32[,]) =
    let r = m |> Array2D.map (fun x -> 2.f * x)
    Util.(=~)(Float32Backend.Add_M_M(m, m), r)

[<Property>]
let ``OpenBLAS.32.Add_S_M``(s:float32, m:float32[,]) =
    let r =
        if Array2D.isEmpty m then
            Array2D.empty
        else
            m |> Array2D.map (fun x -> s + x)
    Util.(=~)(Float32Backend.Add_S_M(s, m), r)

[<Property>]
let ``OpenBLAS.32.Add_V_MCols``(v:float32[]) =
    let m = v.Length
    let n = 3
    let r =
        if m = 0 then
            Array2D.empty
        else
        let rr = Array2D.zeroCreate v.Length n
        for i = 0 to m - 1 do
            for j = 0 to n - 1 do
                rr.[i, j] <- v.[i]
        rr
    Util.(=~)(Float32Backend.Add_V_MCols(v, Array2D.zeroCreate m n), r)

[<Property>]
let ``OpenBLAS.32.Sub_M_M``(m:float32[,]) =
    if not (Util.IsNice(m)) then
        true
    else
        let r =
            if m.Length = 0 then
                Array2D.empty
            else
                Array2D.zeroCreate (Array2D.length1 m) (Array2D.length2 m)
        Util.(=~)(Float32Backend.Sub_M_M(m, m), r)

[<Property>]
let ``OpenBLAS.32.Sub_M_S``(m:float32[,], s:float32) =
    let r =
        if m.Length = 0 then
            Array2D.empty
        else
            m |> Array2D.map (fun x -> x - s)
    Util.(=~)(Float32Backend.Sub_M_S(m, s), r)

[<Property>]
let ``OpenBLAS.32.Sub_S_M``(s:float32, m:float32[,]) =
    let r =
        if m.Length = 0 then
            Array2D.empty
        else
            m |> Array2D.map (fun x -> s - x)
    Util.(=~)(Float32Backend.Sub_S_M(s, m), r)

[<Property>]
let ``OpenBLAS.32.Mul_S_M``(s:float32, m:float32[,]) =
    if not (Util.IsNice(s)) then
        true
    else
        let r =
            if (s = 0.f) then
                Array2D.zeroCreate (Array2D.length1 m) (Array2D.length2 m)
            elif m.Length = 0 then
                Array2D.empty
            else
                m |> Array2D.map (fun x -> s * x)
        Util.(=~)(Float32Backend.Mul_S_M(s, m), r)

[<Property>]
let ``OpenBLAS.32.Mul_Had_M_M``(m:float32[,]) =
    let r = m |> Array2D.map (fun x -> x * x)
    Util.(=~)(Float32Backend.Mul_Had_M_M(m, m), r)

[<Property>]
let ``OpenBLAS.32.Transpose_M``(mm:float32[,]) =
    let m = (Array2D.length1 mm)
    let n = (Array2D.length2 mm)
    let r = Array2D.zeroCreate n m
    for i = 0 to n - 1 do
        for j = 0 to m - 1 do
            r.[i, j] <- mm.[j, i]
    Util.(=~)(Float32Backend.Transpose_M(mm), r)

[<Property>]
let ``OpenBLAS.32.Map_F_M``(m:float32[,]) =
    let f (x:float32) = sin (exp x)
    let r = m |> Array2D.map f
    Util.(=~)(Float32Backend.Map_F_M(f, m), r)

[<Property>]
let ``OpenBLAS.32.Map2_F_M_M``(m:float32[,]) =
    let f (x1:float32) (x2:float32) = sin (exp x1) + cos x2
    let r = Array2D.map2 f m m
    Util.(=~)(Float32Backend.Map2_F_M_M(f, m, m), r)

// float64
[<Property>]
let ``OpenBLAS.64.Mul_Dot_V_V``(v:float[]) =
    let r = v |> Array.map (fun x -> x * x) |> Array.sum
    Util.(=~)(Float64Backend.Mul_Dot_V_V(v, v), r)

[<Property>]
let ``OpenBLAS.64.L1Norm_V``(v:float[]) =
    let r = v |> Array.map abs |> Array.sum
    Util.(=~)(Float64Backend.L1Norm_V(v), r)

[<Property>]
let ``OpenBLAS.64.L2Norm_V``(v:float[]) =
    let r = v |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
    Util.(=~)(Float64Backend.L2Norm_V(v), r)

[<Property>]
let ``OpenBLAS.64.Sum_V``(v:float[]) =
    let r = v |> Array.sum
    Util.(=~)(Float64Backend.Sum_V(v), r)

[<Property>]
let ``OpenBLAS.64.Sum_M``(m:float[,]) =
    let mutable r = 0.
    for i = 0 to (Array2D.length1 m) - 1 do
        for j = 0 to (Array2D.length2 m) - 1 do
            r <- r + m.[i, j]
    Util.(=~)(Float64Backend.Sum_M(m), r)

[<Property>]
let ``OpenBLAS.64.Add_V_V``(v:float[]) =
    let r = Array.map2 (+) v v
    Util.(=~)(Float64Backend.Add_V_V(v, v), r)

[<Property>]
let ``OpenBLAS.64.Add_S_V``(s:float, v:float[]) =
    let r = v |> Array.map ((+) s)
    Util.(=~)(Float64Backend.Add_S_V(s, v), r)

[<Property>]
let ``OpenBLAS.64.Sub_V_V``(v:float[]) =
    let r = v |> Array.map (fun x -> x - x)
    Util.(=~)(Float64Backend.Sub_V_V(v, v), r)

[<Property>]
let ``OpenBLAS.64.Sub_S_V``(s:float, v:float[]) =
    let r = v |> Array.map (fun x -> s - x)
    Util.(=~)(Float64Backend.Sub_S_V(s, v), r)

[<Property>]
let ``OpenBLAS.64.Sub_V_S``(v:float[], s:float) =
    let r = v |> Array.map (fun x -> x - s)
    Util.(=~)(Float64Backend.Sub_V_S(v, s), r)

[<Property>]
let ``OpenBLAS.64.Mul_S_V``(s:float, v:float[]) =
    if not (Util.IsNice(s)) then
        true
    else
        let r =
            if (s = 0.) then
                Array.zeroCreate v.Length
            else
                v |> Array.map (fun x -> s * x)
        Util.(=~)(Float64Backend.Mul_S_V(s, v), r)

[<Property>]
let ``OpenBLAS.64.Diagonal_M``(m:float[,]) =
    let n = min (Array2D.length1 m) (Array2D.length2 m)
    let r = Array.init n (fun i -> m.[i, i])
    Util.(=~)(Float64Backend.Diagonal_M(m), r)

[<Property>]
let ``OpenBLAS.64.Map_F_V``(v:float[]) =
    let f (x:float) = sin (exp x)
    let r = v |> Array.map f
    Util.(=~)(Float64Backend.Map_F_V(f, v), r)

[<Property>]
let ``OpenBLAS.64.Map2_F_V_V``(v:float[]) =
    let f (x1:float) (x2:float) = sin (exp x1) + cos x2
    let r = Array.map2 f v v
    Util.(=~)(Float64Backend.Map2_F_V_V(f, v, v), r)

[<Property>]
let ``OpenBLAS.64.ReshapeCopy_MRows_V``(m:float[,]) =
    let r = Array.zeroCreate m.Length
    let mutable ri = 0
    for i = 0 to (Array2D.length1 m) - 1 do
        for j = 0 to (Array2D.length2 m) - 1 do
            r.[ri] <- m.[i, j]
            ri <- ri + 1
    Util.(=~)(Float64Backend.ReshapeCopy_MRows_V(m), r)

[<Property>]
let ``OpenBLAS.64.Mul_Out_V_V``(v1:float[], v2:float[]) =
    let r =
        if (v1.Length = 0) || (v2.Length = 0) then
            Array2D.empty
        else
            let rr = Array2D.zeroCreate v1.Length v2.Length
            for i = 0 to v1.Length - 1 do
                for j = 0 to v2.Length - 1 do
                    rr.[i, j] <- v1.[i] * v2.[j]
            rr
    Util.(=~)(Float64Backend.Mul_Out_V_V(v1, v2), r)

[<Property>]
let ``OpenBLAS.64.Add_M_M``(m:float[,]) =
    let r = m |> Array2D.map (fun x -> 2. * x)
    Util.(=~)(Float64Backend.Add_M_M(m, m), r)

[<Property>]
let ``OpenBLAS.64.Add_S_M``(s:float, m:float[,]) =
    let r =
        if Array2D.isEmpty m then
            Array2D.empty
        else
            m |> Array2D.map (fun x -> s + x)
    Util.(=~)(Float64Backend.Add_S_M(s, m), r)

[<Property>]
let ``OpenBLAS.64.Add_V_MCols``(v:float[]) =
    let m = v.Length
    let n = 3
    let r =
        if m = 0 then
            Array2D.empty
        else
        let rr = Array2D.zeroCreate v.Length n
        for i = 0 to m - 1 do
            for j = 0 to n - 1 do
                rr.[i, j] <- v.[i]
        rr
    Util.(=~)(Float64Backend.Add_V_MCols(v, Array2D.zeroCreate m n), r)

[<Property>]
let ``OpenBLAS.64.Sub_M_M``(m:float[,]) =
    if not (Util.IsNice(m)) then
        true
    else
        let r =
            if m.Length = 0 then
                Array2D.empty
            else
                Array2D.zeroCreate (Array2D.length1 m) (Array2D.length2 m)
        Util.(=~)(Float64Backend.Sub_M_M(m, m), r)

[<Property>]
let ``OpenBLAS.64.Sub_M_S``(m:float[,], s:float) =
    let r =
        if m.Length = 0 then
            Array2D.empty
        else
            m |> Array2D.map (fun x -> x - s)
    Util.(=~)(Float64Backend.Sub_M_S(m, s), r)

[<Property>]
let ``OpenBLAS.64.Sub_S_M``(s:float, m:float[,]) =
    let r =
        if m.Length = 0 then
            Array2D.empty
        else
            m |> Array2D.map (fun x -> s - x)
    Util.(=~)(Float64Backend.Sub_S_M(s, m), r)

[<Property>]
let ``OpenBLAS.64.Mul_S_M``(s:float, m:float[,]) =
    if not (Util.IsNice(s)) then
        true
    else
        let r =
            if (s = 0.) then
                Array2D.zeroCreate (Array2D.length1 m) (Array2D.length2 m)
            elif m.Length = 0 then
                Array2D.empty
            else
                m |> Array2D.map (fun x -> s * x)
        Util.(=~)(Float64Backend.Mul_S_M(s, m), r)

[<Property>]
let ``OpenBLAS.64.Mul_Had_M_M``(m:float[,]) =
    let r = m |> Array2D.map (fun x -> x * x)
    Util.(=~)(Float64Backend.Mul_Had_M_M(m, m), r)

[<Property>]
let ``OpenBLAS.64.Transpose_M``(mm:float[,]) =
    let m = (Array2D.length1 mm)
    let n = (Array2D.length2 mm)
    let r = Array2D.zeroCreate n m
    for i = 0 to n - 1 do
        for j = 0 to m - 1 do
            r.[i, j] <- mm.[j, i]
    Util.(=~)(Float64Backend.Transpose_M(mm), r)

[<Property>]
let ``OpenBLAS.64.Map_F_M``(m:float[,]) =
    let f (x:float) = sin (exp x)
    let r = m |> Array2D.map f
    Util.(=~)(Float64Backend.Map_F_M(f, m), r)

[<Property>]
let ``OpenBLAS.64.Map2_F_M_M``(m:float[,]) =
    let f (x1:float) (x2:float) = sin (exp x1) + cos x2
    let r = Array2D.map2 f m m
    Util.(=~)(Float64Backend.Map2_F_M_M(f, m, m), r)

//
// Hard-coded tests
//

let m64_1 = array2D [[ 0.62406; 2.19092; 1.93734;-7.41726];
                     [ 0.66847; 7.18858; 9.21412; 1.83647];
                     [-9.13892; 3.36902; 4.14575; 3.64308]];
let m64_2 = array2D [[ 0.62406; 2.19092; 1.93734;-7.41726];
                     [ 0.66847; 7.18858; 9.21412; 1.83647];
                     [-9.13892; 3.36902; 4.14575; 3.64308];
                     [-3.38312;-3.78691;-3.85926;-0.00381]]
let m64_3 = array2D [[ 0.62406; 2.19092; 1.93734;-7.41726];
                     [ 2.19092; 7.18858; 9.21412; 1.83647];
                     [ 1.93734; 9.21412; 4.14575;-3.85926];
                     [-7.41726; 1.83647;-3.85926;-0.00381]]
let m64_4 = array2D [[ 9.24230; 51.73230; 58.05327; 6.48088]
                     [-85.19758; 77.22825; 98.64351; 41.80417]
                     [-53.66380; 4.36692; 16.46500; 89.06226]]
let m64_5 = array2D [[4.25136; 46.74136; 53.06233; 1.48994]
                     [-85.5446; 76.88123; 98.29649; 41.45715]
                     [-47.68089; 10.34983; 22.44791; 95.04517]
                     [24.47297; -53.78882; -63.5988; -2.08733]]
let m64_6 = array2D [[-0.03792; 0.02867;-0.09172;-0.04912]
                     [ 0.02512;-0.68486; 0.39513;-1.19806]
                     [ 0.00872; 0.64692;-0.30733; 0.95965]
                     [-0.12831;-0.03091; 0.02872;-0.10736]]
let m64_7 = array2D [[-4.99094;-0.34702]
                     [ 5.98291;-6.16668]]
let v64_1 =         [|-4.99094;-0.34702; 5.98291;-6.16668|]
let v64_2 =         [|53.45586; 37.97145; 46.78062|]
let v64_3 =         [|-368.78194; 547.68320; 647.37647; -156.33702|]
let v64_4 =         [|-0.06652; 9.86439;-8.02472; 1.48504|]
let v64_5 =         [| 2.04706; 1.31825;-1.70990; 0.78788|]
let s64_1 = 556.04485

let m32_1 = m64_1 |> Array2D.map float32
let m32_2 = m64_2 |> Array2D.map float32
let m32_3 = m64_3 |> Array2D.map float32
let m32_4 = m64_4 |> Array2D.map float32
let m32_5 = m64_5 |> Array2D.map float32
let m32_6 = m64_6 |> Array2D.map float32
let m32_7 = m64_7 |> Array2D.map float32
let v32_1 = v64_1 |> Array.map float32
let v32_2 = v64_2 |> Array.map float32
let v32_3 = v64_3 |> Array.map float32
let v32_4 = v64_4 |> Array.map float32
let v32_5 = v64_5 |> Array.map float32
let s32_1 = s64_1 |> float32

// float32
[<Property>]
let ``OpenBLAS.32.Mul_M_V``() =
    Util.(=~)(Float32Backend.Mul_M_V(m32_1, v32_1), v32_2)

[<Property>]
let ``OpenBLAS.32.Mul_V_M``() =
    Util.(=~)(Float32Backend.Mul_V_M(v32_2, m32_1), v32_3)

[<Property>]
let ``OpenBLAS.32.Solve_M_V``() =
    match Float32Backend.Solve_M_V(m32_2, v32_1) with
    | Some(s) -> Util.(=~)(s, v32_4)
    | _ -> false

[<Property>]
let ``OpenBLAS.32.SolveSymmetric_M_V``() =
    match Float32Backend.SolveSymmetric_M_V(m32_3, v32_1) with
    | Some(s) -> Util.(=~)(s, v32_5)
    | _ -> false

[<Property>]
let ``OpenBLAS.32.Mul_M_M``() =
    Util.(=~)(Float32Backend.Mul_M_M(m32_1, m32_2), m32_4)

[<Property>]
let ``OpenBLAS.32.Mul_M_M_Add_V_MCols``() =
    Util.(=~)(Float32Backend.Mul_M_M_Add_V_MCols(m32_2, m32_2, v32_1), m32_5)

[<Property>]
let ``OpenBLAS.32.Inverse_M``() =
    match Float32Backend.Inverse_M(m32_2) with
    | Some(s) -> Util.(=~)(s, m32_6)
    | _ -> false

[<Property>]
let ``OpenBLAS.32.Det_M``() =
    match Float32Backend.Det_M(m32_2) with
    | Some(s) -> Util.(=~)(s, s32_1)
    | _ -> false

[<Property>]
let ``OpenBLAS.32.ReshapeCopy_V_MRows``() =
    Util.(=~)(Float32Backend.ReshapeCopy_V_MRows(2, v32_1), m32_7)

[<Property>]
let ``OpenBLAS.32.RepeatReshapeCopy_V_MRows``() =
    let m = 2
    let n = v32_1.Length
    let r = Array2D.zeroCreate m n
    for i = 0 to m - 1 do
        for j = 0 to n - 1 do
            r.[i, j] <- v32_1.[j]
    Util.(=~)(Float32Backend.RepeatReshapeCopy_V_MRows(m, v32_1), r)

[<Property>]
let ``OpenBLAS.32.RepeatReshapeCopy_V_MCols``() =
    let m = v32_1.Length
    let n = 2
    let r = Array2D.zeroCreate m n
    for i = 0 to m - 1 do
        for j = 0 to n - 1 do
            r.[i, j] <- v32_1.[i]
    Util.(=~)(Float32Backend.RepeatReshapeCopy_V_MCols(n, v32_1), r)

// float64
[<Property>]
let ``OpenBLAS.64.Mul_M_V``() =
    Util.(=~)(Float64Backend.Mul_M_V(m64_1, v64_1), v64_2)

[<Property>]
let ``OpenBLAS.64.Mul_V_M``() =
    Util.(=~)(Float64Backend.Mul_V_M(v64_2, m64_1), v64_3)

[<Property>]
let ``OpenBLAS.64.Solve_M_V``() =
    match Float64Backend.Solve_M_V(m64_2, v64_1) with
    | Some(s) -> Util.(=~)(s, v64_4)
    | _ -> false

[<Property>]
let ``OpenBLAS.64.SolveSymmetric_M_V``() =
    match Float64Backend.SolveSymmetric_M_V(m64_3, v64_1) with
    | Some(s) -> Util.(=~)(s, v64_5)
    | _ -> false

[<Property>]
let ``OpenBLAS.64.Mul_M_M``() =
    Util.(=~)(Float64Backend.Mul_M_M(m64_1, m64_2), m64_4)

[<Property>]
let ``OpenBLAS.64.Mul_M_M_Add_V_MCols``() =
    Util.(=~)(Float64Backend.Mul_M_M_Add_V_MCols(m64_2, m64_2, v64_1), m64_5)

[<Property>]
let ``OpenBLAS.64.Inverse_M``() =
    match Float64Backend.Inverse_M(m64_2) with
    | Some(s) -> Util.(=~)(s, m64_6)
    | _ -> false

[<Property>]
let ``OpenBLAS.64.Det_M``() =
    match Float64Backend.Det_M(m64_2) with
    | Some(s) -> Util.(=~)(s, s64_1)
    | _ -> false

[<Property>]
let ``OpenBLAS.64.ReshapeCopy_V_MRows``() =
    Util.(=~)(Float64Backend.ReshapeCopy_V_MRows(2, v64_1), m64_7)

[<Property>]
let ``OpenBLAS.64.RepeatReshapeCopy_V_MRows``() =
    let m = 2
    let n = v64_1.Length
    let r = Array2D.zeroCreate m n
    for i = 0 to m - 1 do
        for j = 0 to n - 1 do
            r.[i, j] <- v64_1.[j]
    Util.(=~)(Float64Backend.RepeatReshapeCopy_V_MRows(m, v64_1), r)

[<Property>]
let ``OpenBLAS.64.RepeatReshapeCopy_V_MCols``() =
    let m = v64_1.Length
    let n = 2
    let r = Array2D.zeroCreate m n
    for i = 0 to m - 1 do
        for j = 0 to n - 1 do
            r.[i, j] <- v64_1.[i]
    Util.(=~)(Float64Backend.RepeatReshapeCopy_V_MCols(n, v64_1), r)

[<Test>]
let ``Smoke1``() =
    ()
