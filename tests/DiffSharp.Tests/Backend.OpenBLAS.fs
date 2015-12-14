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

module DiffSharp.Tests.Backend.OpenBLAS

open FsCheck.NUnit
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
let v64_1 =         [|-4.99094;-0.34702; 5.98291;-6.16668|]
let v64_2 =         [|53.45586; 37.97145; 46.78062|]
let v64_3 =         [|-368.78194; 547.68320; 647.37647; -156.33702|]
let v64_4 =         [|-0.06652; 9.86439;-8.02472; 1.48504|]
let v64_5 =         [| 2.04706; 1.31825;-1.70990; 0.78788|]

let m32_1 = m64_1 |> Array2D.map float32
let m32_2 = m64_2 |> Array2D.map float32
let m32_3 = m64_3 |> Array2D.map float32
let v32_1 = v64_1 |> Array.map float32
let v32_2 = v64_2 |> Array.map float32
let v32_3 = v64_3 |> Array.map float32
let v32_4 = v64_4 |> Array.map float32
let v32_5 = v64_5 |> Array.map float32

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