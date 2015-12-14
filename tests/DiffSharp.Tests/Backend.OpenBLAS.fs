module DiffSharp.Tests.Backend.OpenBLAS

open FsCheck.NUnit
open DiffSharp.Backend
open DiffSharp.Tests

let Float32Backend = DiffSharp.Backend.OpenBLAS.Float32Backend() :> Backend<float32>
let Float64Backend = DiffSharp.Backend.OpenBLAS.Float64Backend() :> Backend<float>

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


