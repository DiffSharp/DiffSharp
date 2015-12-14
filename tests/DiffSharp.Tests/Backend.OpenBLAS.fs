module DiffSharp.Tests.Backend.OpenBLAS

open FsCheck.NUnit
open DiffSharp.Config
open DiffSharp.Tests.Util


// float32
[<Property>]
let ``OpenBLAS.32.Mul_Dot_V_V``(v:float32[]) = 
    let r = v |> Array.map (fun x -> x * x) |> Array.sum
    (GlobalConfig.Float32Backend.Mul_Dot_V_V(v, v)) =~ r

[<Property>]
let ``OpenBLAS.32.L1Norm_V``(v:float32[]) = 
    let r = v |> Array.map abs |> Array.sum
    (GlobalConfig.Float32Backend.L1Norm_V(v)) =~ r

[<Property>]
let ``OpenBLAS.32.L2Norm_V``(v:float32[]) = 
    let r = v |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
    (GlobalConfig.Float32Backend.L2Norm_V(v)) =~ r

[<Property>]
let ``OpenBLAS.32.Sum_V``(v:float32[]) = 
    let r = v |> Array.sum
    (GlobalConfig.Float32Backend.Sum_V(v)) =~ r

[<Property>]
let ``OpenBLAS.32.Sum_M``(m:float32[,]) = 
    let mutable r = 0.f
    for i = 0 to (Array2D.length1 m) - 1 do
        for j = 0 to (Array2D.length2 m) - 1 do
            r <- r + m.[i, j]
    (GlobalConfig.Float32Backend.Sum_M(m)) =~ r

// float
[<Property>]
let ``OpenBLAS.64.Mul_Dot_V_V``(v:float[]) = 
    let r = v |> Array.map (fun x -> x * x) |> Array.sum
    (GlobalConfig.Float64Backend.Mul_Dot_V_V(v, v)) =~ r

[<Property>]
let ``OpenBLAS.64.L1Norm_V``(v:float[]) = 
    let r = v |> Array.map abs |> Array.sum
    (GlobalConfig.Float64Backend.L1Norm_V(v)) =~ r

[<Property>]
let ``OpenBLAS.64.L2Norm_V``(v:float[]) = 
    let r = v |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
    (GlobalConfig.Float64Backend.L2Norm_V(v)) =~ r

[<Property>]
let ``OpenBLAS.64.Sum_V``(v:float[]) = 
    let r = v |> Array.sum
    (GlobalConfig.Float64Backend.Sum_V(v)) =~ r

[<Property>]
let ``OpenBLAS.64.Sum_M``(m:float[,]) = 
    let mutable r = 0.
    for i = 0 to (Array2D.length1 m) - 1 do
        for j = 0 to (Array2D.length2 m) - 1 do
            r <- r + m.[i, j]
    (GlobalConfig.Float64Backend.Sum_M(m)) =~ r
