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