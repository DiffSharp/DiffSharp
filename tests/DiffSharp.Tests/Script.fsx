#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


open DiffSharp.AD.Float32
open DiffSharp.Config

let v = [|3.40282347e+38f|]
let r = v |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
GlobalConfig.Float32Backend.L2Norm_V(v)

let a1 = array2D [[3.f; 2.f; -1.f]; [2.f; -2.f; 4.f]; [-1.f; 0.5f; -1.f]]
let b1 = [|1.f; -2.f; 0.f|]
let x1 = GlobalConfig.Float32Backend.Solve_M_V(a1, b1)


let a2 = array2D [[1.f; 22.f]; [20.f; 1.f]]
let b2 = [|6.f; 15.f|]
let x2 = GlobalConfig.Float32Backend.Solve_M_V(a2, b2)

let a3 = array2D [[1.f; 20.f]; [20.f; 1.f]]
let b3 = [|6.f; 15.f|]
let x3 = GlobalConfig.Float32Backend.SolveSymmetric_M_V(a3, b3)

let aa = array2D [[1.f; 2.f; 3.f]; [4.f; 5.f; 6.f]; [7.f; 8.f; 0.f]]
DiffSharp.Backend.OpenBLAS.BLASExtensions.simatcopyT(1.f,aa) |> ignore

let c = GlobalConfig.Float32Backend.Inverse_M(aa)

