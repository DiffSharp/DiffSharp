#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


open DiffSharp.AD.Float32
open DiffSharp.Config

let test() = 
    let m = array2D [[2.f;0.2f];[1.f;0.f]]
    let v = [|1.f; 1.f|]
    match GlobalConfig.Float32Backend.Solve_M_V(m, v) with
    | Some(s) -> true
    | _ -> false

let t = test()