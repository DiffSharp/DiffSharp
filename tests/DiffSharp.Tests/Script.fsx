#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


open DiffSharp.AD.Float32
open DiffSharp.Config

let s = nan
let v = [|13.|]

let r = 
    if (s = 0.) || (System.Double.IsNaN(s)) then
        Array.zeroCreate v.Length
    else
        v |> Array.map (fun x -> s * x)
GlobalConfig.Float64Backend.Mul_S_V(s, v)

