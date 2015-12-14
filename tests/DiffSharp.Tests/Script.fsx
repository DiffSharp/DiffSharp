#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Float32
open DiffSharp.Config
open DiffSharp.Util

let m = array2D [[]]
let s = 0.f

let r = 
    if m.Length = 0 then
        Array2D.empty
    else
        m |> Array2D.map (fun x -> x - s)
let r2 = GlobalConfig.Float32Backend.Sub_M_S(m, s)