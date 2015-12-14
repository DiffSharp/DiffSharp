#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


open DiffSharp.AD.Float32
open DiffSharp.Config

let v1 = [||]
let v2 = [|2.0|]

let r = Array2D.zeroCreate v1.Length v2.Length
for i = 0 to v1.Length - 1 do
    for j = 0 to v2.Length - 1 do
        r.[i, j] <- v1.[i] * v2.[j]
let r2 = GlobalConfig.Float64Backend.Mul_Out_V_V(v1, v2)

