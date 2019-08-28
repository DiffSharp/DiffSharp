#r "../../src/DiffSharp/bin/Debug/netstandard2.0/DiffSharp.dll"

open DiffSharp.AD.Float32
open DiffSharp.Config
open DiffSharp.Util

let v64_1 =         [|-4.99094;-0.34702; 5.98291;-6.16668|]
let v32_1 = v64_1 |> Array.map float32

let m = 2
let n = v32_1.Length
let r = Array2D.zeroCreate<float32> m n
for i = 0 to m - 1 do
    for j = 0 to n - 1 do
        r.[i, j] <- v32_1.[j]
let r2 = GlobalConfig.Float32Backend.RepeatReshapeCopy_V_MRows(m, v32_1)

