
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

open DiffSharp.AD.Forward

let test = curldiv' (fun x -> [|sin (x.[0] + x.[1] + x.[2]); cos (x.[0] * x.[1] * x.[2]); exp (x.[0] / x.[1] - x.[2])|]) [|1.; 2.; 3.|]