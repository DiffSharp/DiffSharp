
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

open DiffSharp.AD.Specialized.Forward1Reverse1

let r2, r3 = jacobianTv'' (fun x -> [|sin (x.[0] * x.[1]); cos (x.[1] * x.[2])|]) [|1.; 2.; 3.|]

let test = r3 [|5.; 6.|]