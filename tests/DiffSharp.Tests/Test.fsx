#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

open DiffSharp.Symbolic

let test = laplacian <@ fun x0 x1 -> (sin x0) * (cos (exp x1)) @> 

let t = test [|2.; 2.3|]