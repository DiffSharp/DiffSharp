
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

open DiffSharp.AD

let test = diff (fun x -> x * diff (fun y -> x * y) (D 3.)) (D 2.)