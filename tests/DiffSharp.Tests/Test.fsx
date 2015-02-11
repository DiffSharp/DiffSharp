#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"


open DiffSharp.Symbolic

let test = grad <@ fun (x:float) y -> x * y @>
let test2 = test [|2.; 1.|]