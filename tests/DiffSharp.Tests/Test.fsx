#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"


open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector

let sigm (x:Adj) = 1. / (1. + exp -x)

let neuron (x:Vector<Adj>) (w:Vector<Adj>) = Vector.sum x


let test = [| 2. |]