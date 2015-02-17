#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


let test = DiffSharp.AD.Forward2.Forward2Ops.diff (fun x -> (sin x) * (cos (exp x))) 42.5
let test2 = DiffSharp.Symbolic.SymbolicOps.diff <@ fun x -> (sin x) * (cos (exp x)) @> 42.5

let tt = test - test2