
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

let fss_Adj = fun (x:DiffSharp.AD.Reverse.Adj) -> (sin x) * (cos (exp x))

let qss =  <@ fun x -> (sin x) * (cos (exp x)) @>

let tr = DiffSharp.AD.Reverse.ReverseOps.diff fss_Adj 72.

let ts = DiffSharp.Symbolic.SymbolicOps.diff qss 72.