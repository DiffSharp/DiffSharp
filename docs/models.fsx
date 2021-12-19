(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net6.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.Torch.dll"
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
DiffSharp.dsharp.config(backend=DiffSharp.Backend.Reference)
DiffSharp.dsharp.seed(123)

(**
Test 
*)

open DiffSharp

dsharp.config(backend=Backend.Reference)

let a = dsharp.tensor([1,2,3])
printfn "%A" a
(*** include-fsi-output ***)