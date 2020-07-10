(*** hide ***)
#r "../../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"
open DiffSharp
let f (i:int) = 1.0
let x = dsharp.tensor [1]
let v = dsharp.tensor [1]
let n = 1

(**
API Overview
============

The following table gives an overview of the DiffSharp API.


*)

