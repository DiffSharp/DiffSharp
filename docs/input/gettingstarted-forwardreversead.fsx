(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"

(**
Reverse-on-Forward AD
=====================

The library provides a reverse-on-forward mode AD implementation, which is a mixed method where a forward mode AD pass is followed by a reverse mode AD pass. This gives an efficient way of computing Hessians and [Hessian-vector products](http://en.wikipedia.org/wiki/Hessian_automatic_differentiation).

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.AD.ForwardReverse
---------------------------

This is a reverse-on-forward mode AD module, used with the **DualAdj** numeric type.
*)

open DiffSharp.AD.Specialized.Forward1Reverse

// f: DualAdj[] -> DualAdj
let f (x:DualAdj[]) = sin (x.[0] * x.[1])

// Gradient of f at (2, 3)
let gf = grad f [|2.; 3.|]

// Hessian of f at (2, 3)
let hf = hessian f [|2.; 3.|]

// Product of the Hessian of f at (2, 3) with the vector (1.2, 2.5)
// Computed in a matrix-free way and significantly faster 
// than computing the full Hessian and multiplying by the vector
let hfv = hessianv f [|2.; 3.|] [|1.2; 2.5|]

// Original value, gradient, and Hessian of f at (2, 3)
let oof, ggf, hhf = gradhessian' f [|2.; 3.|]