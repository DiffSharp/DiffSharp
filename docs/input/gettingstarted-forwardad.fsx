(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Forward AD
==========

The DiffSharp library provides several implementations of forward AD, with distinct advantages under different applications. Here we show several toy examples to get you started using the library.

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.AD.Forward
--------------------
  
This is a forward AD module implemented using [dual numbers](http://en.wikipedia.org/wiki/Dual_number) of primal and tangent values. It provides a performance advantage when computing derivatives of scalar-to-scalar functions, or vector-to-vector functions $f: \mathbb{R}^n \to \mathbb{R}^m$ where $n \ll m$.

This module is used with the **Dual** numeric type.
*)

open DiffSharp.AD.Forward

// f: Dual -> Dual
let f (x:Dual) = sin (3. * sqrt x)

// Derivative of f at 2
let df = diff f 2.

// g: Dual[] -> Dual
let g (x:Dual[]) = sin (x.[0] * x.[1])

// Directional derivative of g at (2, 3) with direction (4, 1)
let ddg = gradv g [|2.; 3.|] [|4.; 1.|]

// Gradient of g at (2, 3)
let gg = grad g [|2.; 3.|]

// h: Dual[] -> Dual[]
let h (x:Dual[]) = [| sin x.[0]; cos x.[1] |]

// Jacobian of h at (2, 3)
let jh = jacobian h [|2.; 3.|]

(**
DiffSharp.AD.Forward2
---------------------

This is a forward AD module that also keeps the tangent-of-tangent values to compute 2nd derivatives.

This module is used with the **Dual2** numeric type.
*)

open DiffSharp.AD.Forward2

// f2: Dual2 -> Dual2
let f2 (x:Dual2) = sin (3. * sqrt x)

// 2nd derivative of f2 at 2
let d2f2 = diff2 f2 2.

// g2: Dual2[] -> Dual2
let g2 (x:Dual2[]) = sin (x.[0] * x.[1])

// Laplacian of g2 at (2, 3)
let lg2 = laplacian g2 [|2.; 3.|]

(**
DiffSharp.AD.ForwardG
---------------------

This is a forward AD module using a vector of gradient components for speeding up gradient calculations. It provides a performance advantage when computing gradients of vector-to-scalar functions.

This module is used with the **DualG** numeric type.
*)

open DiffSharp.AD.ForwardG

// g3: DualG[] -> DualG
let g3 (x:DualG[]) = sin (x.[0] * x.[1])

// Gradient of g3 at (2, 3)
let gg3 = grad g3 [|2.; 3.|]

// h3: DualG[] -> DualG[]
let h3 (x:DualG[]) = [| sin x.[0]; cos x.[1] |]

// Jacobian of h3 at (2, 3)
let jh3 = jacobian h3 [|2.; 3.|]

(**
DiffSharp.AD.ForwardGH
----------------------

This is a forward AD module using a vector of gradient components and a matrix of Hessian components, for speeding up gradient and Hessian calculations. It provides exact Hessians.

This module is used with the **DualGH** numeric type.
*)

open DiffSharp.AD.ForwardGH

// g4: DualGH[] -> DualGH
let g4 (x:DualGH[]) = sin (x.[0] * x.[1])

// Gradient and Hessian of g4 at (2, 3)
let gg4, hg4 = gradhessian g4 [|2.; 3.|]

(**
DiffSharp.AD.ForwardN
---------------------

This is a forward AD module lazily evaluating higher-order derivatives as they are called. It provides higher order derivatives of scalar-to-scalar functions.

This module is used with the **DualN** numeric type.
*)

open DiffSharp.AD.ForwardN

// f5: DualN -> DualN
let f5 (x:DualN) = sin (3. * sqrt x)

// 3rd derivative of f5 at 2.
let d3f5 = diffn 3 f5 2.
