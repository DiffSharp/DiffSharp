(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


(**
Non-nested AD
=============

The DiffSharp library provides several non-nested implementations of forward and reverse AD, for situations where it is known beforehand that nesting will not be needed. This can give better performance for some specific non-nested tasks.

The non-nested AD modules are provided under the **DiffSharp.AD.Specialized** namespace.

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

Forward Mode
------------

### DiffSharp.AD.Specialized.Forward1
  
This is a forward mode AD module implemented using [dual numbers](http://en.wikipedia.org/wiki/Dual_number) of primal and tangent values. It provides a performance advantage when computing derivatives of scalar-to-scalar functions, or vector-to-vector functions $f: \mathbb{R}^n \to \mathbb{R}^m$ where $n \ll m$.

*)

open DiffSharp.AD.Specialized.Forward1

// f: D -> D
let f (x:D) = sin (3. * sqrt x)

// Derivative of f at 2
let df = diff f 2.

// g: D[] -> D
let g (x:D[]) = sin (x.[0] * x.[1])

// Directional derivative of g at (2, 3) with direction (4, 1)
let ddg = gradv g [|2.; 3.|] [|4.; 1.|]

// Gradient of g at (2, 3)
let gg = grad g [|2.; 3.|]

// h: D[] -> D[]
let h (x:D[]) = [| sin x.[0]; cos x.[1] |]

// Jacobian of h at (2, 3)
let jh = jacobian h [|2.; 3.|]

(**
### DiffSharp.AD.Specialized.Forward2

This is a forward mode AD module that also keeps the tangent-of-tangent values to compute 2nd derivatives.

*)

open DiffSharp.AD.Specialized.Forward2

// f2: D -> D
let f2 (x:D) = sin (3. * sqrt x)

// 2nd derivative of f2 at 2
let d2f2 = diff2 f2 2.

// g2: D[] -> D
let g2 (x:D[]) = sin (x.[0] * x.[1])

// Laplacian of g2 at (2, 3)
let lg2 = laplacian g2 [|2.; 3.|]

(**
### DiffSharp.AD.Specialized.ForwardG

This is a forward mode AD module using a vector of gradient components for speeding up gradient calculations. It provides a performance advantage when computing gradients of vector-to-scalar functions.

*)

open DiffSharp.AD.Specialized.ForwardG

// g3: D[] -> D
let g3 (x:D[]) = sin (x.[0] * x.[1])

// Gradient of g3 at (2, 3)
let gg3 = grad g3 [|2.; 3.|]

// h3: D[] -> D[]
let h3 (x:D[]) = [| sin x.[0]; cos x.[1] |]

// Jacobian of h3 at (2, 3)
let jh3 = jacobian h3 [|2.; 3.|]

(**
### DiffSharp.AD.Specialized.ForwardGH

This is a forward mode AD module using a vector of gradient components and a matrix of Hessian components, for speeding up gradient and Hessian calculations. It provides exact Hessians.

*)

open DiffSharp.AD.Specialized.ForwardGH

// g4: D[] -> D
let g4 (x:D[]) = sin (x.[0] * x.[1])

// Gradient and Hessian of g4 at (2, 3)
let gg4, hg4 = gradhessian g4 [|2.; 3.|]

(**
### DiffSharp.AD.Specialized.ForwardN

This is a forward mode AD module lazily evaluating higher-order derivatives as they are called. It provides higher order derivatives of scalar-to-scalar functions.

*)

open DiffSharp.AD.Specialized.ForwardN

// f5: D -> D
let f5 (x:D) = sin (3. * sqrt x)

// 3rd derivative of f5 at 2.
let d3f5 = diffn 3 f5 2.

(**
Reverse Mode
------------

### DiffSharp.AD.Specialized.Reverse1

This is a reverse mode AD module that works by recording a trace of operations in the forward evaluation of a function and using this in the reverse sweep for backpropagating adjoints.

Reverse mode AD provides a performance advantage when computing gradients of vector-to-scalar functions $f: \mathbb{R}^n \to \mathbb{R}$, because it can calculate all partial derivatives in just one forward evaluation and one reverse evaluation of the function, regardless the value of $n$.

*)

open DiffSharp.AD.Specialized.Reverse1

// f6: D[] -> D
let f6 (x:D[]) = sin (x.[0] * x.[1] * x.[2])

// Gradient of f
let g6 = grad f6

(**
Alternatively, we can use a lambda expression to alleviate the need to explicitly state the **D[]** type, as it will be automatically inferred.
*)

// The same gradient
let g7 = grad (fun x -> sin (x.[0] * x.[1] * x.[2]))

