(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Numerical Differentiation
=========================

In addition to AD, DiffSharp also implements [numerical differentiation](http://en.wikipedia.org/wiki/Numerical_differentiation).

Numerical differentiation is based on finite difference approximations of derivative values, using values of the original function at some sample points. Unlike AD, numerical differentiation gives only approximate results and has problems caused by truncation and roundoff errors.

For a complete list of available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.Numerical
-------------------

This is a numerical differentiation module, used with the regular **float** numeric type.
*)

open DiffSharp.Numerical

// f: float -> float
let f x = sin (3. * sqrt x)

// Derivative of f at 2
let df = diff f 2.

// g: float[] -> float
let g (x:float[]) = sin (x.[0] * x.[1])

// Gradient of g at (2, 3)
let gg = grad g [|2.; 3.|]

// h: float[] -> float[]
let h (x:float[]) = [| sin x.[0]; cos x.[1] |]

// Jacobian of h at (2, 3)
let jh = jacobian h [|2.; 3.|]
