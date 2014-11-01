(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Numerical Differentiation
=========================

In addition to AD, the DiffSharp library also implements [numerical differentiation](http://en.wikipedia.org/wiki/Numerical_differentiation).

Numerical differentiation is based on finite difference approximations of derivative values, using values of the original function at some sample points. Unlike AD, numerical differentiation gives only approximate results and has problems caused by truncation and roundoff errors.

Currently the library uses the 1st order central difference

$$$
  \frac{df(x)}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}

for the **diff** and **diffdir** operations; the 2nd order central difference

$$$
  \frac{d^2 f(x)}{dx^2} \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}

for the **diff2** operation; and the 1st order forward difference

$$$
  \frac{df(x)}{dx} \approx \frac{f(x + h) - f(x)}{h}

for the **grad**, **hessian**, **laplacian**, and **jacobian** operations, where $ 0 < h \ll 1 $. The default step size is taken as $h = 10^{-5}$ and in future releases this will be replaced by adaptive methods for choosing the optimal step size for each operation.

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
