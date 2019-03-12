(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Numerical Differentiation
=========================

In addition to AD, DiffSharp also implements [numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation).

Numerical differentiation is based on finite difference approximations of derivative values, using values of the original function evaluated at some sample points. Unlike AD, numerical differentiation gives only approximate results and is unstable due to truncation and roundoff errors.

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.Numerical
-------------------

This is a numerical differentiation module, used with the regular **float** or **float32** numeric types for scalars, **float[]** or **float32[]** for vectors, and **float[,]** or **float[,]** for matrices.

Currently the library uses the 1st order central difference

$$$
  \frac{df(x)}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}

for the **diff** and **diffdir** operations; the 2nd order central difference

$$$
  \frac{d^2 f(x)}{dx^2} \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}

for the **diff2** operation; and the 1st order forward difference

$$$
  \frac{df(x)}{dx} \approx \frac{f(x + h) - f(x)}{h}

for the **grad**, **hessian**, **laplacian**, and **jacobian** operations, where $ 0 < h \ll 1 $. 
*)

open DiffSharp.Numerical.Float64

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

(**
The default step size is $h = 10^{-5}$ and it can be changed by using the method **DiffSharp.Config.GlobalConfig.SetEpsilon**.

Adaptive step size techniques are planned to be implemented in a future release.
*)

let v1 = diff sin 0.2

DiffSharp.Config.GlobalConfig.SetEpsilon(0.001)

let v2 = diff sin 0.2

(*** hide, define-output: o ***)
printf "val v1 : float = 0.9800665778
val v2 : float = 0.9800664145"
(*** include-output: o ***)
