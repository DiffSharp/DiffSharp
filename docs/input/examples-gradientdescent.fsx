(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Gradient Descent
================

The [gradient descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for finding a local minimum of a scalar-valued function near a starting point, taking successive steps in the direction of the negative of the gradient.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, starting from an initial point $\mathbf{x}_0$, the method works by computing successive points in the function domain

$$$
 \mathbf{x}_{n + 1} = \mathbf{x}_n - \eta \left( \nabla f \right)_{\mathbf{x}_n} \; ,

where $\eta > 0$ is a small step size and $\left( \nabla f \right)_{\mathbf{x}_n}$ is the [gradient](https://en.wikipedia.org/wiki/Gradient) of $f$ evaluated at $\mathbf{x}_n$. The successive values of the function 

$$$
 f(\mathbf{x}_0) \ge f(\mathbf{x}_1) \ge f(\mathbf{x}_2) \ge \dots
 
keep decreasing and the sequence $\mathbf{x}_n$ usually converges to a local minimum.

In practice, using a fixed step size $\eta$ yields suboptimal performance and there are adaptive algorithms that select a locally optimal step size $\eta$ on each iteration.

The following code implements gradient descent with fixed step size, stopping when the [norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of the gradient falls below a given threshold.

*)

open DiffSharp.AD.Float64

// Gradient descent
// f: function, x0: starting point, eta: step size, epsilon: threshold
let gd f x0 eta epsilon =
    let rec desc x =
        let g = grad f x
        if DV.l2norm g < epsilon then x else desc (x - eta * g)
    desc x0

(**
Let's find a minimum of $f(x, y) = (\sin x + \cos y)$.
*)

let inline f (x:DV) =  sin x.[0] + cos x.[1]

// Find the minimum of f
// Start from (1, 1), step size 0.9, threshold 0.00001
let xmin = gd f (toDV [1.; 1.]) (D 0.9) (D 0.00001)
let fxmin = f xmin

(*** hide, define-output: o ***)
printf "val xmin : DV = DV [|-1.570790759; 3.141591964|]
val fxmin : D = D -2.0"
(*** include-output: o ***)

(**

A minimum, $f(x, y) = -2$, is found at $(x, y) = \left(-\frac{\pi}{2}, \pi\right)$.

*)
