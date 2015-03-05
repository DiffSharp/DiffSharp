(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Gradient Descent
================

The [gradient descent algorithm](http://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for finding a local minimum of a function near a starting point, taking successive steps in the direction of the negative of the gradient.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, starting from an initial point $\mathbf{x}_0$, the method works by computing succsessive points in the function domain

$$$
 \mathbf{x}_{n + 1} = \mathbf{x}_n - \eta \left( \nabla f \right)_{\mathbf{x}_n} \; ,

where $\eta > 0$ is a small step size and $\left( \nabla f \right)_{\mathbf{x}_n}$ is the [gradient](http://en.wikipedia.org/wiki/Gradient) of $f$ evaluated at $\mathbf{x}_n$. The successive values of the function 

$$$
 f(\mathbf{x}_0) \ge f(\mathbf{x}_1) \ge f(\mathbf{x}_2) \ge \dots
 
keep decreasing and the sequence $\mathbf{x}_n$ usually converges to a local minimum.

Generally speaking, using a fixed step size $\eta$ yields suboptimal performance and there are adaptive variations of the gradient descent algorithm that select a locally optimal step size $\eta$ on every iteration.

Using the DiffSharp library, the following code implements gradient descent with a fixed step size, stopping when the squared [norm](http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of the gradient falls below a given threshold.

*)

open DiffSharp.AD.Forward
open DiffSharp.AD.Forward.Vector
open DiffSharp.Util.LinearAlgebra

// Gradient descent
// f: function, x0: starting point, eta: step size, epsilon: threshold
let gd f x0 (eta:float) epsilon =
    let rec desc x =
        let g = grad f x
        if Vector.normSq g < epsilon then x else desc (x - eta * g)
    desc x0

(**
Let us find a minimum of $f(x, y) = (\sin x + \cos y)$.
*)

let inline f (x:Vector<_>) =  sin x.[0] + cos x.[1]

// Find the minimum of f
// Start from (1, 1), step size 0.9, threshold 0.00001
let xmin = gd f (vector [1.; 1.]) 0.9 0.00001
let fxmin = f xmin

(*** hide, define-output: o ***)
printf "val xmin : Vector<float> = Vector [|-1.570787572; 3.141587977|]
val fxmin : float = -2.0"
(*** include-output: o ***)

(**

A minimum, $f(x, y) = -2$, is found at $(x, y) = (-1.570787572, 3.141587977)$.

*)
