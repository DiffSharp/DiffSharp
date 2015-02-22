(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
Newton's Method
===============

In optimization, [Newton's method](http://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) from numerical analysis is used for finding the roots of the derivative of a function and thereby discovering its local optima.

For a function $f(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}$, starting from an initial point $\mathbf{x}_0$, the method works by computing succsessive points in the function domain

$$$
 \mathbf{x}_{n + 1} = \mathbf{x}_n - \gamma \left(\mathbf{H}_f\right)_{\mathbf{x}_n}^{-1} \left( \nabla f \right)_{\mathbf{x}_n} \; ,

where $\gamma > 0$ is the step size, $\left(\mathbf{H}_f\right)_{\mathbf{x}_n}^{-1}$ is the inverse of the [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) of $f$ evaluated at $\mathbf{x}_n$, and $\left( \nabla f \right)_{\mathbf{x}_n}$ is the [gradient](http://en.wikipedia.org/wiki/Gradient) of $f$ evaluated at $\mathbf{x}_n$.

Newton's method converges faster than gradient descent, but this comes at the cost of computing the Hessian of the function at each iteration. In practice, the Hessian is usually only approximated from the changes in the gradient, giving rise to [quasi-Netwon methods](http://en.wikipedia.org/wiki/Quasi-Newton_method) such as the [BFGS algorithm](http://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).

Using the DiffSharp library, we can compute the exact Hessian via automatic differentiation. The following code implements Newton's method using the **DiffSharp.AD.ForwardGH** module, which provides the **gradhessian'** operation returning the value, the gradient, and the Hessian of a function at a given point using only one forward evaluation.
*)

open DiffSharp.AD.ForwardGH
open DiffSharp.AD.ForwardGH.Vector
open DiffSharp.Util.LinearAlgebra


// Newton's method, with function f, starting point x0, step size a, threshold t
let Newton f x0 (a:float) t =
    let rec desc x =
        let g, h = gradhessian f x
        if Vector.norm g < t then x else desc (x - a * (Matrix.inverse h) * g)
    desc x0


// Newton's method, with function f, starting point x0, step size a, threshold t
// Returns a descending sequence of pairs (x, f(x))
let NewtonSeq f (x0:Vector<float>) (a:float) t =
    Seq.unfold (fun x -> 
                    // Get value, gradient, hessian of f at x
                    let v, g, h = gradhessian' f x
                    if Vector.l2normSq g < t then
                        None
                    else
                        let p = (Matrix.inverse h) * (-g)
                        let x' = x + a * p
                        Some((x, v), x'))
                (x0)

(**

Let us find a (local) extremum of the function

$$$
 f(\mathbf{x}) = e^{x_1 - 1} + e^{-x_2 + 1} + (x_1 + x_2)^2

around the point $(0, 0)$.

*)

let dseq =
    NewtonSeq (fun x -> (exp (x.[0] - 1)) + (exp (- x.[1] + 1)) + ((x.[0] - x.[1]) ** 2)) 
           (vector [0.; 0.]) 1. 0.0001

let xext, fxext = Seq.last dseq
let numsteps = Seq.length dseq

(**

The extremum is found as $f(0.7749209799, 1.162864526) = 1.798659611 in 3 iterations.
   
*)

(*** hide, define-output: o ***)
printf "val xext : Vector<float> = Vector [|0.7749209799; 1.162864526|]
val fxext : float = 1.798659611
val numsteps : int = 3"
(*** include-output: o ***)
