(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

(**
Newton's Method
===============

In optimization, [Newton's method](http://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) from numerical analysis is used for finding the roots of the derivative of a function and thereby discovering its local optima.

For a function $f(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}$, starting from an initial point $\mathbf{x}_0$, the method works by computing succsessive points in the function domain

$$$
 \mathbf{x}_{n + 1} = \mathbf{x}_n - \gamma \left(\mathbf{H}_f\right)_{\mathbf{x}_n}^{-1} \left( \nabla f \right)_{\mathbf{x}_n} \; ,

where $\gamma > 0$ is the step size, $\left(\mathbf{H}_f\right)_{\mathbf{x}_n}^{-1}$ is the inverse of the [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) of $f$ evaluated at $\mathbf{x}_n$, and $\left( \nabla f \right)_{\mathbf{x}_n}$ is the [gradient](http://en.wikipedia.org/wiki/Gradient) of $f$ evaluated at $\mathbf{x}_n$.

Newton's method converges faster than gradient descent, but this comes at the cost of computing the inverse of the Hessian of the function at each iteration. In practice, the Hessian is usually only approximated from the changes in the gradient, giving rise to [quasi-Netwon methods](http://en.wikipedia.org/wiki/Quasi-Newton_method) such as the [BFGS algorithm](http://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).

Using the DiffSharp library, we can compute the exact Hessian matrix via automatic differentiation. The following code implements Newton's method using the **DiffSharp.AD.ForwardGH** module, which provides the **gradhessian'** operation returning the value, gradient, and Hessian of a function at a given point using only one forward pass.
*)

open DiffSharp.AD.ForwardGH
open DiffSharp.AD.ForwardGH.Vector
open DiffSharp.Util.LinearAlgebra

// Newton's method, with function f, starting at x0, step size a, threshold t
let Newton f (x0:Vector<float>) (a:float) t =
    // Descending sequence of x, f(x)
    let dseq = Seq.unfold (fun x -> 
                            // Get value, gradient, hessian of f at x
                            let v, g, h = gradhessian' f x
                            if Vector.norm g < t then
                                None
                            else
                                let p = (Matrix.inverse h) * (-g)
                                let x' = x + a * p
                                Some((x, v), x'))
                        (x0)
    (Seq.last dseq, dseq)

(**

Let us find a (local) optimum of the function

$$$
 f(\mathbf{x}) = e^{x_1 - 1} + e^{-x_2 + 1} + (x_1 + x_2)^2

around the point $(0, 0)$.

*)

let (xopt, fxopt), dseq =
    Newton (fun x -> (exp (x.[0] - 1)) + (exp (- x.[1] + 1)) + ((x.[0] - x.[1]) ** 2)) 
           (vector [0.; 0.]) 1. 0.00001

let numsteps = Seq.length dseq

(*** hide, define-output: o ***)
printf "val xopt : Vector<float> = Vector [|0.7958861818; 1.203482609|]
val fxopt : float = 1.797388803
val dseq : seq<Vector<float> * float>
val numsteps : int = 4"
(*** include-output: o ***)

(**

The optimum is found in 4 iterations.
   
*)