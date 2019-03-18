(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Newton's Method
===============

In optimization, [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) from numerical analysis is used for finding the roots of the derivative of a function and thereby discovering its local extrema.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, starting from an initial point $\mathbf{x}_0$, the method works by computing succsessive points in the function domain

$$$
 \mathbf{x}_{n + 1} = \mathbf{x}_n - \eta \left(\mathbf{H}_f\right)_{\mathbf{x}_n}^{-1} \left( \nabla f \right)_{\mathbf{x}_n} \; ,

where $\eta > 0$ is the step size, $\left(\mathbf{H}_f\right)_{\mathbf{x}_n}^{-1}$ is the inverse of the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) of $f$ evaluated at $\mathbf{x}_n$, and $\left( \nabla f \right)_{\mathbf{x}_n}$ is the [gradient](https://en.wikipedia.org/wiki/Gradient) of $f$ evaluated at $\mathbf{x}_n$.

Newton's method converges faster than gradient descent, but this comes at the cost of computing the Hessian of the function at each iteration. In practice, the Hessian is usually only approximated from the changes in the gradient, giving rise to [quasi-Netwon methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) such as the [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).

Using DiffSharp, we can compute the exact Hessian efficiently via automatic differentiation. The following code implements Newton's method using the **DiffSharp.AD.Float64** module, which provides the **gradhessian** operation returning both the gradient and the Hessian of a function at a given point using forward-on-reverse AD.
*)

open DiffSharp.AD.Float64

// Newton's method
// f: function, x0: starting point, eta: step size, epsilon: threshold
let Newton f x0 (eta:D) epsilon =
    let rec desc x =
        let g, h = gradhessian f x
        if DV.l2norm g < epsilon then x else desc (x - eta * (DM.inverse h) * g)
    desc x0

(**

Let us find a (local) extremum of the function

$$$
 f(\mathbf{x}) = e^{x_1 - 1} + e^{-x_2 + 1} + (x_1 - x_2)^2

around the point $(0, 0)$.

*)

let f (x:DV) = (exp (x.[0] - 1)) + (exp (- x.[1] + 1)) + ((x.[0] - x.[1]) ** 2)

let xmin = Newton f (toDV [0.; 0.]) (D 1.) (D 0.001)
let fxmin = f xmin

(*** hide, define-output: o ***)
printf "val xmin : DV = DV [|0.7958861818; 1.203482609|]
val fxmin : D = D 1.797388803"
(*** include-output: o ***)

(**

The extremum is found as $f(0.7958861818, 1.203482609) = 1.797388803$.
   
*)
