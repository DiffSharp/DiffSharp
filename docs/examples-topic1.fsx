(*** condition: prepare ***)
#r "../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget:RestoreSources=https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#i "nuget: https://ci.appveyor.com/nuget/diffsharp"
#r "nuget: DiffSharp-lite,{{package-version}}"

Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB


(**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dsyme/DiffSharp/gh-pages?filepath=examples-topic1.ipynb)

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

open DiffSharp

// Shorthand for dsharp.tensor
let t x = dsharp.tensor x

// Gradient descent
//   f: function
//   x0: starting point
//   eta: step size
//   epsilon: threshold
let rec gradientDescent f x eta epsilon =
    let g = dsharp.grad f x
    // TODO: this should be norm
    if g.sum() < epsilon then x 
    else gradientDescent f (x - eta * g) eta epsilon

(**
Let's find a minimum of $f(x, y) = (\sin x + \cos y)$.
*)

let inline f (x:Tensor) =  sin x.[0] + cos x.[1]

// Find the minimum of f
// Start from (1, 1), step size 0.9, threshold 0.00001
let xmin = gradientDescent f (t [1.; 1.]) (t 0.9) (t 0.00001)
let fxmin = f xmin

(*** hide, define-output: o ***)
printf "val xmin : Tensor = tensor [ -1.570790759; 3.141591964 ]
val fxmin : Tensor = tensor -2.0"
(*** include-output: o ***)

(**

A minimum, $f(x, y) = -2$, is found at $(x, y) = \left(-\frac{\pi}{2}, \pi\right)$.

*)