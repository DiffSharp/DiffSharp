(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
Gradient Descent
================

The [gradient descent algorithm](http://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for finding a local minimum of a function near a starting point, taking successive steps in the direction of the negative of the gradient.

For a function $f(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}$, starting from an initial point $\mathbf{x}_0$, the method works by computing succsessive points in the function domain

$$$
 \mathbf{x}_{n + 1} = \mathbf{x}_n - \gamma \left( \nabla f \right)_{\mathbf{x}_n} \; ,

where $\gamma > 0$ is a small step size and $\left( \nabla f \right)_{\mathbf{x}_n}$ is the [gradient](http://en.wikipedia.org/wiki/Gradient) of $f$ evaluated at $\mathbf{x}_n$. The successive values of the function 

$$$
 f(\mathbf{x}_0) \ge f(\mathbf{x}_1) \ge f(\mathbf{x}_2) \ge \dots
 
keep decreasing and the sequence $\mathbf{x}_n$ usually converges to a local minimum.

Generally speaking, using a fixed step size $\gamma$ yields suboptimal performance and there are adaptive variations of the gradient descent algorithm that select a locally optimal step size $\gamma$ on every iteration.

Using the DiffSharp library, the following code implements gradient descent with a fixed step size, stopping when the [norm](http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of the gradient falls below a given threshold.

*)

open DiffSharp.AD.Forward
open DiffSharp.AD.Forward.Vector
open DiffSharp.Util.LinearAlgebra

// Gradient descent, with function f, starting point x0, step size a, threshold t
let gd f x0 (a:float) t =
    let rec desc x =
        let g = grad f x
        if Vector.norm g < t then x else desc (x - a * g)
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

The following code is an alternative that returns the sequence of points that are visited during the descent, so that we can plot them.
*)

// Gradient descent, with function f, starting point x0, step size a, threshold t
// Returns a descending sequence of pairs (x, f(x))
let gdSeq f x0 (a:float) t =
    Seq.unfold (fun x -> 
                    // Get value, gradient of f at x
                    let v, g = grad' f x
                    if Vector.norm g < t then 
                        None 
                    else 
                        let x' = x - a * g
                        Some((x, v), x'))
                (x0)

(**
We can draw some contours of $ \sin x + \cos y $ and show the trajectory of gradient descent, using the [F# Charting](http://fsharp.github.io/FSharp.Charting/index.html) library.
*)

open FSharp.Charting

let dseq = gdSeq f (vector [1.; 1.]) 0.9 0.00001

// Draw the contour line Sin(x) + Cos(y) = v
let c v = Chart.Line(List.append [for x in -1.6..0.01..1.4->(x,acos(v-sin x))] 
                                 [for y in 3.0..0.01..4.->(asin(v-cos y),y)], 
                     Color = System.Drawing.Color.Tomato)

// Draw some contours and combine with the graph of the descent xseq
Chart.Combine(List.append [for x in -2.2..0.4..2. -> c x]
                          [Chart.Line(Seq.map (fun (x:Vector<float>, _)->(x.[0],x.[1])) dseq)]
             ).WithXAxis(Min = -1.6, Max = 1.4, Title="x")
             .WithYAxis(Min = 0., Max = 4., Title="y")

(**
This gives us the following chart.

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-gradientdescent-chart.png" alt="Chart" style="width:550px"/>
    </div>
</div>
*)