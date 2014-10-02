(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

(**
Gradient Descent
================

The [gradient descent algorithm](http://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for finding a local minimum of a function near a starting point, taking successive steps in the direction of the negative of the gradient. 

Using the DiffSharp library, the following code implements gradient descent with a fixed step size, stopping when the norm of the gradient falls below a given threshold.

*)

open DiffSharp.AD.Forward.Vector
open DiffSharp.Util.LinearAlgebra

// Gradient descent, with function f, starting at x0, step size a, threshold t
let argmin f x0 a t =
    // Descending sequence, with state s (current x, gradient of f at current x)
    let gd = Seq.unfold (fun s -> 
                         if norm (snd s) < t then None 
                         else Some(fst s, (fst s - a * snd s, grad f (fst s))))
                        (x0, grad f x0)
    (Seq.last gd, gd)

(**
Let us compute $\; \underset{x,\;y}{\operatorname{argmin}} (\sin x + \cos y) $.
*)

// Find the minimum of Sin(x) + Cos(y)
// Start from (1, 1), step size 0.01, threshold 0.00001
let xmin, xseq =
    argmin (fun x -> (sin x.[0]) + cos x.[1]) (vector [|1.; 1.|]) 0.01 0.00001

(** *)

// The local minimum is found at (-1.570787495, 3.141587987)
// val xmin : Vector = Vector [|-1.570787495; 3.141587987|]
// val xseq : seq<Vector>

(**
We can draw some contours of $ \sin x + \cos y $ and show the trajectory of gradient descent, using the [F# Charting](http://fsharp.github.io/FSharp.Charting/index.html) library.
*)

open FSharp.Charting

// Draw the contour line Sin(x) + Cos(y) = v
let c v = Chart.Line(List.append [for x in -1.6..0.01..1.4->(x,acos(v-sin x))] 
                                 [for y in 3.0..0.01..4.->(asin(v-cos y),y)], 
                     Color = System.Drawing.Color.Tomato)

// Draw some contours and combine with the graph of the descent xseq
Chart.Combine(List.append [for x in -2.2..0.4..2. -> c x]
                          [Chart.Line(Seq.map (fun (x:Vector)->(x.[0],x.[1])) xseq)]
             ).WithXAxis(Min = -1.6, Max = 1.4).WithYAxis(Min = 0., Max = 4.)

(**
This results in the following chart.

<div class="row">
    <div class="span6">
        <img src="img/examples-gradientdescent-chart.png" alt="Chart" style="width:500px"/>
    </div>
</div>
*)
