(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Gradient Descent
================

The [gradient descent algorithm](http://en.wikipedia.org/wiki/Gradient_descent) is an optimization algorithm for finding a local minimum of a function near a starting point, taking successive steps in the direction of the negative of the gradient. 

The following code implements gradient descent with a fixed step size, stopping when the norm of the gradient falls below a given threshold.

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

$$$
\underset{x,\;y}{\operatorname{argmin}} (\sin x + \cos y)

*)

let test = argmin (fun x -> (sin x.[0]) + cos x.[1]) (vector [|1.; 1.|]) 0.001 0.000001