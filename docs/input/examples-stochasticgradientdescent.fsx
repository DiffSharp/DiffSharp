(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
Stochastic Gradient Descent
===========================

*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra

let rnd = new System.Random()

// Stochastic gradient descent
// f: function, w0: starting weights, eta: step size, epsilon: threshold, t: training set
let sgd f w0 (eta:float) epsilon (t:(Vector<float>*Vector<float>)[]) =
    let ta = Array.map (fun (x, y) -> Vector.map adj x, Vector.map adj y) t
    let rec desc w =
        let x, y = ta.[rnd.Next(ta.Length)]
        let g = grad (fun wi -> Vector.normSq (y - (f wi x))) w
        if Vector.normSq g < epsilon then w else desc (w - eta * g)
    desc w0


let inline f (w:Vector<_>) (x:Vector<_>) =
    w.[0] + w.[1] * x.[0] + w.[2] * x.[0] * x.[0]


let points = [|0.5, 2.
               3.2, 1.
               5.2, 4.|]

let train = Array.map (fun (x, y) -> (vector [x]), (vector [y])) points

let wopt = sgd f (vector [0.; 0.; 0.]) 0.0001 0.01 train




//
//
//namedParams ["x", box [for x in 0. ..0.1..6. -> x]
//             "y", box [for x in 0. ..0.1..6. -> f wopt (vector [x])]
//             "xlab", box ""
//             "ylab", box ""
//             "type", box "l"]
//|> R.plot
//
//namedParams ["x", box (List.map fst points)
//             "y", box (List.map snd points)
//             "cex", box 1.5]
//|> R.points

