
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"


open DiffSharp.AD.Forward
open DiffSharp.AD.Forward.Vector
open DiffSharp.Util.LinearAlgebra

let test = diff sqrt 0.

let inline f x = sqrt x

let tt = diff f 42.
let ttt = f 42.


// Gradient descent, with function f, starting at x0, step size a, threshold t
let argmin f x0 (a:float) t =
    // Descending sequence, with state s (current x, gradient of f at current x)
    let gd = Seq.unfold (fun s -> 
                         if Vector.norm (snd s) < t then None 
                         else Some(fst s, (fst s - a * snd s, grad f (fst s))))
                        (x0, grad f x0)
    (Seq.last gd, gd)



let rnd = new System.Random()
let data = Array.init 20 (fun i -> Vector.Create(2, fun i -> (rnd.NextDouble())))


let wccs (data:Vector<Dual>[]) (means:Vector<Dual>[]) =
    Array.sum [|for d in data do
                    let dist = Array.map (fun m -> Vector.norm (d - m)) means
                    yield (Array.sort dist).[0]|]


let kmeans k (data:Vector<float>[]) =
    let dim = data.[0].Length
    let d = Array.map (Vector.map dual) data
    let inline extract (x:Vector<_>) = Array.init k (fun i -> Vector.init dim (fun j -> x.[i * dim + j]))
    let means = Vector.init (k * dim) (fun _ -> (rnd.NextDouble()))
    let min, _ = argmin (extract >> wccs d) means 0.0001 1.
    extract min



let kmeans2 k (data:Vector<float>[]) =
    let dim = data.[0].Length
    let d = Array.map (Vector.map dual) data
    let inline extract (x:Vector<_>) = Array.init k (fun i -> Vector.init dim (fun j -> x.[i * dim + j]))
    let means = Vector.init (k * dim) (fun _ -> (rnd.NextDouble()))
    let min, _ = argmin (fun x -> 
                            let wccs (means) (data:Vector<_>[]) =
                                Array.sum [|for d in data do
                                                let dist = Array.map (fun m -> Vector.norm (d - m)) means
                                                yield (Array.sort dist).[0]|]
                            wccs (extract x) d) means 0.0001 1.
    extract min



let means = kmeans 4 data

open FSharp.Charting

Chart.Combine(
    [Chart.Point (Array.map (fun (v:Vector<_>) -> (v.[0], v.[1])) means)
     Chart.Point (Array.map (fun (v:Vector<_>) -> (v.[0], v.[1])) data)]).WithXAxis(Min=0., Max=1.).WithYAxis(Min=0., Max=1.)
