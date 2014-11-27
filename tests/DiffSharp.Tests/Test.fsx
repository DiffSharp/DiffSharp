
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"


open DiffSharp.AD.Forward
open DiffSharp.AD.Forward.Vector

// Gradient descent, with function f, starting at x0, step size a, threshold t
let argmin f x0 (a:float) t =
    // Descending sequence, with state s (current x, gradient of f at current x)
    let gd = Seq.unfold (fun s -> 
                         if norm (snd s) < t then None 
                         else Some(fst s, (fst s - a * snd s, grad f (fst s))))
                        (x0, grad f x0)
    (Seq.last gd, gd)



let rnd = new System.Random()
let data = Array.init 20 (fun i -> Vector.Create(2, fun i -> (rnd.NextDouble())))

//
//let wccs (means:Vector<Dual>[]) (data:Vector<Dual>[]) =
//    Array.sum [|for d in data do
//                    let dist = Array.map (fun m -> (d - m).GetNorm()) means
//                    yield (Array.sort dist).[0]|]
//
//
//let kmeans k (data:Vector<float>[]) =
//    let dim = data.[0].Length
//    let d = Array.map (fun (a:Vector<float>) -> a.Convert(dual)) data
//    let inline extract (x:Vector<_>) = Array.init k (fun i -> Vector.Create(dim, fun j -> x.[i * dim + j]))
//    let means = Vector.Create(k * dim, fun _ -> (rnd.NextDouble()))
//    let min, _ = argmin (fun x -> wccs (extract x) d) means 0.0001 1.
//    extract min



let kmeans2 k (data:Vector<float>[]) =
    let dim = data.[0].Length
    let d = Array.map (fun (a:Vector<float>) -> a.Convert(dual)) data
    let inline extract (x:Vector<_>) = Array.init k (fun i -> Vector.Create(dim, fun j -> x.[i * dim + j]))
    let means = Vector.Create(k * dim, fun _ -> (rnd.NextDouble()))
    let min, _ = argmin (fun x -> 
                            let wccs (means) (data:Vector<_>[]) =
                                Array.sum [|for d in data do
                                                let dist = Array.map (fun m -> (d - m).GetNorm()) means
                                                yield (Array.sort dist).[0]|]
                            wccs (extract x) d) means 0.0001 1.
    extract min



let means = kmeans2 4 data

open FSharp.Charting

Chart.Combine(
    [Chart.Point (Array.map (fun (v:Vector<float>) -> (v.[0], v.[1])) means)
     Chart.Point (Array.map (fun (v:Vector<float>) -> (v.[0], v.[1])) data)]).WithXAxis(Min=0., Max=1.).WithYAxis(Min=0., Max=1.)

