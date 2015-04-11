
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

open DiffSharp.AD.Nested.Reverse
open DiffSharp.AD.Nested.Reverse.Vector
open FsAlg.Generic
open FSharp.Charting

let dt = D 0.1
let x0 = vector [D 0.; D 8.]
let v0 = vector [D 0.75; D 0.]

let p w (x:Vector<_>) = (1. / Vector.norm (x - vector [D 10.; D 10. - w])) + (1. / Vector.norm (x - vector [D 10.; D 0.]))

let trajectory (w:D) = (x0, v0) |> Seq.unfold (fun (x, v) -> 
                                                    let a = -grad (p w) x
                                                    let v = v + dt * a
                                                    let x = x + dt * v
                                                    Some(x, (x, v)))

let error w =
    trajectory w 
    |> Seq.takeWhile (fun (x:Vector<_>) -> x.[1] > D 0.)
    |> Seq.last
    |> Vector.get 0
    |> fun x -> x * x

let optimize w0 threshold =
    w0 |> Seq.unfold (fun w ->
                        let w = w - (diff error w) / diff (diff error) w
                        if abs(error w) < threshold then None else Some(w, w))
    |>Seq.append (Seq.ofList [w0])

let plot w = 
    trajectory w
    |> Seq.takeWhile (fun (x:Vector<_>) -> x.[1] > D 0.)
    |> Seq.map (Vector.map float) 
    |> Seq.map (fun (x:Vector<_>) -> x.[0], x.[1])
    |> Chart.Line


let ws = optimize (D 0.) (D 1e-1)

let test = ws |> Seq.toArray
let test2 = ws |> Seq.last |> error

ws
|> Seq.map plot
|> Chart.Combine
|> Chart.WithXAxis(Min = 0., Max = 10.)

//let test3 = diff (fun x -> x * (diff (fun y -> x * y) (D 3.))) (D 6.)