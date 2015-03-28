
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

open DiffSharp.AD.Nested.Forward
open DiffSharp.AD.Nested.Forward.Vector
open DiffSharp.Util.LinearAlgebra
open FSharp.Charting

let dt = Df 0.1
let x0 = vector [Df 0.; Df 8.]
let v0 = vector [Df 0.75; Df 0.]

let p w (x:Vector<_>) = (1. / Vector.norm (x - vector [Df 10.; Df 10. - w])) + (1. / Vector.norm (x - vector [Df 10.; Df 0.]))

let trajectory (w:D) = (x0, v0) |> Seq.unfold (fun (x, v) -> 
                                                    let a = -grad (p w) x
                                                    let v = v + dt * a
                                                    let x = x + dt * v
                                                    Some(x, (x, v)))

let error w =
    trajectory w 
    |> Seq.takeWhile (fun (x:Vector<_>) -> x.[1] > Df 0.)
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
    |> Seq.takeWhile (fun (x:Vector<_>) -> x.[1] > Df 0.)
    |> Seq.map (Vector.map float) 
    |> Seq.map (fun (x:Vector<_>) -> x.[0], x.[1])
    |> Chart.Line


let ws = optimize (Df 0.) (Df 1e-11)

let test = ws |> Seq.toArray
let test2 = ws |> Seq.last |> error

ws
|> Seq.map plot
|> Chart.Combine
|> Chart.WithXAxis(Min = 0., Max = 10.)