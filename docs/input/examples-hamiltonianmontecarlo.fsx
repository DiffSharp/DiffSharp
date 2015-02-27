(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
Hamiltonian Monte Carlo
=======================


*)


open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra

let rnd = new System.Random()

// Uniform random number
let unifRnd() = rnd.NextDouble()

// Gaussian random number
let rec gaussRnd() =
    let x, y = unifRnd() * 2.0 - 1.0, unifRnd() * 2.0 - 1.0
    let s = x * x + y *y
    if s > 1.0 then gaussRnd() else x * sqrt (-2.0 * (log s) / s)


let leapFrog u k d steps (x0, p0) =
    let hd = d / 2.
    [1..steps] 
    |> List.fold (fun (x, p) _ ->
        let p' = p - hd * grad u x
        let x' = x + d * grad k p'
        x', p' - hd * grad u x') (x0, p0)
   
let hmc n hdelta hsteps (x0:Vector<_>) (dist:Vector<Adj>->Adj) =
    let u x = -log (dist x)
    let k p = 0.5 * Vector.fold (fun acc a -> acc + a * a) (adj 0.) p
    let hamilton x p = u (Vector.map adj x) + k (Vector.map adj p) |> float
    let x = ref x0
    [|for i in 1..n do
        let p = Vector.init x0.Length (fun _ -> gaussRnd())
        let x', p' = leapFrog u k hdelta hsteps (!x, p)
        if unifRnd() < exp ((hamilton !x p) - (hamilton x' p')) then x := x'
        yield !x|]



let multiNormal mu sigma (x:Vector<Adj>) =
    let s = sigma |> Matrix.inverse |> Matrix.map adj
    let m = mu |> Vector.map adj
    exp (-((x - m) * s * (x - m)) / 2.)


let samples = 
    multiNormal (vector [1.; 5.]) (matrix [[1.; 0.8]; [0.8; 1.]])
    |> hmc 5000 0.1 10 (vector [0.; 0.])


open FSharp.Charting

Chart.Point(samples |> Array.map (fun v -> v.[0], v.[1]), MarkerSize = 2).WithXAxis(Min = -2., Max = 4.)
