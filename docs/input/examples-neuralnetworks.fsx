(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

(**
Neural Networks
===============

link to reverse ad page for an explanation of the usage of adjoints in code.

linear separability of or
*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra


type Neuron =
    {mutable w:Vector<Adj> // Weight vector of this neuron
     mutable b:Adj} // Bias of this neuron
    
type Layer =
    {n:Neuron[]} // The neurons forming this layer

type Network =
    {l:Layer[]} // The layers forming this network


let r = new System.Random()
let rs = r.Next()
//let rnd = new Random(1082061959)
//let rnd = new Random(622422388)
let rnd = new System.Random()

let createNetwork (inputs:int) (layers:int[]) =
    {l = Array.init layers.Length (fun i -> 
        {n = Array.init layers.[i] (fun j -> 
            {w = Vector.init
                     (if i = 0 then inputs else layers.[i - 1])
                     (fun k -> adj (-0.5 + rnd.NextDouble()))
             b = adj (-0.5 + rnd.NextDouble())})})}

let sigmoid (x:Adj) = 1. / (1. + exp -x)

let runNeuron (x:Vector<Adj>) (n:Neuron) =
    x * n.w + n.b
    |> sigmoid

let runLayer (x:Vector<Adj>) (l:Layer) =
    Array.map (runNeuron x) l.n
    |> vector

let runNetwork (x:Vector<Adj>) (n:Network) =
    Seq.fold (fun o l -> runLayer o l) x n.l


let backprop (t:(Vector<float>*Vector<float>)[]) (eta:float) (n:Network) =
    let ta = Array.map (fun x -> Vector.map adj (fst x), Vector.map adj (snd x)) t
    seq {for i in 0 .. 10000 do // An arbitrary timeout
            Trace.Clear()
            let error = 
                (1. / float t.Length) * Array.sumBy 
                    (fun t -> Vector.norm ((snd t) - runNetwork (fst t) n)) ta
            error.A <- 1.
            Trace.ReverseSweep()
            for l in n.l do
                for n in l.n do
                    n.b <- n.b - eta * n.b.A
                    n.w <- Vector.map (fun (w:Adj) -> w - eta * w.A) n.w
            if i = 10000 then printfn "Failed to converge within 10000 steps"
            yield primal error}
    |> Seq.takeWhile (fun x -> x > 0.005)

let net = createNetwork 2 [|3; 1|]

//let test2 = runNetwork (Vector.create 2 (adj (rnd.NextDouble()))) test

let trainOR = [|vector [0.; 0.], vector [0.]
                vector [0.; 1.], vector [1.]
                vector [1.; 0.], vector [1.]
                vector [1.; 1.], vector [1.]|]

let trainXOR = [|vector [0.; 0.], vector [0.]
                 vector [0.; 1.], vector [1.]
                 vector [1.; 0.], vector [1.]
                 vector [1.; 1.], vector [0.]|]

let test = backprop trainXOR 0.9 net


open FSharp.Charting

Chart.Line test
//
//Chart.Line([for s in 0..1000 -> (s, primal (backprop trainingSet 0.01 s net))])