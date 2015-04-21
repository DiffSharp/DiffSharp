
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.10/FSharp.Charting.fsx"

open DiffSharp.AD
open FsAlg.Generic

type Layer =
    {W:Matrix<D>  // Weigth matrix
     b:Vector<D>} // Bias matrix

type Network =
    {layers:Layer[]} // The layers forming this network

let sigmoid (x:D) = 1. / (1. + exp -x)

let runLayer (x:Vector<D>) (l:Layer) =
    l.W * x + l.b
    |> Vector.map sigmoid

let runNetwork (x:Vector<D>) (n:Network) =
    Array.fold runLayer x n.layers


let rnd = System.Random()

let createNetwork (l:int[]) =
    {layers = Array.init (l.Length - 1) (fun i ->
        {W = Matrix.init l.[i + 1] l.[i] (fun _ _ -> D (-0.5 + rnd.NextDouble()))
         b = Vector.init l.[i + 1] (fun _ -> D (-0.5 + rnd.NextDouble()))})}


let backprop (n:Network) eta epsilon timeout (t:(Vector<_>*Vector<_>)[]) =
    let i = DiffSharp.Util.General.GlobalTagger.Next
    seq {for j in 0 .. timeout do
            for l in n.layers do
                l.W |> Matrix.replace (makeDR i)
                l.b |> Vector.replace (makeDR i) 

            let error = t |> Array.sumBy (fun (x, y) -> Vector.normSq (y - runNetwork x n))
            error |> resetTrace
            error |> reverseTrace (D 1.)

            for l in n.layers do
                l.W |> Matrix.replace (fun (x:D) -> x.P - eta * x.A)
                l.b |> Vector.replace (fun (x:D) -> x.P - eta * x.A)

            if j = timeout then printfn "Failed to converge within %i steps." timeout
            yield float error}
    |> Seq.takeWhile ((<) epsilon)



let trainOR = [|vector [D 0.; D 0.], vector [D 0.]
                vector [D 0.; D 1.], vector [D 1.]
                vector [D 1.; D 0.], vector [D 1.]
                vector [D 1.; D 1.], vector [D 0.]|]


//let net1 = train [|2; 2; 1|] 0.9 0.005 10000 trainOR

let net1 = createNetwork [|2; 10; 1|]

let train1 = backprop net1 0.9 0.005 10000 trainOR

open FSharp.Charting

Chart.Line train1