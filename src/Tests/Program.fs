// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.NN
open DiffSharp.RawTensor


type Model() =
    inherit Layer()
    let fc1 = Linear(2, 16)
    let fc2 = Linear(16, 1)
    do base.AddParameters(["fc1", fc1; "fc2", fc2])
    override l.Forward(x) =
        x 
        |> fc1.Forward |> Tensor.LeakyRelu
        |> fc2.Forward |> Tensor.LeakyRelu

let optimize (model:Layer) (lr:Tensor) =
    let update k (p:Parameter) = 
        // printfn "updating %A" k; 
        p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative
    model.Map(update)

[<AutoOpen>]
module ExtraPrimitives =
    let inline tryUnbox<'a> (x:obj) =
        match x with
        | :? 'a as result -> Some (result)
        | _ -> None


    // if dim = 0 then
    //     let mutable s = Tensor.ZerosLike(t).[0]
    //     for i=0 to t.Shape.[0]-1 do
    //         s <- s + t.[i]
    //     s
    // elif dim = 1 then
    //     let mutable s = Tensor.ZerosLike(t).[*,0]
    //     for i=0 to t.Shape.[1]-1 do
    //         s <- s + t.[*,i]
    //     s
    // else
    //     failwith "Not implemented"

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    DiffSharp.Seed(12)
    DiffSharp.NestReset()
    let model = Model()
    printfn "%A" model.Parameters
    model.ReverseDiff()
    let data = Tensor.Create([[0.;0.;0.];[0.;1.;1.];[1.;0.;1.];[1.;1.;0.]])
    let x = data.[*,0..1]
    let y = data.[*,2..]
    printfn "%A" x
    printfn "%A" y

    let mseloss (x:Tensor) (y:Tensor) = Tensor.Sum((x - y) * (x - y)) / x.Shape.[0]

    for i=0 to 10000 do    
        model.ReverseDiff()
        let o = model.Forward(x).View(-1)
        let loss = mseloss o y
        printfn "prediction: %A, loss: %A" (o.NoDiff()) (loss.NoDiff())
        // printfn "%A" loss
        loss.Reverse()
        optimize model (Tensor.Create(0.01))

    printfn "%A" model.Parameters
    // let a = Tensor.RandomNormal([5;4])
    // let b = a.View([|2;-1;5|])
    // printfn "%A %A" a a.Shape
    // printfn "%A %A" b b.Shape

    0 // return an integer exit code
