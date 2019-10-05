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
    do base.AddParameters(["fc1", fc1])
    override l.Forward(x) =
        x 
        |> fc1.Forward |> Tensor.Relu
        |> fc2.Forward |> Tensor.Relu

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

    // DiffSharp.Seed(125)
    // DiffSharp.NestReset()
    // let model = Model()
    // model.ReverseDiff()
    // let data = Tensor.Create([[0.;0.;0.];[0.;1.;1.];[1.;0.;1.];[1.;1.;0.]])
    // let x = data.[*,0..1]
    // let y = data.[*,2..]
    // printfn "%A" x
    // printfn "%A" y

    // let mseloss (x:Tensor) (y:Tensor) = Tensor.Sum((x - y) * (x - y)) / x.Shape.[0]

    // for i=0 to 10000 do    
    //     model.ReverseDiff()
    //     let o = model.Forward(x).View([4])
    //     let loss = mseloss o y
    //     printfn "prediction: %A, loss: %A" (o.NoDiff()) (loss.NoDiff())
    //     // printfn "%A" loss
    //     loss.Reverse()
    //     optimize model (Tensor.Create(0.01))

    let x = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
    let x0 = x.Sum(0)
    let x1 = x.Sum(1)
    let x2 = x.Sum(2)
    printfn "%A" x
    printfn "\n%A" x0
    printfn "\n%A" x1
    printfn "\n%A" x2
    0 // return an integer exit code
