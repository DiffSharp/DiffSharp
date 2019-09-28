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
        
[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    // DiffSharp.Seed(123)
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

    // let a = Tensor.Create(0.)
    // let b = Tensor.Create(0.).ForwardDiff(Tensor.Create(2.))
    // let c = Tensor.Create(1.).ReverseDiff()

    let p = Tensor.Create([[0.1;0.1;0.8]; [0.8;0.1;0.1]])
    let d = Categorical(p)
    let s = d.Sample(10)
    printfn "%A" d
    printfn "%A" s


    0 // return an integer exit code
