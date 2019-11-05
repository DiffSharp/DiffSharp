// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.Model
open DiffSharp.Optim


type FeedforwardNet() =
    inherit Model()
    let fc1 = Linear(2, 32)
    let fc2 = Linear(32, 1)
    do base.AddParameters(["fc1", fc1; "fc2", fc2])
    override l.Forward(x) =
        x 
        |> fc1.Forward |> Tensor.LeakyRelu
        |> fc2.Forward |> Tensor.LeakyRelu


[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    DiffSharp.Seed(12)
    DiffSharp.NestReset()
    let model = FeedforwardNet()
    let optimizer = SGD(model, Tensor.Create(0.01))
    printfn "%A" model.Parameters
    let data = Tensor.Create([[0.;0.;0.];[0.;1.;1.];[1.;0.;1.];[1.;1.;0.]])
    let x = data.[*,0..1]
    let y = data.[*,2..]
    // printfn "%A" x
    // printfn "%A" y

    for i=0 to 1000 do
        model.ReverseDiff()
        let o = model.Forward(x).View(-1)
        let loss = Tensor.MSELoss(o, y)
        printfn "prediction: %A, loss: %A" (o.NoDiff()) (loss.NoDiff())
        loss.Reverse()
        optimizer.Step()

    model.NoDiff()
    printfn "%A" model.Parameters

    0 // return an integer exit code
