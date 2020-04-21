// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.Data
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Backend.None
#nowarn "0058"


type Net() =
    inherit Model()
    let fc1 = Linear(28*28, 64)
    let fc2 = Linear(64, 64)
    let fc3 = Linear(64, 10)
    do base.addParameters(["fc1", fc1; "fc2", fc2])
    override __.forward(x) =
        x
        |> dsharp.view [-1; 28*28]
        |> fc1.forward
        |> dsharp.tanh
        |> fc2.forward
        |> dsharp.tanh
        |> fc3.forward


[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    dsharp.seed(12)

    let dataset = MNIST("./data", train=true, transform=fun t -> ((t/255)-0.1307)/0.3081)
    let dataloader = dataset.loader(32)

    let net = Net()
    let optimizer = SGD(net, learningRate=dsharp.tensor(0.1))
    for i, data, targets in dataloader do
        net.reverseDiff()
        let o = net.forward(data)
        let loss = dsharp.crossEntropyLoss(o, targets.view(-1))
        loss.reverse()
        optimizer.step()
        printfn "minibatch %A, loss %A" i (loss.toScalar())



    0 // return an integer exit code
