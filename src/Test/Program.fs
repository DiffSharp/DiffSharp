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
// #nowarn "0058"


type Net() =
    inherit Model()
    let fc1 = Linear(28*28, 64)
    let fc2 = Linear(64, 32)
    let fc3 = Linear(32, 10)
    do base.addParameters(["fc1", fc1; "fc2", fc2; "fc3", fc3])
    override __.forward(x) =
        x
        |> dsharp.view [-1; 28*28]
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward
        |> dsharp.relu
        |> fc3.forward


[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    dsharp.seed(12)

    let dataset = MNIST("./data", train=true)
    let dataloader = dataset.loader(32, shuffle=true)

    let net = Net()
    let optimizer = SGD(net, learningRate=dsharp.tensor(0.01), momentum=dsharp.tensor(0.9), nesterov=true)

    let mutable epoch = -1
    let mutable stop = false
    while not stop do
        epoch <- epoch + 1
        for i, data, targets in dataloader.epoch() do
            net.reverseDiff()
            let o = net.forward(data)
            let loss = dsharp.crossEntropyLoss(o, targets)
            loss.reverse()
            optimizer.step()
            
            let loss = loss.toScalar() :?> float32
            printfn "epoch %A, minibatch %A, loss %A\r" epoch i loss


    0 // return an integer exit code
