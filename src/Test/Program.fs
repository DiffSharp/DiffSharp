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
    do base.add(["fc1", fc1; "fc2", fc2; "fc3", fc3])
    override __.forward(x) =
        x
        |> dsharp.view [-1; 28*28]
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward
        |> dsharp.relu
        |> fc3.forward

// type Net() =
//     inherit Model()
//     let conv1 = Conv2d(1, 8, 3)
//     let conv2 = Conv2d(8, 16, 3)
//     let fc1 = Linear(9216, 128)
//     let fc2 = Linear(128, 10)
//     do 
//         base.add(["conv1", conv1; "conv2", conv2])
//         base.add(["fc1", fc1; "fc2", fc2])
//     override __.forward(x) =
//         x
//         // |> dsharp.view [-1; 28*28]
//         |> conv1.forward
//         |> dsharp.relu
//         |> conv2.forward
//         |> dsharp.relu
//         |> dsharp.flatten 1
//         |> fc1.forward
//         |> dsharp.relu
//         |> fc2.forward


[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    // dsharp.seed(12)

    let dataset = MNIST("./data", train=true)
    let dataloader = dataset.loader(32, shuffle=true, numBatches=800)

    let net = Net()
    printfn "params: %A" (net.nparameters())
    let optimizer = SGD(net, learningRate=dsharp.tensor(0.01), momentum=dsharp.tensor(0.9), nesterov=true)

    printfn "%A" (net.Parameters.values)

    // let mutable epoch = -1
    // let mutable stop = false
    // while not stop do
    //     epoch <- epoch + 1
    //     for i, data, targets in dataloader.epoch() do
    //         net.reverseDiff()
    //         let o = net.forward(data)
    //         let loss = dsharp.crossEntropyLoss(o, targets)
    //         loss.reverse()
    //         optimizer.step()
            
    //         let loss = loss.toScalar() :?> float32
    //         printfn "epoch %A, minibatch %A, loss %A\r" epoch i loss

    // let a = dsharp.randn(5)
    // let p = Parameter(a)
    // printfn "%A" a
    // printfn "%A" p.value
    // p.value <- dsharp.rand(3)
    // printfn "%A" p.value

    // let loss data target p = net.forwardCompose (dsharp.crossEntropyLoss(target=target)) data p
    // let loss = net.forwardLoss dsharp.crossEntropyLoss
    // let mutable p = net.getParameters()
    // for i, data, target in dataloader.epoch() do
    //     let loss, g = dsharp.pgrad (loss data target) p
    //     p <- p - 0.01 * g
    //     printfn "%A %A" i loss


    0 // return an integer exit code
