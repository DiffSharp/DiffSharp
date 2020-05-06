// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.Data
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Backends.None
// #nowarn "0058"


// type Net() =
//     inherit Model()
//     let fc1 = Linear(28*28, 64)
//     let fc2 = Linear(64, 32)
//     let fc3 = Linear(32, 10)
//     do base.add(["fc1", fc1; "fc2", fc2; "fc3", fc3])
//     override __.forward(x) =
//         x
//         |> dsharp.view [-1; 28*28]
//         |> fc1.forward
//         |> dsharp.relu
//         |> fc2.forward
//         |> dsharp.relu
//         |> fc3.forward

type Net() =
    inherit Model()
    let conv1 = Conv2d(1, 2, 3)
    let conv2 = Conv2d(2, 4, 3)
    let k = dsharp.randn([1;1;28;28]) |> conv1.forward |> conv2.forward |> dsharp.nelement
    let fc1 = Linear(k, 128)
    let fc2 = Linear(128, 10)
    do base.add([conv1; conv2; fc1; fc2])
    override __.forward(x) =
        x
        // |> dsharp.view [-1; 28*28]
        |> conv1.forward
        |> dsharp.relu
        |> conv2.forward
        |> dsharp.relu
        |> dsharp.flatten 1
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward


[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    dsharp.seed(12)

    // let dataset = MNIST("./data", train=true)
    // let dataloader = dataset.loader(8, shuffle=true, numBatches=50)

    // // let net = Net()

    // let cnn () =
    //     let conv1 = Conv2d(1, 2, 3)
    //     let conv2 = Conv2d(2, 4, 3)
    //     let k = dsharp.randn([1;1;28;28]) |> conv1.forward |> conv2.forward |> dsharp.nelement
    //     let fc1 = Linear(k, 128)
    //     let fc2 = Linear(128, 10)
    //     Model.create [conv1; conv2; fc1; fc2] 
    //                  (conv1.forward
    //                     >> dsharp.relu
    //                     >> conv2.forward
    //                     >> dsharp.relu
    //                     >> dsharp.flatten 1
    //                     >> fc1.forward
    //                     >> dsharp.relu
    //                     >> fc2.forward)
    // let net = cnn()

    // printfn "params: %A" (net.nparameters())
    // printfn "params: %A" (net.Parameters)

    // let optimizer = SGD(net, learningRate=dsharp.tensor(0.01), momentum=dsharp.tensor(0.9), nesterov=true)
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
        
    // let loss data target p = net.forwardCompose (dsharp.crossEntropyLoss(target=target)) data p
    // let loss = net.forwardLoss dsharp.crossEntropyLoss
    // let mutable p = net.getParameters()
    // for i, data, target in dataloader.epoch() do
    //     let loss, g = dsharp.pgrad (loss data target) p
    //     p <- p - 0.1 * g
    //     printfn "%A %A" i loss

    let a = dsharp.randn([])
    let b = dsharp.randn([10])
    let c = dsharp.randn([10;20])
    let d = dsharp.randn([10;20;30])
    let e = dsharp.randn([10;20;30;40])
    // let at = a.transpose() // Fails because a.dim < 2
    // let bt = b.transpose() // Fails because b.dim < 2
    let ct = c.transpose()
    let dt = d.transpose()
    let et = e.transpose()

    // printfn "%A %A" a.shape at.shape
    // printfn "%A %A" b.shape bt.shape
    printfn "%A %A" c.shape ct.shape
    printfn "%A %A" d.shape dt.shape
    printfn "%A %A" e.shape et.shape

    0 // return an integer exit code
