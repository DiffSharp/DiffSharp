// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.Data
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Backends.Reference
// #nowarn "0058"


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
    //     let convall = conv1.forward 
    //                   >> dsharp.relu 
    //                   >> conv2.forward 
    //                   >> dsharp.relu 
    //                   >> dsharp.maxpool2d 2
    //     let k = dsharp.randn([1;1;28;28]) |> convall |> dsharp.nelement
    //     let fc1 = Linear(k, 128)
    //     let fc2 = Linear(128, 10)
    //     Model.create [conv1; conv2; fc1; fc2] 
    //                  (convall
    //                     >> dsharp.flatten 1
    //                     >> fc1.forward
    //                     >> dsharp.relu
    //                     >> fc2.forward)
    // let net = cnn()

    // printfn "params: %A" (net.nparameters())
    // // printfn "params: %A" (net.Parameters)

    // // let optimizer = SGD(net, learningRate=dsharp.tensor(0.01), momentum=dsharp.tensor(0.9), nesterov=true)
    // // let mutable epoch = -1
    // // let mutable stop = false
    // // while not stop do
    // //     epoch <- epoch + 1
    // //     for i, data, targets in dataloader.epoch() do
    // //         net.reverseDiff()
    // //         let o = net.forward(data)
    // //         let loss = dsharp.crossEntropyLoss(o, targets)
    // //         loss.reverse()
    // //         optimizer.step()
            
    // //         let loss = loss.toScalar() :?> float32
    // //         printfn "epoch %A, minibatch %A, loss %A\r" epoch i loss
        
    // // let loss data target p = net.forwardCompose (dsharp.crossEntropyLoss(target=target)) data p
    // let loss = net.forwardLoss dsharp.crossEntropyLoss
    // let mutable p = net.getParameters()
    // for i, data, target in dataloader.epoch() do
    //     let loss, g = dsharp.fgrad (loss data target) p
    //     p <- p - 0.1 * g
    //     printfn "%A %A" i loss

    let rosenbrock (x:Tensor) = 
        let x, y = x.[0], x.[1]
        (1. - x)**2 + 100. * (y - x**2)**2

    // let mutable x = dsharp.randn([2])
    // for i=0 to 10 do
    //     let f, g = dsharp.fg rosenbrock x
    //     printfn "i:%A f:%A g:%A x:%A" i f g x
    //     x <- x - 0.001 * g

    let x0 = dsharp.tensor([1.5, 1.5])
    let fx, x = Optimizer.sgd(rosenbrock, x0, iters=2000, lr=dsharp.tensor(0.001), momentum=dsharp.tensor(0.9), threshold=1e-8)
    printfn "%A %A %A" fx (rosenbrock x) x

    // let x0 = dsharp.randn([2])
    // let fx, lr = SGD.optimize(
    //                     (fun lr -> 
    //                         printfn "lr %A" lr; fst <| SGD.optimize(rosenbrock, x0, lr=lr, iters=10))
    //                         , dsharp.tensor(0.0001), lr=dsharp.tensor(1e-9), iters=10)
    // printfn "%A %A" fx lr

    0 // return an integer exit code
