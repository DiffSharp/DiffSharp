// Learn more about F# at http://fsharp.org

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data

[<EntryPoint>]
let main _ =
    printfn "Hello World from F#!"

    // let din, dout = 1024, 512
    // let batchSize = 512
    // let net = Linear(din, dout)
    // let optimizer = SGD(net, lr=dsharp.tensor(0.01))

    // let steps = 10
    // for i in 0..steps do
    //     let inputs, targets = dsharp.randn([batchSize; din]), dsharp.randn([batchSize; dout])
    //     net.reverseDiff()
    //     let y = net.forward(inputs)
    //     let loss = dsharp.mseLoss y targets
    //     loss.reverse()
    //     optimizer.step()
    //     printfn "%A %A" i (float loss)
    
    dsharp.config(backend=Backend.Torch)
    dsharp.seed(12)

    let dataset = MNIST("./data", train=true)
    let dataloader = dataset.loader(64, shuffle=true)

    // let net =
    //     let convs = Conv2d(1, 32, 3)
    //                 --> dsharp.relu
    //                 --> Conv2d(32, 64, 3)
    //                 --> dsharp.relu
    //                 --> dsharp.maxpool2d 2
    //     let k = dsharp.randn([1;1;28;28]) --> convs --> dsharp.nelement
    //     convs
    //     --> dsharp.flatten 1
    //     --> Linear(k, 128)
    //     --> dsharp.relu
    //     --> Linear(128, 10)

    let net =
        dsharp.flatten 1
        --> Linear(28*28, 128)
        --> Linear(128, 10)

    printfn "net params: %A" net.nparameters

    printfn "%A" net.parameters.backend
    Optimizer.adam(net, dataloader, dsharp.crossEntropyLoss, iters=200, threshold=0.1)
    let i, t = dataset.item(0)
    let o = i --> dsharp.move() --> dsharp.unsqueeze(0) --> net
    printfn "%A %A %A" o o.backend t    
    0 // return an integer exit code
