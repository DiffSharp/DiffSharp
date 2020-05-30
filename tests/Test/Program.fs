// Learn more about F# at http://fsharp.org

open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data


[<EntryPoint>]
let main _ =
    printfn "Hello World from F#!"

    dsharp.seed(12)
    dsharp.config(backend=Backend.Torch)

    let dataset = MNIST("./data", train=true)
    let dataloader = dataset.loader(16, shuffle=true)

    let cnn() =
        let convs = Conv2d(1, 32, 3) 
                    --> dsharp.relu 
                    --> Conv2d(32, 64, 3) 
                    --> dsharp.relu 
                    --> dsharp.maxpool2d 2
        let k = dsharp.randn([1;1;28;28]) --> convs --> dsharp.nelement
        convs 
        --> dsharp.flatten 1 
        --> Linear(k, 128) 
        --> dsharp.relu 
        --> Linear(128, 10)

    let feedforward() = 
        dsharp.flatten 1 
        --> Linear(28*28, 128) 
        --> Linear(128, 10)


    let net = cnn()
    // let net = feedforward()
    printfn "net params: %A" net.nparameters

    printfn "%A" net.parameters.backend
    Optimizer.adam(net, dataloader, dsharp.crossEntropyLoss, iters=10, threshold=0.1)
    let o = dsharp.randn([1;1;28;28]) --> net
    printfn "%A %A" o o.backend


    0 // return an integer exit code
