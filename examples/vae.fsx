// Learn more about F# at http://fsharp.org
#r "../bundles/DiffSharp-cpu/bin/Debug/netcoreapp3.0/DiffSharp.Core.dll"
#r "../bundles/DiffSharp-cpu/bin/Debug/netcoreapp3.0/DiffSharp.Backends.Torch.dll"
#r "/home/gunes/.nuget/packages/torchsharp/0.3.52267/lib/netcoreapp3.0/TorchSharp.dll"
open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data


type VAE(xDim:int, zDim:int, ?hDims:seq<int>, ?activation:Tensor->Tensor, ?activationLast:Tensor->Tensor) =
    inherit Model()
    let hDims = defaultArg hDims (let d = (xDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let activation = defaultArg activation dsharp.relu
    let activationLast = defaultArg activationLast dsharp.sigmoid
    let dims =
        if hDims.Length = 0 then
            [|xDim; zDim|]
        else
            Array.append (Array.append [|xDim|] hDims) [|zDim|]
            
    let enc = Array.append [|for i in 0..dims.Length-2 -> Linear(dims.[i], dims.[i+1])|] [|Linear(dims.[dims.Length-2], dims.[dims.Length-1])|]
    let dec = [|for i in 0..dims.Length-2 -> Linear(dims.[i+1], dims.[i])|] |> Array.rev
    do 
        base.add([for m in enc -> box m])
        base.add([for m in dec -> box m])

    let encode x =
        let mutable x = x
        for i in 0..enc.Length-3 do
            x <- activation <| enc.[i].forward(x)
        let mu = enc.[enc.Length-2].forward(x)
        let logVar = enc.[enc.Length-1].forward(x)
        mu, logVar

    let latent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    let decode z =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- activation <| dec.[i].forward(h)
        activationLast <| dec.[dec.Length-1].forward(h)

    member _.encodeDecode(x:Tensor) =
        let mu, logVar = encode (x.view([-1; xDim]))
        let z = latent mu logVar
        decode z, mu, logVar

    member _.sample(?numSamples:int) = 
        let numSamples = defaultArg numSamples 1
        dsharp.randn([|numSamples; zDim|]) |> decode
    override _.ToString() = sprintf "VAE(%A, %A, %A)" xDim hDims zDim
    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x



dsharp.config(backend=Backend.Torch, device=Device.GPU)
dsharp.seed(0)

let trainSet = MNIST("./mnist", train=true)
let validSet = MNIST("./mnist", train=false)
let trainLoader = trainSet.loader(batchSize=32, shuffle=true)
let validLoader = validSet.loader(batchSize=32)

let model = VAE(28*28, 16, [512; 256])
printfn "%A" model

let optimizer = Adam(model, lr=dsharp.tensor(0.001))

let loss xRecon (x:Tensor) (mu:Tensor) (logVar:Tensor) =
    let bce = dsharp.bceLoss(xRecon, x.view([|-1; 28*28|]), reduction="sum")
    let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
    bce + kl

for epoch = 0 to 2 do
    printfn "Epoch %A" epoch
    for _, x, _ in trainLoader.epoch() do
        model.reverseDiff()
        let xRecon, mu, logVar = model.encodeDecode x
        let l = loss xRecon x mu logVar
        l.reverse()
        optimizer.step()
        printfn "%A" (float(l))

        printfn "%A" (model.sample())
//         printfn "%A %A %A" z mu logVar
        // let loss = dsharp.bin

