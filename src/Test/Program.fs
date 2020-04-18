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
open System.Runtime.Serialization
open System.Text
open System.Xml 
open System.IO
#nowarn "0058"
open System.IO
open System.Runtime.Serialization.Formatters.Binary


type Net() =
    inherit Model()
    let fc1 = Linear(28*28, 64)
    let fc2 = Linear(64, 10)
    do base.addParameters(["fc1", fc1; "fc2", fc2])
    override __.forward(x) =
        x
        |> dsharp.view [-1; 28*28]
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward
        |> dsharp.softmax 1


[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    dsharp.seed(12)

    let dataset = MNIST("./data", train=false, transform=fun t -> ((t/255)-0.1307)/0.3081)
    let dataloader = dataset.loader(64)

    for i, data, targets in dataloader do
        printfn "minibatch %A" i
        printfn "data %A" data.shape
        printfn "targets %A" targets.shape


    0 // return an integer exit code
