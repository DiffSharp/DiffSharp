// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Backend.None

#nowarn "0058"

[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    dsharp.seed(12)

    // let f x =
    //     dsharp.sum(x)

    // let x = dsharp.rand([])
    // let fx, g = dsharp.pgrad f x

    // printfn "%A" x
    // printfn "%A" fx
    // printfn "%A" g

    // let x = dsharp.tensor(2.)
    // printfn "%A" x
    // dsharp.nest()
    // let x = x.forwardDiff(dsharp.tensor(1.))
    // printfn "%A" x
    // dsharp.nest()
    // let x = x.forwardDiff(dsharp.tensor(1.))
    // printfn "%A" x
    // dsharp.nest()
    // let x = x.forwardDiff(dsharp.tensor(1.))
    // printfn "%A" x
    // let z = sin x
    // printfn "%A" z

    dsharp.nestReset()
    let x = dsharp.tensor(2.)
    printfn "\n###1\n%A" x
    let z = dsharp.ppdiffn 1 sin x
    printfn "%A" z

    dsharp.nestReset()
    let x = dsharp.tensor(2.)
    printfn "\n###2\n%A" x
    let z = dsharp.ppdiffn 2 sin x
    printfn "%A" z

    dsharp.nestReset()
    let x = dsharp.tensor(2.)
    printfn "\n###3\n%A" x
    let z = dsharp.ppdiffn 3 sin x
    printfn "%A" z

    dsharp.nestReset()
    let x = dsharp.tensor(2.)
    printfn "\n###4\n%A" x
    let z = dsharp.ppdiffn 4 sin x
    printfn "%A" z


    0 // return an integer exit code
