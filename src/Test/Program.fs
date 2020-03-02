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

type dsharp = DiffSharp
[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    DiffSharp.seed(12)

    0 // return an integer exit code
