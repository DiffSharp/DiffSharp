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

    let rosenbrock (x:Tensor) = 
        let x, y = x.[0], x.[1]
        (1. - x)**2 + 100. * (y - x**2)**2
    let rosenbrockGrad (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.tensor([-2*(1-x)-400*x*(-(x**2) + y); 200*(-(x**2) + y)])
    let rosenbrockHessian (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.tensor([[2.+1200.*x*x-400.*y, -400.*x],[-400.*x, 200.]])

    let x = dsharp.tensor([1.5, 2.5])
    let v = dsharp.tensor([0.5, -2.])
    printfn "%A" x
    printfn "%A" v
    let fx, gv, hv = dsharp.pgradhessianv rosenbrock x v
    printfn "\n\nfx\n%A" fx
    printfn "\n\ngv\n%A" gv
    printfn "\n\nhv\n%A" hv

    let fx, h = dsharp.phessian rosenbrock x
    printfn "\n\nfx\n%A" fx
    printfn "\n\nh\n%A" h

    0 // return an integer exit code
