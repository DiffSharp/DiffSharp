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

    // dsharp.nestReset()
    // let x = dsharp.tensor(2.)
    // printfn "\n###1\n%A" x
    // let z = dsharp.ppdiffn 1 sin x
    // printfn "%A" z

    // dsharp.nestReset()
    // let x = dsharp.tensor(2.)
    // printfn "\n###2\n%A" x
    // let z = dsharp.ppdiffn 2 sin x
    // printfn "%A" z

    // dsharp.nestReset()
    // let x = dsharp.tensor(2.)
    // printfn "\n###3\n%A" x
    // let z = dsharp.ppdiffn 3 sin x
    // printfn "%A" z

    // dsharp.nestReset()
    // let x = dsharp.tensor(2.)
    // printfn "\n###4\n%A" x
    // let z = dsharp.ppdiffn 4 sin x
    // printfn "%A" z


    // let f (x:Tensor) =
    //     sin (dsharp.stack([x.[2]*x.[1]; x.[1]*x.[0]; x.[0]*x.[2]]))

    // let x = dsharp.tensor([1,2,3])
    // printfn "%A" x
    // let j = dsharp.jacobian f x
    // printfn "%A" j

    let fvect2vect2 (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.stack([x*x*y; 5*x+sin y])
    let fvect2vect2Jacobian (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.tensor([[2*x*y; x*x];[dsharp.tensor(5.); cos y]])

    let fvect3vect2 (x:Tensor) = 
        let x, y, z = x.[0], x.[1], x.[2]
        dsharp.stack([x*y+2*y*z;2*x*y*y*z])
    let fvect3vect2Jacobian (x:Tensor) = 
        let x, y, z = x.[0], x.[1], x.[2]
        dsharp.tensor([[y;x+2*z;2*y];[2*y*y*z;4*x*y*z;2*x*y*y]])

    let fvect3vect4 (x:Tensor) =
        let y1, y2, y3, y4 = x.[0], 5*x.[2], 4*x.[1]*x.[1]-2*x.[2],x.[2]*sin x.[0]
        dsharp.stack([y1;y2;y3;y4])

    let fvect3vect4Jacobian (x:Tensor) =
        let z, o = dsharp.zero(), dsharp.one()
        dsharp.tensor([[o,z,z],[z,z,5*o],[z,8*x.[1],-2*o],[x.[2]*cos x.[0],z,sin x.[0]]])

    let x = dsharp.tensor([1.,2.])
    let fx = fvect2vect2 x
    let j = dsharp.jacobian fvect2vect2 x
    let jtrue = fvect2vect2Jacobian x
    printfn "%A" x
    printfn "%A" fx
    printfn "%A" j
    printfn "%A" jtrue

    let x = dsharp.tensor([1.2,2.,3.])
    let fx = fvect3vect2 x
    let j = dsharp.jacobian fvect3vect2 x
    let jtrue = fvect3vect2Jacobian x
    printfn "%A" x
    printfn "%A" fx
    printfn "%A" j
    printfn "%A" jtrue

    let x = dsharp.tensor([1.2,2.,3.])
    let fx = fvect3vect4 x
    let j = dsharp.jacobian fvect3vect4 x
    let jtrue = fvect3vect4Jacobian x
    printfn "%A" x
    printfn "%A" fx
    printfn "%A" j
    printfn "%A" jtrue

    0 // return an integer exit code
