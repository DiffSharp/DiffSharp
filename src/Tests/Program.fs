// Learn more about F# at http://fsharp.org

open System
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    // let a = flatArrayAndShape<float32> 2.f
    // let b = flatArrayAndShape<float32> [2.f; 3.f]
    // let c = flatArrayAndShape<float32> [[2.f; 3.f; 7.f]; [4.f; 5.f; 6.f]]
    // printfn "%A" a
    // printfn "%A" b
    // printfn "%A" c

    // let t1 = Tensor.Create(2.f)
    // let t2 = Tensor.Create([3.f;4.f;5.f])
    // let t3 = Tensor.Create([[1.f; 2.f]; [3.f; 4.f]])
    // printfn "t1 %A" t1
    // printfn "t2 %A" t2
    // printfn "t3 %A" t3
    // printfn "t1 + t2 %A" (t1 + t2)
    // printfn "t1 + t3 %A" (t1 + t3)

    let f (x:Tensor) =
        x + 2.f

    // let x = Tensor.Create([1.f; 2.f; 3.f])
    // let z, z' = DiffSharp.grad' (fun t -> t.Sum()) x
    // printfn "%A" x
    // printfn "%A" z
    // printfn "%A" z'

    // let a = Tensor.Create([[[1., 2., 3.], [4., 5., 6.]]])
    // printfn "%A" a

    let a = Normal(Tensor.Create(0.), Tensor.Create(1.))
    printfn "%A" (a.Sample())
    printfn "%A" (a.Sample(2))

    let b = Normal(Tensor.Create([0.;10.]), Tensor.Create([1.;1.]))
    printfn "%A" (b.Sample())
    printfn "%A" (b.Sample(2))

    // let a = Tensor.RandomNormal([||])
    // let b = Tensor.RandomNormal([||])
    // let c = Tensor.RandomNormal([||])
    // let d = Tensor.Stack([|a; b; c|])
    // printfn "%A %A" a a.Shape
    // printfn "%A %A" b b.Shape
    // printfn "%A %A" c c.Shape
    // printfn "%A %A" d d.Shape

    0 // return an integer exit code
