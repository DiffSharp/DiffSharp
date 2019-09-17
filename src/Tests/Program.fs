// Learn more about F# at http://fsharp.org

open System
open DiffSharp

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    // let a = toFlatArrayAndShape<float32> 2.f
    // let b = toFlatArrayAndShape<float32> [2.f; 3.f]
    // let c = toFlatArrayAndShape<float32> [[2.f; 3.f; 7.f]; [4.f; 5.f; 6.f]]
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

    let x = Tensor.Create([[1.f; 2.f; 3.f]; [4.f; 5.f; 6.f]])
    let y, y' = DiffSharp.diff' (fun t -> t.Sum()) x
    let z, z' = DiffSharp.grad' (fun t -> t.Sum()) x
    printfn "%A" x
    printfn "%A" y
    printfn "%A" y'
    printfn "%A" z
    printfn "%A" z'

    0 // return an integer exit code
