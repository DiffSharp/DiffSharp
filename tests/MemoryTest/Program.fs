open System
open DiffSharp
//open TorchSharp

[<EntryPoint>]
let main _ =
    //dsharp.config(backend=Backend.Reference) // Uses around 23 MB
    dsharp.config(backend=Backend.Torch) // Uses around 4GB

    printfn "Press any key to continue..."
    Console.ReadKey() |> ignore

    for i in 0..1024 do
        let x = dsharp.zeros([1024; 1024])
        //let x = TorchSharp.Tensor.FloatTensor.Zeros([|1024L;1024L|])
        //x.Dispose()
        printfn "%A" i
        System.GC.Collect()

    printfn "Press any key to continue..."
    Console.ReadKey() |> ignore

    0 // return an integer exit code
