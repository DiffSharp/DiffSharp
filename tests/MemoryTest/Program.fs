open System
open DiffSharp

[<EntryPoint>]
let main _ =
    //dsharp.config(backend=Backend.Reference)
    dsharp.config(backend=Backend.Torch)
    dsharp.seed(1)

    printfn "Press any key to continue..."
    Console.ReadKey() |> ignore

    for i in 0..1024 do
        let _ = dsharp.randn([1024; 1024])
        printfn "%A" i
        System.GC.Collect()

    printfn "Press any key to continue..."
    Console.ReadKey() |> ignore

    0 // return an integer exit code
