// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.NN
open DiffSharp.RawTensor


type Model() =
    inherit Layer()
    let fc1 = Linear(2, 2)
    let fc2 = Linear(2, 1)
    do base.AddParameters(["fc1", fc1; "fc2", fc2])
    override l.Forward(x) =
        x 
        |> fc1.Forward |> Tensor.Relu
        |> fc2.Forward |> Tensor.Relu

let optimize (model:Layer) (lr:Tensor) =
    let update k (p:Parameter) = printfn "updating %A" k; p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative
    model.Map(update)

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    // DiffSharp.Seed(123)
    // DiffSharp.NestReset()
    // let model = Model()
    // model.ReverseDiff()
    // let x = Tensor.Random([1; 2])
    // let o = model.Forward(x)
    // o.Reverse(Tensor.Random([1; 1]))

    // printfn "\n%A\n" model.Parameters
    // optimize model (Tensor.Create(0.1))
    // printfn "\n%A\n" model.Parameters
    
    let a = RawTensorFloat32CPUBase.Create([[1.;2.;3.];[4.;5.;6.];[7.;8.;9.]])
    let b = RawTensorFloat32CPUBase.Create([[10.;20.];[30.;40.]])
    printfn "%A" a
    printfn "%A" b
    let c = a.AddTTSlice([|0;0|], b)
    printfn "%A" c


    0 // return an integer exit code
