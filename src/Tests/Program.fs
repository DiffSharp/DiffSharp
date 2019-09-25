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
    

    let t = Tensor.Random([2; 2])
    printfn "%A" t
    let tt = t.[0, *]
    printfn "%A" tt
    let tt = t.[1, *]
    printfn "%A" tt    
    let tt = t.[0, 1]
    printfn "%A" tt

    // let tt = t.[1]
    // let tt = t.[0, 1]
    // let tt = t.[0, 0, 1]
    // let tt = t.[0,*]
    // let tt = t.[*,*]
    // let tt = t.[*,0]
    // let tt = t.[0,*]

    // let a = array2D [[1;2];[3;4];[5;6]]
    // let b = a.[1..,*]
    // printfn "%A" b

    // let r = RawTensorFloat32CPUBase.Create([[1.;2.];[3.;4.]])
    // let i = r.[0]
    // let i = r.[0, 1]
    // let i = r.[*, 1]
    // let i = r.[1.., 1]
    // let i = r.[..1, 1]
    // printfn "%A" r
    // printfn "%A" i

    0 // return an integer exit code
