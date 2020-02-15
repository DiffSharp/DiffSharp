// Learn more about F# at http://fsharp.org

open System
open System.Collections.Generic
open DiffSharp
open DiffSharp.Util
open DiffSharp.Distributions
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Backend.None

// type FeedForwardNet() =
//     inherit Model()
//     let fc1 = Linear(2, 64)
//     let fc2 = Linear(64, 1)
//     do base.AddParameters(["fc1", fc1; "fc2", fc2])
//     override l.Forward(x) =
//         x |> fc1.Forward |> Tensor.LeakyRelu |> fc2.Forward |> Tensor.LeakyRelu



[<EntryPoint>]
let main _argv =
    printfn "Hello World from F#!"

    DiffSharp.Seed(12)
    // DiffSharp.NestReset()
    // // let model = Linear(2, 1)
    // let model = FeedForwardNet()
    // let optimizer = SGD(model, Tensor.Create(0.01))
    // printfn "%A" model.Parameters.Tensors
    // let data = Tensor.Create([[0.;0.;0.];[0.;1.;1.];[1.;0.;1.];[1.;1.;0.]])
    // let x = data.[*,0..1]
    // let y = data.[*,2..]
    // printfn "%A" x
    // printfn "%A" y

    // for i=0 to 1000 do
    //     model.ReverseDiff()
    //     let o = model.Forward(x).View(-1)
    //     let loss = Tensor.MSELoss(o, y)
    //     printfn "prediction: %A, loss: %A" (o.NoDiff()) (loss.NoDiff())
    //     loss.Reverse()
    //     optimizer.Step()

    // printfn "%A" model.Parameters.Tensors
    // let a : Dictionary<string, Tensor> = Dictionary()
    // a.["test"] <- Tensor.Create([1;2;3])
    // printfn "%A" a
    // // model.NoDiff()
    // a.["test"] <- Tensor.Create([1;2;4])
    // printfn "%A" a
    // // printfn "%A" model.Parameters

    // t1: input, NxCxI (batchSize x numChannels, inputLength)
    // t2: filters, KxCxF (numKernels x numChannels, kernelLength)

    // let t1 = Tensor.RandomNormal([|3; 4; 5|])
    // let t2 = Tensor.RandomNormal([|2; 4; 3|])
    // let t3 = Tensor.Conv1D(t1, t2)

    let t1 = Tensor.Create([[[0.3460; 0.4414; 0.2384; 0.7905; 0.2267];
                             [0.5161; 0.9032; 0.6741; 0.6492; 0.8576];
                             [0.3373; 0.0863; 0.8137; 0.2649; 0.7125];
                             [0.7144; 0.1020; 0.0437; 0.5316; 0.7366]];

                            [[0.9871; 0.7569; 0.4329; 0.1443; 0.1515];
                             [0.5950; 0.7549; 0.8619; 0.0196; 0.8741];
                             [0.4595; 0.7844; 0.3580; 0.6469; 0.7782];
                             [0.0130; 0.8869; 0.8532; 0.2119; 0.8120]];

                            [[0.5163; 0.5590; 0.5155; 0.1905; 0.4255];
                             [0.0823; 0.7887; 0.8918; 0.9243; 0.1068];
                             [0.0337; 0.2771; 0.9744; 0.0459; 0.4082];
                             [0.9154; 0.2569; 0.9235; 0.9234; 0.3148]]])
    let t2 = Tensor.Create([[[0.4941; 0.8710; 0.0606];
                             [0.2831; 0.7930; 0.5602];
                             [0.0024; 0.1236; 0.4394];
                             [0.9086; 0.1277; 0.2450]];

                            [[0.5196; 0.1349; 0.0282];
                             [0.1749; 0.6234; 0.5502];
                             [0.7678; 0.0733; 0.3396];
                             [0.6023; 0.6546; 0.3439]]])

    let t3 = Tensor.Conv1D(t1, t2, padding=1, stride=2)
    let _t3Correct = Tensor.Create([[[2.8516; 2.0732; 2.6420];
                                     [2.3239; 1.7078; 2.7450]];
    
                                    [[3.0127; 2.9651; 2.5219];
                                     [3.0899; 3.1496; 2.4110]];
    
                                    [[3.4749; 2.9038; 2.7131];
                                     [2.7692; 2.9444; 3.2554]]])

    printfn "t1 %A" t1.Shape
    printfn "t2 %A" t2.Shape
    printfn "t3 %A" t3.Shape

    printfn "t1 %A" t1
    printfn "t2 %A" t2
    printfn "t3 %A" t3
    // printfn "t3Correct %A" t3Correct

    // let a = 7
    // let b = 2
    // let c = (float a) / (float b) |> ceil |> int
    // printfn "%A" c

    0 // return an integer exit code
