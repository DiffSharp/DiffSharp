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
let main argv =
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

    let fwdx = Tensor.Create([[[  0.1264;   5.3183;   6.6905; -10.6416];
                             [ 13.8060;   4.5253;   2.8568;  -3.2037];
                             [ -0.5796;  -2.7937;  -3.3662;  -1.3017]];

                            [[ -2.8910;   3.9349;  -4.3892;  -2.6051];
                             [  4.2547;   2.6049;  -9.8226;  -5.4543];
                             [ -0.9674;   1.0070;  -4.6518;   7.1702]]])
    let fwdx = fwdx.ForwardDiff(Tensor.Create([[[-4.3197; -6.5898; -6.2003;  2.1058];
                             [ 7.0684; -3.7964;  4.4218;  3.9533];
                             [-7.1559; -7.6799; -9.5234; -3.9351]];

                            [[-0.2089; -7.8695;  6.5383;  5.1090];
                             [-3.8272;  7.6264;  6.8205;  5.7346];
                             [ 6.5570;  7.7248;  6.3494; -2.9007]]]))

    let fwdy = Tensor.Create([[[ 4.0332e+00;  6.3036e+00];
                             [ 8.4410e+00; -5.7543e+00];
                             [-5.6937e-03; -6.7241e+00]];

                            [[-2.2619e+00;  1.2082e+00];
                             [-1.2203e-01; -4.9373e+00];
                             [-4.1881e+00; -3.4198e+00]]])
    let fwdy = fwdy.ForwardDiff(Tensor.Create([[[-1.5107; -0.0610];
                             [-0.2609;  5.9220];
                             [ 2.8221; -5.7314]];

                            [[ 5.0064;  3.8631];
                             [-4.6264; -7.9380];
                             [ 8.2204; -1.9833]]]))

    let fwdz = Tensor.Conv1D(fwdx, fwdy, padding=0, stride=1)
    let fwdzCorrect = Tensor.Create([[[ 143.3192;  108.0332;   11.2241];
                                     [  -5.9062;    4.6091;    6.0273]];

                                    [[  27.3032;   97.9855; -133.8372];
                                     [  -1.4792;   45.6659;   29.8705]]])
    let fwdzd = fwdz.Derivative
    let fwdzdCorrect = Tensor.Create([[[ 111.2865;  -40.3692;   -1.8573];
                                     [   -1.9154;   43.3470;   29.3626]];

                                    [[ -168.6758;  -43.1578;   25.4470];
                                     [ -149.6851;   23.1963;  -50.1932]]])

    printfn "fwdx %A" fwdx.Shape
    printfn "fwdy %A" fwdy.Shape
    printfn "fwdz %A" fwdz.Shape

    // printfn "t1 %A" t1
    // printfn "t2 %A" t2
    printfn "fwdz %A" fwdz
    printfn "fwdzCorrect %A" fwdzCorrect

    printfn "fwdzd %A" fwdzd
    printfn "fwdzdCorrect %A" fwdzdCorrect

    // printfn "t3Correct %A" t3Correct

    // let a = 7
    // let b = 2
    // let c = (float a) / (float b) |> ceil |> int
    // printfn "%A" c

    let mirrorCoordinates (coordinates:int[]) (shape:int[]) (mirrorDims:int[]) =
        if coordinates.Length <> shape.Length then invalidOp <| sprintf "Expecting coordinates and shape of the same dimension, received %A, %A" coordinates.Length shape.Length
        let result = Array.copy coordinates
        for d=0 to coordinates.Length-1 do
            if mirrorDims |> Array.contains d then
                result.[d] <- abs (coordinates.[d] - shape.[d] + 1)
        result

    let a = Tensor.Create([[0; 1]; [2; 3]])
    printfn "a %A" a
    let b = a.Flip([|1|])
    printfn "b %A" b
    let c = b.Flip([|1|])
    printfn "c %A" c

    // let duplicates l =
    //    l |> List.ofSeq
    //    |> List.groupBy id
    //    |> List.choose ( function
    //           | _, x::_::_ -> Some x
    //           | _ -> None )

    // let hasDuplicates l =
    //     (duplicates l) |> List.isEmpty |> not

    // let a = [|1; 2; 3; 0|]
    // printfn "%A" (duplicates a)        
    // printfn "%A" (hasDuplicates a)

    0 // return an integer exit code
