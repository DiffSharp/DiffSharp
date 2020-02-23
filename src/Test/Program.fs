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

    DiffSharp.Seed(12)


    // let t1 = RawTensorFloat32CPU.Create([[[[3.4798e-01; 2.5763e-01; 8.5592e-02; 5.5208e-01];
    //       [5.3767e-01; 2.8232e-01; 2.7177e-01; 6.5287e-01];
    //       [9.1777e-01; 1.5579e-01; 1.9189e-01; 5.3538e-01];
    //       [6.1284e-01; 8.4073e-01; 3.1288e-02; 4.9212e-01]];

    //      [[1.0122e-01; 2.1809e-01; 2.4404e-01; 8.2020e-01];
    //       [2.4450e-01; 1.4915e-01; 4.2428e-02; 3.1314e-01];
    //       [8.1286e-02; 3.8522e-01; 6.1468e-01; 9.7954e-01];
    //       [1.7899e-01; 4.9997e-01; 9.7368e-01; 9.9865e-01]]];


    //     [[[6.6835e-04; 1.6774e-01; 3.2205e-01; 4.9608e-01];
    //       [7.9847e-01; 5.9450e-01; 7.4723e-01; 7.1045e-02];
    //       [8.5804e-01; 9.8996e-01; 1.4214e-01; 5.7838e-01];
    //       [8.1200e-01; 1.3486e-01; 7.7813e-01; 9.3069e-01]];

    //      [[5.9376e-01; 2.1951e-01; 2.6291e-01; 2.2962e-01];
    //       [3.5108e-01; 5.3628e-01; 1.8297e-01; 6.1111e-01];
    //       [3.6474e-01; 4.0430e-01; 3.3405e-01; 9.4802e-02];
    //       [2.2734e-01; 4.7429e-01; 4.7744e-01; 6.0280e-01]]]])
    // let t2 = RawTensorFloat32CPU.Create([[[[0.5665; 0.7224; 0.5928];
    //                                       [0.8205; 0.2581; 0.8510];
    //                                       [0.7683; 0.7726; 0.3186]];

    //                                      [[0.5755; 0.9148; 0.3518];
    //                                       [0.8185; 0.0672; 0.9901];
    //                                       [0.1678; 0.4778; 0.6461]]];


    //                                     [[[0.8326; 0.2475; 0.4060];
    //                                       [0.2546; 0.6160; 0.9317];
    //                                       [0.1285; 0.3622; 0.3116]];

    //                                      [[0.1366; 0.2714; 0.5941];
    //                                       [0.1808; 0.8697; 0.2536];
    //                                       [0.1550; 0.7463; 0.0313]]];


    //                                     [[[0.8763; 0.2988; 0.6834];
    //                                       [0.4062; 0.3982; 0.5589];
    //                                       [0.0075; 0.8414; 0.6794]];

    //                                      [[0.0649; 0.0053; 0.8807];
    //                                       [0.1245; 0.7355; 0.3074];
    //                                       [0.8194; 0.1950; 0.0755]]]])

    // let t3 = t1.Conv2D(t2, [|1; 2|], [|0; 2|])
    // printfn "%A" t1.Shape
    // printfn "%A" t2.Shape
    // printfn "%A" t3.Shape

    // printfn "\n%A" t3

    // let a = Tensor.Create([[[[1.]]; 
    //                         [[2.]]; 
    //                         [[3.]]]; 
    //                        [[[4.]]; 
    //                         [[5.]]; 
    //                         [[6.]]]])

    // printfn "ashape %A" a.Shape
    // printfn "a %A" a

    // let b = a.[0]
    // printfn "\nbshape %A" b.Shape
    // printfn "b %A" b

    // let c = b.Unsqueeze(0)
    // printfn "\ncshape %A" c.Shape
    // printfn "c %A" c

    // let a = [1; 2]
    // let b = (fun aa -> Array.ofSeq aa) a


    // let t = Tensor.Create([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
    // let tSum0 = t.Sum(0)
    // let tSum0Correct = Tensor.Create([[14.0f, 16.0f, 18.0f, 20.0f], [22.0f, 24.0f, 26.0f, 28.0f], [30.0f, 32.0f, 34.0f, 36.0f]])
    // let tSum1 = t.Sum(1)
    // let tSum1Correct = Tensor.Create([[15.0f, 18.0f, 21.0f, 24.0f], [51.0f, 54.0f, 57.0f, 60.0f]])
    // let tSum2 = t.Sum(2)
    // let tSum2Correct = Tensor.Create([[10.0f, 26.0f, 42.0f], [58.0f, 74.0f, 90.0f]])

    // let t4 = Tensor.Create([[[[1.]]; 
    //                          [[2.]]; 
    //                          [[3.]]]; 
    //                         [[[4.]]; 
    //                          [[5.]]; 
    //                          [[6.]]]])
    // let t4s1 = t4.[*,0]
    // let t4s2 = t4.[*,0,*,*]

    // printfn "t4shape\n%A" t4.Shape
    // printfn "t4\n%A" t4
    // printfn "\nt4s1shape\n%A" t4s1.Shape
    // printfn "t4s1\n%A" t4s1
    // printfn "\nt4s2shape\n%A" t4s2.Shape
    // printfn "t4s2\n%A" t4s2
    0 // return an integer exit code
