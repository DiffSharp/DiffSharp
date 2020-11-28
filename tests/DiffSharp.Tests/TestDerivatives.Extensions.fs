// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp

#nowarn "0058"

[<TestFixture>]
module TestOps =

    type Tensor with
        static member mull (a, b) = 
            Tensor.Op
                { new BinaryOp() with 
                    member _.ComputeRaw(a,b) = a.MulTT(b)
                    // f(a,b)               = a*b
                    // Jacobian             = [b; a]
                    // Jacobian * [ad;bd].t = b*ad + a*bd
                    member _.Forward(fab, a, ad, b, bd) = Tensor.mull(b, ad) + Tensor.mull(a, bd)
                    member _.ForwardA(fab, a, ad, b) = Tensor.mull(b, ad)
                    member _.ForwardB(fab, a, b, bd) = Tensor.mull(a, bd)
                    // Jacobian             = [b; a]
                    // Jacobian.t * td.t    = [b*td; a*td]
                    member _.ReverseA(fab, a, b, td) = Tensor.mull(b, td)
                    member _.ReverseB(fab, a, b, td) = Tensor.mull(a, td)
                }
                (a, b)

    type Tensor with
        member a.sinn() = 
            Tensor.Op
               { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.SinT()
                    member _.Forward(fab,a,ad) = a.coss() * ad
                    member _.Reverse(fab,a,ad) = a.coss() * ad }
               a

        member a.coss() = 
            Tensor.Op
               { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.CosT()
                    member _.Forward(fa,a,ad) = -a.sinn() * ad 
                    member _.Reverse(fab,a,ad) = -a.sinn() * ad }
                a

    type Tensor with

        member a.poww(b) = 
            Tensor.Op
                { new BinaryOp() with 
                    // f(a,b) = a^b
                    // df(a,b)/da = b * a^(b-1)
                    // df(a,b)/db = a^b * log(a)
                    // J = [ df(a,b)/da ; df(a,b)/db ]
                    // J * [da;db]
                    //    = b * a^(b-1) * da + a^b * log(a) * db
                    //    = a^(b-1) * (b * da + a * log(a) * db)
                    // JTA = [ df(a,b)/da ; df(a,b)/db * db ]
                    // JT * td
                    member _.ComputeRaw(a,b) = a.PowTT(b)
                    member _.Forward(fa, a, ad, b, ab) = (a ** (b - 1.)) * (b * ad + a * log a * ab)
                    member _.ForwardA(fa, a, ad, b) = (a ** (b - 1.)) * b * ad
                    member _.ForwardB(fa, a, b, bd) = fa * log a * bd
                    member _.ReverseA(fab, a, b, td) = (a ** (b - 1.)) * b * td
                    member _.ReverseB(fab, a, b, td) = (a ** b) * log a * td }
                (a,b)

    type Tensor with
        member a.conv1dd(b:Tensor, ?stride:int, ?padding:int) : Tensor = 
            let stride = defaultArg stride 1
            let padding = defaultArg padding 0
            Tensor.Op
                { new BinaryOp() with 
                    member _.ComputeRaw(a,b) = a.Conv1D(b, stride, padding)
                    member _.Forward(fab, a, ad, b, bd) = ad.conv1dd(b, stride, padding) + a.conv1dd(bd, stride, padding)
                    member _.ForwardA(fab, a, ad, b) = ad.conv1dd(b, stride, padding)
                    member _.ForwardB(fab, a, b, bd) = a.conv1dd(bd, stride, padding)
                    member _.ReverseA(fab, a, b, td) = 
                        let batchSize = td.shape.[0]
                        let outputChannels = td.shape.[1]
                        let inputChannels = a.shape.[1]
                        let kernelLength = b.shape.[2]
                        let mutable tderivative = td
                        if stride > 1 then
                            tderivative <- tderivative.dilate([|1;1;stride|])
                        let bFlipped = b.flip([|2|])
                        let mutable aderivative = a.zerosLike()
                        for k=0 to outputChannels-1 do
                            let b = bFlipped.[k].view([|inputChannels; 1; kernelLength|])
                            let dBounds: int[,] = array2D [[0; batchSize-1; 1]; [k; k; 1]; [0; tderivative.shape.[2]-1; 1]]
                            let d = tderivative.GetSlice(dBounds).view([|batchSize; 1; -1|])
                            let mutable c = d.conv1dd(b, padding=kernelLength-1)
                            if padding > 0 then
                                let cBounds = array2D [[0; batchSize-1; 1]; [0; inputChannels-1; 1]; [padding; c.shape.[2]-1-padding; 1]]
                                c <- c.GetSlice(cBounds).view([|batchSize; inputChannels; -1|])
                            aderivative <- aderivative + c
                        aderivative

                    member _.ReverseB(fab, a, b, td) = 
                        let batchSize = td.shape.[0]
                        let outputChannels = td.shape.[1]
                        let inputChannels = a.shape.[1]
                        let inputLength = a.shape.[2]
                        let kernelLength = b.shape.[2]
                        let mutable tderivative = td
                        if stride > 1 then
                            tderivative <- tderivative.dilate([|1;1;stride|])
                        let mutable bderivative = b.zerosLike()
                        for n=0 to batchSize-1 do
                            let aa = a.[n].view([|inputChannels; 1; inputLength|]) 
                            let d = tderivative.[n]
                            for k=0 to outputChannels-1 do
                                let dd = d.[k].view([|1; 1; tderivative.shape.[2]|])
                                let c = aa.conv1dd(dd, padding=padding).view([|1; inputChannels; kernelLength|])
                                bderivative <- bderivative.addSlice([|k; 0; 0|], c)
                        bderivative }
                (a,b)

    let CompareUnaryOps op1 op2 inp dinp dout = 
        let fwdx = dsharp.tensor(inp).forwardDiff(dsharp.tensor(dinp))
        let fwdz : Tensor = op1 fwdx
        let fwdzCorrect : Tensor = op2 fwdx
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = fwdzCorrect.derivative

        let x = dsharp.tensor(inp)
        let revx = x.reverseDiff()
        let revxCorrect = x.reverseDiff()
        let revz = op1 revx
        let revzCorrect = op2 revxCorrect
        revz.reverse(dsharp.tensor(dout))
        revzCorrect.reverse(dsharp.tensor(dout))
        let revxd = revx.derivative
        let revxdCorrect = revxCorrect.derivative

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    let CompareBinaryOps op1 op2 inpx inpy dinpx dinpy dout = 
        let fwdx = dsharp.tensor(inpx).forwardDiff(dsharp.tensor(dinpx))
        let fwdy = dsharp.tensor(inpy).forwardDiff(dsharp.tensor(dinpy))
        let fwdz : Tensor = op1 (fwdx, fwdy)
        let fwdzCorrect : Tensor = op2 (fwdx, fwdy)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = fwdzCorrect.derivative

        let x = dsharp.tensor(inpx)
        let y = dsharp.tensor(inpy)
        let revx = x.reverseDiff()
        let revy = y.reverseDiff()
        let revxCorrect = x.reverseDiff()
        let revyCorrect = y.reverseDiff()
        let revz = op1 (revx, revy)
        let revzCorrect = op2 (revxCorrect, revyCorrect)
        revz.reverse(dsharp.tensor(dout))
        revzCorrect.reverse(dsharp.tensor(dout))
        let revxd = revx.derivative
        let revyd = revy.derivative
        let revxdCorrect = revxCorrect.derivative
        let revydCorrect = revyCorrect.derivative

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    let ``test mull extension``() = 
        CompareBinaryOps 
            (fun (a,b) -> Tensor.mull(a, b))
            (fun (a,b) -> a * b)
            [5.; 6.; 7.]
            [2.; 2.; 3.]
            [5.; 6.; 7.]
            [2.; 2.; 3.]
            [5.; 15.; 25.]

    [<Test>]
    let ``test sinn extension``() = 
        CompareUnaryOps 
            (fun t -> t.sinn())
            (fun t -> t.sin())
            [0.9473; 1.4891; 0.2015; 0.5818; 0.8439]
            [1.7164; 0.2905; 1.4872; 1.2580; 0.5778]
            [5.; 5.; 5.; 5.; -5.]

    [<Test>]
    let ``test poww extension``() = 
        CompareBinaryOps 
            (fun (t,u) -> t.poww(u))
            (fun (t,u) -> t.pow(u))
            [0.9473; 1.4891; 0.2015; 0.5818; 0.8439]
            [0.9473; 1.4891; 0.2015; 0.5818; 0.8439]
            [1.7164; 0.2905; 1.4872; 1.2580; 0.5778]
            [1.7164; 0.2905; 1.4872; 1.2580; 0.5778]
            [5.; 5.; 5.; 5.; 1.]


    [<Test>]
    let ``test conv1dd extension``() = 
        CompareBinaryOps
            (fun (a,b) -> a.conv1d(b))
            (fun (a,b) -> a.conv1dd(b))
            [[[  0.1264;   5.3183;   6.6905; -10.6416];
              [ 13.8060;   4.5253;   2.8568;  -3.2037];
              [ -0.5796;  -2.7937;  -3.3662;  -1.3017]];

             [[ -2.8910;   3.9349;  -4.3892;  -2.6051];
              [  4.2547;   2.6049;  -9.8226;  -5.4543];
              [ -0.9674;   1.0070;  -4.6518;   7.1702]]]

            [[[ 4.0332e+00;  6.3036e+00];
              [ 8.4410e+00; -5.7543e+00];
              [-5.6937e-03; -6.7241e+00]];

             [[-2.2619e+00;  1.2082e+00];
              [-1.2203e-01; -4.9373e+00];
              [-4.1881e+00; -3.4198e+00]]]

            [[[-4.3197; -6.5898; -6.2003;  2.1058];
              [ 7.0684; -3.7964;  4.4218;  3.9533];
              [-7.1559; -7.6799; -9.5234; -3.9351]];

             [[-0.2089; -7.8695;  6.5383;  5.1090];
              [-3.8272;  7.6264;  6.8205;  5.7346];
              [ 6.5570;  7.7248;  6.3494; -2.9007]]]

            [[[-1.5107; -0.0610];
              [-0.2609;  5.9220];
              [ 2.8221; -5.7314]];

             [[ 5.0064;  3.8631];
              [-4.6264; -7.9380];
              [ 8.2204; -1.9833]]]

            [[[ 4.5763;  2.7538;  2.0173];
              [-2.7543;  7.9257; -1.3670]];

             [[ 1.7997; -1.2354;  4.6313];
              [-4.0646;  0.0384;  4.1437]]]


    [<Test>]
    let TestDerivativePowTT () =
        CompareBinaryOps 
            (fun (a,b) -> a.poww(b))
            (fun (a,b) -> a ** b)
            [5.; 6.; 7.]
            [2.; 2.; 3.]
            [5.; 6.; 7.]
            [2.; 2.; 3.]
            [5.; 15.; 25.]
    
