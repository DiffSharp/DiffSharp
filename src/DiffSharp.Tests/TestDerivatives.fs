namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDerivatives () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestDerivativeAddTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdy = Tensor.Create([5.; 6.; 7.]).ForwardDiff(Tensor.Create([2.; 2.; 3.]))
        let fwdz = fwdx + fwdy
        let fwdzCorrect = Tensor.Create([6.; 8.; 10.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([4.; 5.; 7.])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.; 7.]).ReverseDiff()
        let revz = revx + revy
        let revzCorrect = Tensor.Create([6.; 8.; 10.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.; 5.; 5.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([5.; 5.; 5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    [<Test>]
    member this.TestDerivativeAddT2T1 () =
        let fwdx = Tensor.Create([[1.; 2.]; [3.; 4.]]).ForwardDiff(Tensor.Create([[2.; 3.]; [4.; 5.]]))
        let fwdy = Tensor.Create([5.; 6.]).ForwardDiff(Tensor.Create([2.; 3.]))
        let fwdz = fwdx + fwdy
        let fwdzCorrect = Tensor.Create([[6.; 8.]; [8.; 10.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[4.; 6.]; [6.; 8.]])

        let revx = Tensor.Create([[1.; 2.]; [3.; 4.]]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.]).ReverseDiff()
        let revz = revx + revy
        let revzCorrect = Tensor.Create([[6.; 8.]; [8.; 10.]])
        revz.Reverse(Tensor.Create([[2.; 3.]; [4.; 5.]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[2.; 3.]; [4.; 5.]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([6.; 8.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    // TODO: add test for AddTTConst
    // TODO: add test for AddTT0
    // TODO: add test for AddTT0Const
    // TODO: add test for AddTConstT0
    // TODO: add test for AddT2T1Const
    // TODO: add test for AddT2ConstT1

    [<Test>]
    member this.TestDerivativeSubTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdy = Tensor.Create([5.; 6.; 7.]).ForwardDiff(Tensor.Create([2.; 2.; 3.]))
        let fwdz = fwdx - fwdy
        let fwdzCorrect = Tensor.Create([-4.; -4.; -4.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 1.; 1.])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.; 7.]).ReverseDiff()
        let revz = revx - revy
        let revzCorrect = Tensor.Create([-4.; -4.; -4.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.; 5.; 5.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([-5.; -5.; -5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    // TODO: add test for SubTTConst
    // TODO: add test for SubTConstT
    // TODO: add test for SubT0T
    // TODO: add test for SubT0TConst
    // TODO: add test for SubT0ConstT
    // TODO: add test for SubTT0
    // TODO: add test for SubTT0Const
    // TODO: add test for SubTConstT0

    [<Test>]
    member this.TestDerivativeMulTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdy = Tensor.Create([5.; 6.; 7.]).ForwardDiff(Tensor.Create([2.; 2.; 3.]))
        let fwdz = fwdx * fwdy
        let fwdzCorrect = Tensor.Create([5.; 12.; 21.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([12.; 22.; 37.])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.; 7.]).ReverseDiff()
        let revz = revx * revy
        let revzCorrect = Tensor.Create([5.; 12.; 21.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([25.; 30.; 35.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([5.; 10.; 15.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    // TODO: add test for MulTTConst
    // TODO: add test for MulTT0
    // TODO: add test for MulTConstT0
    // TODO: add test for MulTT0Const

    [<Test>]
    member this.TestDerivativeDivTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdy = Tensor.Create([5.; 6.; 7.]).ForwardDiff(Tensor.Create([2.; 2.; 3.]))
        let fwdz = fwdx / fwdy
        let fwdzCorrect = Tensor.Create([0.2; 0.333333; 0.428571])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.32; 0.388889; 0.387755])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.; 7.]).ReverseDiff()
        let revz = revx / revy
        let revzCorrect = Tensor.Create([0.2; 0.333333; 0.428571])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([1.; 0.833333; 0.714286])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([-0.2; -0.277778; -0.306122])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))
    
    // TODO: add test for DivTTConst
    // TODO: add test for DivTConstT
    // TODO: add test for DivT0T
    // TODO: add test for DivT0TConst
    // TODO: add test for DivT0ConstT
    // TODO: add test for DivTT0
    // TODO: add test for DivTT0Const
    // TODO: add test for DivTConstT0

    [<Test>]
    member this.TestDerivativePowTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdy = Tensor.Create([5.; 6.; 7.]).ForwardDiff(Tensor.Create([2.; 2.; 3.]))
        let fwdz = fwdx ** fwdy
        let fwdzCorrect = Tensor.Create([1.; 64.; 2187.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([10.; 664.723; 27620.])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.; 7.]).ReverseDiff()
        let revz = revx ** revy
        let revzCorrect = Tensor.Create([1.; 64.; 2187.])
        revz.Reverse(Tensor.Create([5.; 15.; 25.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([25.; 2880.; 127575.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([0.; 665.421; 60066.6])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect,0.1))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect,0.1))
        Assert.True(revz.ApproximatelyEqual(revzCorrect,0.1))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect,0.1))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect,0.1))
    
    // TODO: add test for PowTTConst
    // TODO: add test for PowTConstT
    // TODO: add test for PowT0T
    // TODO: add test for PowT0TConst
    // TODO: add test for PowT0ConstT
    // TODO: add test for PowTT0
    // TODO: add test for PowTT0Const
    // TODO: add test for PowTConstT0

    [<Test>]
    member this.TestDerivativeConv1D () =
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
        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))

    //TODO: add test for Conv1DTTConst
    //TODO: add test for Conv1DTConstT

    [<Test>]
    member this.TestDerivativeConv1Dp2s2 () =
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

        let fwdz = Tensor.Conv1D(fwdx, fwdy, padding=2, stride=2)
        let fwdzCorrect = Tensor.Create([[[   0.0000;  143.3192;   11.2241;    0.0000];
                                          [   0.0000;   -5.9062;    6.0273;    0.0000]];

                                         [[   0.0000;   27.3032; -133.8372;    0.0000];
                                          [   0.0000;   -1.4792;   29.8705;    0.0000]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[   0.0000;  111.2865;   -1.8573;    0.0000];
                                          [    0.0000;   -1.9154;   29.3626;    0.0000]];

                                         [[   0.0000;  -168.6758;   25.4470;    0.0000];
                                          [   0.0000;  -149.6851;  -50.1932;    0.0000]]])
        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))

    [<Test>]
    member this.TestDerivativeMatMulT2T2 () =
        let fwdx = Tensor.Create([[6.2381; 0.0393; 8.2364; 3.9906; 6.2291];
            [9.8762; 3.2263; 6.2866; 4.7111; 0.0652];
            [3.5832; 7.9801; 1.9854; 4.4965; 4.1712]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[4.6453; 8.4388; 4.6549; 9.5680; 1.5756];
            [3.2066; 4.2429; 2.2028; 9.1037; 3.4022];
            [4.2324; 4.5508; 3.4755; 2.7196; 5.5344]]))
        let fwdy = Tensor.Create([[4.4220; 3.7293];
            [6.1928; 2.1446];
            [0.0525; 1.2494];
            [7.5281; 1.4816];
            [5.0328; 2.2756]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[1.4749; 9.7608];
            [3.6599; 7.9553];
            [3.5503; 1.3757];
            [8.3172; 6.6748];
            [2.2959; 0.6784]]))
        let fwdz = Tensor.MatMul(fwdx, fwdy)
        let fwdzCorrect = Tensor.Create([[ 89.6516; 53.7260];
            [ 99.7751; 58.7331];
            [120.2113; 49.1116]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[239.0819; 162.3930];
            [214.2522; 207.2430];
            [183.9220; 180.5424]])

        let revx = Tensor.Create([[6.2381; 0.0393; 8.2364; 3.9906; 6.2291];
            [9.8762; 3.2263; 6.2866; 4.7111; 0.0652];
            [3.5832; 7.9801; 1.9854; 4.4965; 4.1712]]).ReverseDiff()
        let revy = Tensor.Create([[4.4220; 3.7293];
            [6.1928; 2.1446];
            [0.0525; 1.2494];
            [7.5281; 1.4816];
            [5.0328; 2.2756]]).ReverseDiff()
        let revz = Tensor.MatMul(revx, revy)
        let revzCorrect = Tensor.Create([0.2; 0.333333; 0.428571])
        let revzCorrect = Tensor.Create([[ 89.6516; 53.7260];
            [ 99.7751; 58.7331];
            [120.2113; 49.1116]])
        revz.Reverse(Tensor.Create([[7.3984; 0.1849];
            [1.2520; 9.5731];
            [6.8201; 9.5221]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[33.4050; 46.2136;  0.6191; 55.9696; 37.6556];
            [41.2370; 28.2842; 12.0266; 23.6085; 28.0854];
            [65.6689; 62.6571; 12.2551; 65.4497; 55.9926]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[ 82.9549;129.8180];
            [ 58.7551;106.8801];
            [ 82.3474; 80.6097];
            [ 66.0888; 88.6534];
            [ 74.6154; 41.4950]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    //TODO: add test for MatMulT2T2Const
    //TODO: add test for MatMulT2ConstT2

    [<Test>]
    member this.TestTensorStackTs () =
        let fwdxa = Tensor.Create([1.; 2.]).ForwardDiff(Tensor.Create([10.; 20.]))
        let fwdxb = Tensor.Create([3.; 4.]).ForwardDiff(Tensor.Create([30.; 40.]))
        let fwdxc = Tensor.Create([5.; 6.]).ForwardDiff(Tensor.Create([50.; 60.]))
        let fwdz = Tensor.Stack([fwdxa;fwdxb;fwdxc])
        let fwdzCorrect = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[10.;20.];[30.;40.];[50.;60.]])

        let revxa = Tensor.Create([1.; 2.]).ReverseDiff()
        let revxb = Tensor.Create([3.; 4.]).ReverseDiff()
        let revxc = Tensor.Create([5.; 6.]).ReverseDiff()
        let revz = Tensor.Stack([revxa;revxb;revxc])
        let revzCorrect = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]])
        revz.Reverse(Tensor.Create([[10.;20.];[30.;40.];[50.;60.]]))
        let revxda = revxa.Derivative
        let revxdaCorrect = Tensor.Create([10.; 20.])
        let revxdb = revxb.Derivative
        let revxdbCorrect = Tensor.Create([30.; 40.])
        let revxdc = revxc.Derivative
        let revxdcCorrect = Tensor.Create([50.; 60.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxda, revxdaCorrect)
        Assert.AreEqual(revxdb, revxdbCorrect)
        Assert.AreEqual(revxdc, revxdcCorrect)

    // TODO: add test for UnstackT

    [<Test>]
    member this.TestDerivativeNeg () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdz = -fwdx
        let fwdzCorrect = Tensor.Create([-1.; -2.; -3.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.; -3.; -4.])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revz = -revx
        let revzCorrect = Tensor.Create([-1.; -2.; -3.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([-5.; -5.; -5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeSum () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdz = fwdx.Sum()
        let fwdzCorrect = Tensor.Create(6.)
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create(9.)

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revz = revx.Sum()
        let revzCorrect = Tensor.Create(6.)
        revz.Reverse(Tensor.Create(5.))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.; 5.; 5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeSumT2Dim0 () =
        let fwdx = Tensor.Create([[1.; 2.]; [3.; 4.]]).ForwardDiff(Tensor.Create([[2.; 3.]; [4.; 5.]]))
        let fwdz = fwdx.SumT2Dim0()
        let fwdzCorrect = Tensor.Create([4.; 6.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([6.; 8.])

        let revx = Tensor.Create([[1.; 2.]; [3.; 4.]]).ReverseDiff()
        let revz = revx.SumT2Dim0()
        let revzCorrect = Tensor.Create([4.; 6.])
        revz.Reverse(Tensor.Create([5.; 6.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[5.; 6.]; [5.; 6.]])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)        

    [<Test>]
    member this.TestDerivativeTransposeT2 () =
        let fwdx = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]]).ForwardDiff(Tensor.Create([[2.; 3.; 4.]; [10.; 20.; 30.]]))
        let fwdz = fwdx.Transpose()
        let fwdzCorrect = Tensor.Create([[1.; 4.]; [2.; 5.]; [3.; 6.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[2.; 10.]; [3.; 20.]; [4.; 30.]])

        let revx = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]]).ReverseDiff()
        let revz = revx.Transpose()
        let revzCorrect = Tensor.Create([[1.; 4.]; [2.; 5.]; [3.; 6.]])
        revz.Reverse(Tensor.Create([[5.; 5.]; [2.; 5.]; [3.; 7.]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[5.; 2.; 3.]; [5.; 5.; 7.]])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeSignT () =
        let fwdx = Tensor.Create([-1.; 0.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdz = fwdx.Sign()
        let fwdzCorrect = Tensor.Create([-1.; 0.; 1.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.])

        let revx = Tensor.Create([-1.; 0.; 3.]).ReverseDiff()
        let revz = revx.Sign()
        let revzCorrect = Tensor.Create([-1.; 0.; 1.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeFloorT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Floor()
        let fwdzCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Floor()
        let revzCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeCeilT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Ceil()
        let fwdzCorrect = Tensor.Create([1.; 1.; 1.; 1.; 1.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Ceil()
        let revzCorrect = Tensor.Create([1.; 1.; 1.; 1.; 1.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeRoundT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Round()
        let fwdzCorrect = Tensor.Create([1.; 0.; 0.; 1.; 1.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Round()
        let revzCorrect = Tensor.Create([1.; 0.; 0.; 1.; 1.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeAbsT () =
        let fwdx = Tensor.Create([-1.; 0.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdz = fwdx.Abs()
        let fwdzCorrect = Tensor.Create([1.; 0.; 3.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.; 0.; 4.])

        let revx = Tensor.Create([-1.; 0.; 3.]).ReverseDiff()
        let revz = revx.Abs()
        let revzCorrect = Tensor.Create([1.; 0.; 3.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([-5.; 0.; 5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeReluT () =
        let fwdx = Tensor.Create([-1.; -2.; 0.; 3.; 10.]).ForwardDiff(Tensor.Create([2.; 3.; 4.; 5.; 6.]))
        let fwdz = fwdx.Relu()
        let fwdzCorrect = Tensor.Create([0.; 0.; 0.; 3.; 10.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.; 5.; 6.])

        let revx = Tensor.Create([-1.; -2.; 0.; 3.; 10.]).ReverseDiff()
        let revz = revx.Relu()
        let revzCorrect = Tensor.Create([0.; 0.; 0.; 3.; 10.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.; 5.; -5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeLeakyRelu () =
        let fwdx = Tensor.Create([-1.; -2.; 0.; 3.; 10.]).ForwardDiff(Tensor.Create([2.; 3.; 4.; 5.; 6.]))
        let fwdz = fwdx.LeakyRelu()
        let fwdzCorrect = Tensor.Create([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])
        let fwdzd = fwdz.Derivative
        // TODO: behavior of derivative at 0 (where it is undefined) can be reconsidered
        // let fwdzdCorrect = Tensor.Create([0.0200; 0.0300; 0.0400; 5.; 6.])
        let fwdzdCorrect = Tensor.Create([0.0200; 0.0300; 2.02; 5.; 6.])

        let revx = Tensor.Create([-1.; -2.; 0.; 3.; 10.]).ReverseDiff()
        let revz = revx.LeakyRelu()
        let revzCorrect = Tensor.Create([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        // TODO: behavior of derivative at 0 (where it is undefined) can be reconsidered
        // let revxdCorrect = Tensor.Create([0.0500; 0.0500; 0.0500; 5.; -5.])
        let revxdCorrect = Tensor.Create([0.0500; 0.0500; 2.52; 5.; -5.])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeSigmoidT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Sigmoid()
        let fwdzCorrect = Tensor.Create([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.3456; 0.0684; 0.3681; 0.2893; 0.1215])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Sigmoid()
        let revzCorrect = Tensor.Create([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([1.0067;  1.1781;  1.2374;  1.1499; -1.0514])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeExpT () =
        let fwdx = Tensor.Create([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).ForwardDiff(Tensor.Create([-1.9015; 0.4606; -0.1030; 0.0466; -0.2321]))
        let fwdz = fwdx.Exp()
        let fwdzCorrect = Tensor.Create([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.5300; 0.1606; -0.2845; 0.0710; -0.8331])

        let revx = Tensor.Create([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).ReverseDiff()
        let revz = revx.Exp()
        let revzCorrect = Tensor.Create([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([6.6526; 1.7435; 13.8140; 7.6152; -17.9474])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeLogT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Log()
        let fwdzCorrect = Tensor.Create([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.8118; 0.1951; 7.3820; 2.1624; 0.6847])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Log()
        let revzCorrect = Tensor.Create([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.2780; 3.3576; 24.8177; 8.5945; -5.9248])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeLog10T () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Log10()
        let fwdzCorrect = Tensor.Create([-0.0235;  0.1729; -0.6957; -0.2352; -0.0737])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.7869; 0.0847; 3.2054; 0.9391; 0.2974])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Log10()
        let revzCorrect = Tensor.Create([-0.0235;  0.1729; -0.6957; -0.2352; -0.0737])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([2.2923;  1.4582; 10.7765;  3.7323; -2.5731])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        
    [<Test>]
    member this.TestDerivativeSqrtT () =
        let fwdx = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).ForwardDiff(Tensor.Create([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]))
        let fwdz = fwdx.Sqrt()
        let fwdzCorrect = Tensor.Create([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.5972; 0.1617; 0.1971; 0.5093; 0.0061])

        let revx = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).ReverseDiff()
        let revz = revx.Sqrt()
        let revzCorrect = Tensor.Create([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
        revz.Reverse(Tensor.Create([7.0478; 2.0493; 1.8341; 0.0166; 9.4089]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.4761; 0.1219; 0.2286; 0.0010; 0.5166])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeSinT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Sin()
        let fwdzCorrect = Tensor.Create([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.0022; 0.0237; 1.4571; 1.0510; 0.3840])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Sin()
        let revzCorrect = Tensor.Create([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([2.9194;  0.4080;  4.8988;  4.1774; -3.3228])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeCosT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Cos()
        let fwdzCorrect = Tensor.Create([0.5839; 0.0816; 0.9798; 0.8355; 0.6646])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-1.3934; -0.2895; -0.2976; -0.6913; -0.4318])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Cos()
        let revzCorrect = Tensor.Create([0.5839; 0.0816; 0.9798; 0.8355; 0.6646])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([-4.0592; -4.9833; -1.0007; -2.7476;  3.7362])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeTanT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Tan()
        let fwdzCorrect = Tensor.Create([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([5.0347; 43.6222;  1.5493;  1.8022;  1.3083])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Tan()
        let revzCorrect = Tensor.Create([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([14.6665; 750.8119;   5.2086;   7.1631; -11.3217])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
    
    [<Test>]
    member this.TestDerivativeSinhT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Sinh()
        let fwdzCorrect = Tensor.Create([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([2.5459; 0.6767; 1.5175; 1.4770; 0.7960])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Sinh()
        let revzCorrect = Tensor.Create([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([7.4163; 11.6467;  5.1018;  5.8704; -6.8886])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeCoshT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Cosh()
        let fwdzCorrect = Tensor.Create([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.8803; 0.6111; 0.3017; 0.7739; 0.5476])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Cosh()
        let revzCorrect = Tensor.Create([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.4774; 10.5188;  1.0143;  3.0759; -4.7385])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeTanhT () =
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Tanh()
        let fwdzCorrect = Tensor.Create([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.7802; 0.0535; 1.4284; 0.9126; 0.3044])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Tanh()
        let revzCorrect = Tensor.Create([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([2.2727;  0.9215;  4.8024;  3.6273; -2.6342])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
    
    [<Test>]
    member this.TestDerivativeAsinT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Asin()
        let fwdzCorrect = Tensor.Create([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([5.3579; 0.3331; 1.5183; 1.5467; 1.0770])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Asin()
        let revzCorrect = Tensor.Create([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([15.6080;  5.7324;  5.1047;  6.1476; -9.3197])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeAcosT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Acos()
        let fwdzCorrect = Tensor.Create([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-5.3579; -0.3331; -1.5183; -1.5467; -1.0770])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Acos()
        let revzCorrect = Tensor.Create([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([-15.6080;  -5.7324;  -5.1047;  -6.1476;   9.3197])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeAtanT () =
        let fwdx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.Atan()
        let fwdzCorrect = Tensor.Create([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.9046; 0.2344; 1.4292; 0.9399; 0.3375])

        let revx = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.Atan()
        let revzCorrect = Tensor.Create([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([2.6352;  4.0348;  4.8049;  3.7355; -2.9203])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeStackTs () =
        let fwdxa = Tensor.Create([1.; 2.]).ForwardDiff(Tensor.Create([10.;20.]))
        let fwdxb = Tensor.Create([3.; 4.]).ForwardDiff(Tensor.Create([30.;40.]))
        let fwdxc = Tensor.Create([5.; 6.]).ForwardDiff(Tensor.Create([50.;60.]))
        let fwdz = Tensor.Stack([fwdxa;fwdxb;fwdxc])
        let fwdzCorrect = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[10.;20.];[30.;40.];[50.;60.]])

        let revxa = Tensor.Create([1.; 2.]).ReverseDiff()
        let revxb = Tensor.Create([3.; 4.]).ReverseDiff()
        let revxc = Tensor.Create([5.; 6.]).ReverseDiff()
        let revz = Tensor.Stack([revxa;revxb;revxc])
        let revzCorrect = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]])
        revz.Reverse(Tensor.Create([[10.;20.];[30.;40.];[50.;60.]]))
        let revxda = revxa.Derivative
        let revxdb = revxb.Derivative
        let revxdc = revxc.Derivative
        let revxdaCorrect = Tensor.Create([10.; 20.])
        let revxdbCorrect = Tensor.Create([30.; 40.])
        let revxdcCorrect = Tensor.Create([50.; 60.])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxda.ApproximatelyEqual(revxdaCorrect))
        Assert.True(revxdb.ApproximatelyEqual(revxdbCorrect))
        Assert.True(revxdc.ApproximatelyEqual(revxdcCorrect))

    [<Test>]
    member this.TestDerivativeUnstackT () =
        let fwdx = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]]).ForwardDiff(Tensor.Create([[10.;20.];[30.;40.];[50.;60.]]))
        let fwdz = Tensor.Unstack(fwdx) |> Seq.toArray
        let fwdza = fwdz.[0]
        let fwdzb = fwdz.[1]
        let fwdzc = fwdz.[2]
        let fwdzda = fwdza.Derivative
        let fwdzdb = fwdzb.Derivative
        let fwdzdc = fwdzc.Derivative
        let fwdzaCorrect = Tensor.Create([1.; 2.])
        let fwdzbCorrect = Tensor.Create([3.; 4.])
        let fwdzcCorrect = Tensor.Create([5.; 6.])
        let fwdzdaCorrect = Tensor.Create([10.; 20.])
        let fwdzdbCorrect = Tensor.Create([30.; 40.])
        let fwdzdcCorrect = Tensor.Create([50.; 60.])

        let revx = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]]).ReverseDiff()
        let revz = Tensor.Unstack(revx) |> Seq.toArray
        let revza = revz.[0]
        let revzb = revz.[1]
        let revzc = revz.[2]
        let revzaCorrect = Tensor.Create([1.; 2.])
        let revzbCorrect = Tensor.Create([3.; 4.])
        let revzcCorrect = Tensor.Create([5.; 6.])
        revza.Reverse(Tensor.Create([10.; 20.]))
        revzb.Reverse(Tensor.Create([30.; 40.]), zeroDerivatives=false)
        revzc.Reverse(Tensor.Create([50.; 60.]), zeroDerivatives=false)
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[10.;20.];[30.;40.];[50.;60.]])

        Assert.True(fwdza.ApproximatelyEqual(fwdzaCorrect))
        Assert.True(fwdzb.ApproximatelyEqual(fwdzbCorrect))
        Assert.True(fwdzc.ApproximatelyEqual(fwdzcCorrect))
        Assert.True(fwdzda.ApproximatelyEqual(fwdzdaCorrect))
        Assert.True(fwdzdb.ApproximatelyEqual(fwdzdbCorrect))
        Assert.True(fwdzdc.ApproximatelyEqual(fwdzdcCorrect))
        Assert.True(revza.ApproximatelyEqual(revzaCorrect))
        Assert.True(revzb.ApproximatelyEqual(revzbCorrect))
        Assert.True(revzc.ApproximatelyEqual(revzcCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeSliceT () =
        let fwdx = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).ForwardDiff(Tensor.Create([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]))
        let fwdz = fwdx.[2..]
        let fwdzCorrect = Tensor.Create([16.0868; 74.5486; 82.9318])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.5814; 8.7951; 0.1119])

        let revx = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).ReverseDiff()
        let revz = revx.[2..]
        let revzCorrect = Tensor.Create([16.0868; 74.5486; 82.9318])
        revz.Reverse(Tensor.Create([0.9360; 0.8748; 0.4353]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.9360; 0.8748; 0.4353])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeAddTTSlice () =
        let fwdx = Tensor.Create([[-0.2754;  0.0172;  0.7105];
            [-0.1890;  1.7664;  0.5377];
            [-0.5313; -2.2530; -0.6235];
            [ 0.6776;  1.5844; -0.5686]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[-0.0552;  0.6113; -0.2341];
            [ 1.4232; -1.2062;  0.3189];
            [ 0.6859; -0.3385; -0.1263];
            [-0.5159; -1.1882; -1.3437]]))
        let fwdy = Tensor.Create([[-111.8892;   -7.0328];
            [  18.7557;  -86.2308]])            
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[ 1.3431; 23.0647];
            [71.1838; 39.8339]]))        
        let fwdz = Tensor.AddSlice(fwdx, [0;1], fwdy)
        let fwdzCorrect = Tensor.Create([[  -0.2754; -111.8720;   -6.3222];
            [  -0.1890;   20.5221;  -85.6932];
            [  -0.5313;   -2.2530;   -0.6235];
            [   0.6776;    1.5844;   -0.5686]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[-5.5237e-02;  1.9544e+00;  2.2831e+01];
            [ 1.4232e+00;  6.9978e+01;  4.0153e+01];
            [ 6.8592e-01; -3.3845e-01; -1.2635e-01];
            [-5.1592e-01; -1.1882e+00; -1.3437e+00]])

        let revx = Tensor.Create([[-0.2754;  0.0172;  0.7105];
            [-0.1890;  1.7664;  0.5377];
            [-0.5313; -2.2530; -0.6235];
            [ 0.6776;  1.5844; -0.5686]]).ReverseDiff()
        let revy = Tensor.Create([[-111.8892;   -7.0328];
            [  18.7557;  -86.2308]]).ReverseDiff()
        let revz = Tensor.AddSlice(revx, [0;1], revy)
        let revzCorrect = Tensor.Create([[  -0.2754; -111.8720;   -6.3222];
            [  -0.1890;   20.5221;  -85.6932];
            [  -0.5313;   -2.2530;   -0.6235];
            [   0.6776;    1.5844;   -0.5686]])
        revz.Reverse(Tensor.Create([[ 1.2453;  1.2199; -0.5281];
            [ 1.2203; -0.8378; -0.3876];
            [ 0.3626; -0.1200; -0.1496];
            [-0.6304;  1.0198; -0.4969]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[ 1.2453;  1.2199; -0.5281];
            [ 1.2203; -0.8378; -0.3876];
            [ 0.3626; -0.1200; -0.1496];
            [-0.6304;  1.0198; -0.4969]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[1.2199; -0.5281]; [-0.8378; -0.3876]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeSqueezeT () =
        let fwdx = Tensor.Create([[[1.; 2.]]; [[3.;4.]]]).ForwardDiff(Tensor.Create([[[10.; 20.]]; [[30.;40.]]]))
        let fwdz = fwdx.Squeeze()
        let fwdzCorrect =  Tensor.Create([[1.;2.];[3.;4.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect =  Tensor.Create([[10.;20.];[30.;40.]])

        let revx = Tensor.Create([[[1.; 2.]]; [[3.;4.]]]).ReverseDiff()
        let revz = revx.Squeeze()
        let revzCorrect =  Tensor.Create([[1.;2.];[3.;4.]])
        revz.Reverse(Tensor.Create([[10.;20.];[30.;40.]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[10.; 20.]]; [[30.;40.]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeUnsqueezeT () =
        let fwdx = Tensor.Create([[1.;2.];[3.;4.]]).ForwardDiff(Tensor.Create([[10.;20.];[30.;40.]]))
        let fwdz = fwdx.Unsqueeze(1)
        let fwdzCorrect =  Tensor.Create([[[1.; 2.]]; [[3.;4.]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect =  Tensor.Create([[[10.; 20.]]; [[30.;40.]]])

        let revx = Tensor.Create([[1.;2.];[3.;4.]]).ReverseDiff()
        let revz = revx.Unsqueeze(1)
        let revzCorrect =  Tensor.Create([[[1.; 2.]]; [[3.;4.]]])
        revz.Reverse(Tensor.Create([[[10.; 20.]]; [[30.;40.]]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[10.;20.];[30.;40.]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeFlipT () =
        let fwdx = Tensor.Create([[1.;2.];[3.;4.]]).ForwardDiff(Tensor.Create([[10.;20.];[30.;40.]]))
        let fwdz = fwdx.Flip([|0; 1|])
        let fwdzCorrect =  Tensor.Create([[4.; 3.]; [2.;1.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect =  Tensor.Create([[40.; 30.]; [20.;10.]])

        let revx = Tensor.Create([[1.;2.];[3.;4.]]).ReverseDiff()
        let revz = revx.Flip([|0; 1|])
        let revzCorrect =  Tensor.Create([[4.; 3.]; [2.;1.]])
        revz.Reverse(Tensor.Create([[40.; 30.]; [20.;10.]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[10.;20.];[30.;40.]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeSoftmax () =
        let fwdx = Tensor.Create([[4.6815; 5.6441; 7.4689];
            [9.1976; 8.1241; 7.4521]]).ForwardDiff(Tensor.Create([[8.0030; 7.0798; 6.8637];
                [9.5760; 7.4524; 2.6404]]))
        let fwdz = fwdx.Softmax(dim=1)
        let fwdzCorrect = Tensor.Create([[0.0504; 0.1319; 0.8178];
            [0.6595; 0.2254; 0.1151]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[0.0530; 0.0172; -0.0702]; [0.8422; -0.1908; -0.6514]])

        let revx = Tensor.Create([[4.6815; 5.6441; 7.4689];
            [9.1976; 8.1241; 7.4521]]).ReverseDiff()
        let revz = revx.Softmax(dim=1)
        let revzCorrect = Tensor.Create([[0.0504; 0.1319; 0.8178];
            [0.6595; 0.2254; 0.1151]])
        revz.Reverse(Tensor.Create([[6.0933; 9.6456; 7.0996];
            [0.2617; 1.7002; 4.9711]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[-0.0649; 0.2988; -0.2329]; [-0.5713; 0.1291; 0.4426]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeMaxBinary () =
        let fwdx = Tensor.Create([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).ForwardDiff(Tensor.Create([1.9788; 0.2861; 4.2025; 0.5602; 7.9510]))
        let fwdy = Tensor.Create([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).ForwardDiff(Tensor.Create([9.6600; 6.9111; 9.7303; 0.1491; 7.7003]))
        let fwdz = Tensor.Max(fwdx, fwdy)
        let fwdzCorrect = Tensor.Create([19.3520;  8.9730;  4.2405; -3.5977; 12.2048])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.9788; 0.2861; 9.7303; 0.5602; 7.7003])

        let revx = Tensor.Create([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).ReverseDiff()
        let revy = Tensor.Create([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).ReverseDiff()
        let revz = Tensor.Max(revx, revy)
        let revzCorrect = Tensor.Create([19.3520;  8.9730;  4.2405; -3.5977; 12.2048])
        revz.Reverse(Tensor.Create([  9.7293; -10.2704; -13.7527;  -3.9050;  -1.6439]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([9.7293; -10.2704; 0.; -3.9050; 0.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([0.; 0.; -13.7527; 0.; -1.6439])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeMinBinary () =
        let fwdx = Tensor.Create([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).ForwardDiff(Tensor.Create([1.9788; 0.2861; 4.2025; 0.5602; 7.9510]))
        let fwdy = Tensor.Create([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).ForwardDiff(Tensor.Create([9.6600; 6.9111; 9.7303; 0.1491; 7.7003]))
        let fwdz = Tensor.Min(fwdx, fwdy)
        let fwdzCorrect = Tensor.Create([-17.1885;  -4.0684; -23.6274; -21.7158; -20.3245])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([9.6600; 6.9111; 4.2025; 0.1491; 7.9510])

        let revx = Tensor.Create([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).ReverseDiff()
        let revy = Tensor.Create([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).ReverseDiff()
        let revz = Tensor.Min(revx, revy)
        let revzCorrect = Tensor.Create([-17.1885;  -4.0684; -23.6274; -21.7158; -20.3245])
        revz.Reverse(Tensor.Create([  9.7293; -10.2704; -13.7527;  -3.9050;  -1.6439]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; -13.7527; 0.; -1.6439])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([9.7293; -10.2704; 0.; -3.9050; 0.])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))