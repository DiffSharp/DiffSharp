namespace Tests

open NUnit.Framework
open DiffSharp

#nowarn "0058"

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)
        Assert.AreEqual(revydCorrect, revyd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)
        Assert.AreEqual(revydCorrect, revyd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)
        Assert.AreEqual(revydCorrect, revyd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)
        Assert.AreEqual(revydCorrect, revyd)

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

        let fwdz = Tensor.Conv1D(fwdx, fwdy, stride=1)
        let fwdzCorrect = Tensor.Create([[[ 143.3192;  108.0332;   11.2241];
                                         [  -5.9062;    4.6091;    6.0273]];

                                        [[  27.3032;   97.9855; -133.8372];
                                         [  -1.4792;   45.6659;   29.8705]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[ 111.2865;  -40.3692;   -1.8573];
                                         [   -1.9154;   43.3470;   29.3626]];

                                        [[ -168.6758;  -43.1578;   25.4470];
                                         [ -149.6851;   23.1963;  -50.1932]]])

        let revx = Tensor.Create([[[ 2.8564;  0.0424;  7.0984; -2.5130];
                                 [-1.1502;  0.1410;  2.5438;  4.4798];
                                 [ 0.4381; -4.3649;  2.5502;  2.5141]];

                                [[-2.8894; -7.1729; -7.1368;  1.1060];
                                 [-1.3253;  0.0257; -2.8552; -0.4933];
                                 [ 4.7305; -5.6787;  3.4658;  4.5768]]]).ReverseDiff()
        let revy = Tensor.Create([[[ 0.6355; -5.8100];
                                 [ 0.6244;  6.0336];
                                 [ 4.8205;  1.1716]];

                                [[-8.2315; -3.0400];
                                 [-2.2282; -2.9084];
                                 [-0.9613;  1.0958]]]).ReverseDiff()
        let revz = Tensor.Conv1D(revx, revy, stride=1)
        let revzCorrect = Tensor.Create([[[ -1.3005; -43.8321;  62.9678];
                                         [-26.6931; -22.6506; -69.1848]];

                                        [[ 55.3161;  -3.6187;   6.3480];
                                         [ 37.6982;  98.2438;  64.8643]]])
        revz.Reverse(Tensor.Create([[[ 4.5763;  2.7538;  2.0173];
                                     [-2.7543;  7.9257; -1.3670]];

                                    [[ 1.7997; -1.2354;  4.6313];
                                     [-4.0646;  0.0384;  4.1437]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[ 25.5806; -81.7051; -27.5597;  -7.5648];
                                         [  8.9949;  19.6812;  -2.1304;  16.1472];
                                         [ 24.7076;   7.9984;  22.9497;   0.8655]];

                                        [[ 34.6019;   0.7992; -24.1050; -39.5052];
                                         [ 10.1808;  21.8231; -13.9067;  15.8920];
                                         [ 12.5828;  -8.3376;  16.9365;   9.9666]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[ -1.8835;  15.7019];
                                         [-15.3840;  17.9761];
                                         [ 26.7091;  -1.1857]];

                                        [[-35.3382;  93.0419];
                                         [ -5.6351;  11.3910];
                                         [-44.3729;  70.9775]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeConv1Dp1 () =
        let fwdx = Tensor.Create([[[ 2.0028; -8.1570;  8.1037; -6.6905];
                             [ 3.6960; -3.8631; -7.0608; -1.4756];
                             [ 0.8208; -1.9973;  1.9964; -0.8280]];

                            [[-0.9567;  0.2317; -1.7773; -1.1823];
                             [ 5.1062;  0.2814;  6.3003;  1.3638];
                             [-4.9674;  3.9325;  3.8709; -0.6739]]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[[-5.6993;  4.2450; 16.2727; -6.0774];
                                                 [ 2.2534; -0.2354;  6.3848;  4.8030];
                                                 [-3.0135;  4.5033; -1.8186; -8.0432]];

                                                [[ 1.0174;  4.6637;  0.7299; -2.4792];
                                                 [-4.0121;  5.3963; -0.1097;  9.4151];
                                                 [11.4479;  9.9700;  4.8665;  0.8840]]]))

        let fwdy = Tensor.Create([[[-1.7830; -1.9625];
                             [-5.0868;  3.1041];
                             [ 7.7795;  1.4873]];

                            [[-1.3655;  1.6386];
                             [-6.1317;  3.5536];
                             [ 5.2382;  9.9893]]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[[ 7.7903; -0.8083];
                                                 [-4.3881; -1.4926];
                                                 [-1.7475; -7.8380]];

                                                [[-0.5209; -3.6855];
                                                 [ 6.9068;  1.8811];
                                                 [ 0.0273; -0.1305]]]))

        let fwdz = Tensor.Conv1D(fwdx, fwdy, padding=1)
        let fwdzCorrect = Tensor.Create([[[  8.7631; -14.9407; -16.1941;  44.3169;  12.9940];
                                         [ 24.6148; -68.1444;  32.4942;  18.2088;  13.8465]];

                                        [[ 10.3392; -56.6450;  57.5502;   6.7860; -10.0721];
                                         [-33.0434; -15.3614;  76.7016; -19.7505; -10.2782]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[  0.1285; -1.1446; -40.9217;  43.9575;  -120.3672];
                                         [ -31.9704; 76.8472;  -15.4720;  -175.0760;  -70.0134]];

                                        [[ 34.6626; 77.4712;  2.6527;  28.4661; -50.6120];
                                         [115.5451; 244.3823;  82.0146; 114.9505; -39.6976]]])

        let revx = Tensor.Create([[[ 2.0028; -8.1570;  8.1037; -6.6905];
                             [ 3.6960; -3.8631; -7.0608; -1.4756];
                             [ 0.8208; -1.9973;  1.9964; -0.8280]];

                            [[-0.9567;  0.2317; -1.7773; -1.1823];
                             [ 5.1062;  0.2814;  6.3003;  1.3638];
                             [-4.9674;  3.9325;  3.8709; -0.6739]]]).ReverseDiff()
        let revy = Tensor.Create([[[-1.7830; -1.9625];
                             [-5.0868;  3.1041];
                             [ 7.7795;  1.4873]];

                            [[-1.3655;  1.6386];
                             [-6.1317;  3.5536];
                             [ 5.2382;  9.9893]]]).ReverseDiff()
        let revz = Tensor.Conv1D(revx, revy, padding=1)
        let revzCorrect = Tensor.Create([[[  8.7631; -14.9407; -16.1941;  44.3169;  12.9940];
                                         [ 24.6148; -68.1444;  32.4942;  18.2088;  13.8465]];

                                        [[ 10.3392; -56.6450;  57.5502;   6.7860; -10.0721];
                                         [-33.0434; -15.3614;  76.7016; -19.7505; -10.2782]]])
        revz.Reverse(Tensor.Create([[[-3.7189; -0.4834;  1.2958; -6.2053;  3.5560];
                                     [ 5.8734;  0.3692; -6.7996;  5.7922;  3.0245]];

                                    [[-1.5334;  1.5764; -5.1078;  3.8610;  3.4756];
                                     [-7.4071;  6.3234; -3.9537;  5.0018;  3.8255]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[ 17.2804;   8.5280; -10.5300;  11.1987];
                                         [  9.5227;  34.9135; -24.0920; -35.3127];
                                         [ 51.3131; -22.5681; -83.9293;  92.1382]];

                                        [[-20.5736;  21.7742; -10.1688; -10.8016];
                                         [-77.8735;  77.5891; -80.2143; -11.3768];
                                         [-30.8856;   5.0636;   9.1459; 102.7838]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[-99.2786;  54.8582];
                                         [ 67.4523; -46.1721];
                                         [-33.6315;  -2.9197]];

                                        [[ 62.5286; -75.4390];
                                         [ 50.1777;   5.6146];
                                         [ -7.2318;  28.6983]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeConv1Ds2p2 () =
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

        let fwdz = Tensor.Conv1D(fwdx, fwdy, stride=2, padding=2)
        let fwdzCorrect = Tensor.Create([[[   0.0000;  143.3192;   11.2241;    0.0000];
                                          [   0.0000;   -5.9062;    6.0273;    0.0000]];

                                         [[   0.0000;   27.3032; -133.8372;    0.0000];
                                          [   0.0000;   -1.4792;   29.8705;    0.0000]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[   0.0000;  111.2865;   -1.8573;    0.0000];
                                          [    0.0000;   -1.9154;   29.3626;    0.0000]];

                                         [[   0.0000;  -168.6758;   25.4470;    0.0000];
                                          [   0.0000;  -149.6851;  -50.1932;    0.0000]]])

        let revx = Tensor.Create([[[  4.4675;  -3.3205;  -1.5695;   2.6373];
                                     [ -2.0373;  -1.6156;  -5.4200;   2.1263];
                                     [ -7.6023;  -3.8521;   4.1061; -11.9378]];

                                    [[ -3.6480;  -6.2680;  10.2511;   8.2932];
                                     [  6.7741;   1.4493;   0.0978;   1.8473];
                                     [  1.7488;   5.7890;  -3.9845; -10.2116]]]).ReverseDiff()
        let revy = Tensor.Create([[[ 0.5392; -7.2312];
                                 [-6.4932;  6.0252];
                                 [ 5.4071; -1.3692]];

                                [[ 2.3730; -3.1319];
                                 [-4.3207;  2.2916];
                                 [-2.1185;  5.0338]]]).ReverseDiff()
        let revz = Tensor.Conv1D(revx, revy, stride=2, padding=2)
        let revzCorrect = Tensor.Create([[[  0.0000;  -5.9184;  66.6342;   0.0000];
                                         [  0.0000;  22.8156; -52.4840;   0.0000]];

                                        [[  0.0000;   9.6349; -51.5099;   0.0000];
                                         [  0.0000;  10.4620; -40.7989;   0.0000]]])
        revz.Reverse(Tensor.Create([[[ -3.2046;  -1.7019;   5.4695;  -1.0924];
                                     [  2.3244;   2.3524;   1.4251;   5.4270]];

                                    [[ -1.2459;   3.5582;   0.4258;  -9.7433];
                                     [-10.7600;  -1.3447;   2.6181;  -1.3003]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[  4.6644;   4.9396;   6.3311; -44.0143];
                                         [  0.8869;  -4.8637; -41.6721;  36.2206];
                                         [-14.1861;  14.1717;  26.5551;  -0.3149]];

                                        [[ -1.2721; -21.5189;   6.4422; -11.2788];
                                         [-17.2943;  18.3576; -14.0771;   8.5654];
                                         [ 22.0885; -11.6407;  -3.2439;  12.5958]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[-24.8029;   1.3047];
                                         [ -2.0322;  20.3232];
                                         [ 39.9225; -42.4877]];

                                        [[ 40.0164;  26.0883];
                                         [-21.3696;   2.1173];
                                         [-24.8156; -60.5938]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeConv1Ds2p2d3 () =
        let fwdx = Tensor.Create([[[  4.4675;  -3.3205;  -1.5695;   2.6373];
                                 [ -2.0373;  -1.6156;  -5.4200;   2.1263];
                                 [ -7.6023;  -3.8521;   4.1061; -11.9378]];

                                [[ -3.6480;  -6.2680;  10.2511;   8.2932];
                                 [  6.7741;   1.4493;   0.0978;   1.8473];
                                 [  1.7488;   5.7890;  -3.9845; -10.2116]]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[[ -2.4789;  -2.3435;  -1.7153;  -9.8687];
                                                     [  9.5786; -10.2393;   8.3291;  -8.8992];
                                                     [-10.1198;  -1.4206;   5.4935;   0.2305]];

                                                    [[ -5.6670;  -5.2314;  -0.1757;  -0.0272];
                                                     [ -2.3740;  -0.1860;   1.9684;  -1.5754];
                                                     [  1.5551;  -1.1761;   7.5176;  -1.9207]]]))

        let fwdy = Tensor.Create([[[ 0.5392; -7.2312];
                                 [-6.4932;  6.0252];
                                 [ 5.4071; -1.3692]];

                                [[ 2.3730; -3.1319];
                                 [-4.3207;  2.2916];
                                 [-2.1185;  5.0338]]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[[ -7.0064;   0.3474];
                                                 [  1.8052;   3.9392];
                                                 [ -6.8035;  -4.4947]];

                                                [[-10.6156;  -2.6311];
                                                 [  8.8583;   3.0635];
                                                 [  5.1788;  -4.6292]]]))

        let fwdz = Tensor.Conv1D(fwdx, fwdy, stride=2, padding=2, dilation=3)
        let fwdzCorrect = Tensor.Create([[[  19.5508;  -15.3841;   56.5490];
                                         [ -12.6935;  -27.9701;   10.9953]];

                                        [[  46.1316;  -71.3546;  -16.6521];
                                         [  52.0922; -114.7731;   32.3440]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[  -33.0063;  -21.1314;   -52.0266];
                                         [ -1.6565;  -64.1607;  -61.7825]];

                                        [[  15.8308;  96.0475;  -16.7656];
                                         [  4.1709; 119.2216;  -153.4385]]])

        let revx = Tensor.Create([[[  4.4675;  -3.3205;  -1.5695;   2.6373];
                                 [ -2.0373;  -1.6156;  -5.4200;   2.1263];
                                 [ -7.6023;  -3.8521;   4.1061; -11.9378]];

                                [[ -3.6480;  -6.2680;  10.2511;   8.2932];
                                 [  6.7741;   1.4493;   0.0978;   1.8473];
                                 [  1.7488;   5.7890;  -3.9845; -10.2116]]]).ReverseDiff()
        let revy = Tensor.Create([[[ 0.5392; -7.2312];
                                 [-6.4932;  6.0252];
                                 [ 5.4071; -1.3692]];

                                [[ 2.3730; -3.1319];
                                 [-4.3207;  2.2916];
                                 [-2.1185;  5.0338]]]).ReverseDiff()
        let revz = Tensor.Conv1D(revx, revy, stride=2, padding=2, dilation=3)
        let revzCorrect = Tensor.Create([[[  19.5508;  -15.3841;   56.5490];
                                         [ -12.6935;  -27.9701;   10.9953]];

                                        [[  46.1316;  -71.3546;  -16.6521];
                                         [  52.0922; -114.7731;   32.3440]]])
        revz.Reverse(Tensor.Create([[[  2.9290;  -8.6265;   2.4380];
                                     [-11.2377;  -8.5424;   3.5779]];

                                    [[ -2.1010;  -5.1271;   2.4158];
                                     [  7.3964;  -1.2906;   4.3965]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[-24.9225;  14.0146;   9.8048;  89.1337];
                                         [ 92.9230;  -8.1042; -31.2896; -71.5521];
                                         [-28.5474; -60.5786;   5.6028; -31.1895]];

                                        [[ -5.8272;  -7.9718;  11.7355;  41.1170];
                                         [ 38.8675;   4.2906; -34.6827; -33.8492];
                                         [-24.9887;  40.1087;   3.7486;   0.5234]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[  1.1031; -61.8278];
                                         [-30.1347; -35.5914];
                                         [ 57.0000; 131.8920]];

                                        [[  5.9986; -42.2784];
                                         [-10.3015;   8.3273];
                                         [ 59.8580; 201.2624]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    //TODO: add test for Conv1DTTConst
    //TODO: add test for Conv1DTConstT

    [<Test>]
    member this.TestDerivativeConv2D () =
        let fwdx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
          [  4.1074,  12.5312,  -3.2430,   4.4820],
          [-10.0428,   5.0817,   0.4602,  -0.9825],
          [  4.5867,   1.2517,   4.2247,   0.0669]],

         [[  3.6238,  -5.6671,  -4.1270,   1.9999],
          [  4.3497,  -3.8131,  -3.6954,   2.5138],
          [  4.2289,   4.4896,  -0.8233,  10.3737],
          [ -9.1522,  -8.0464,  -2.1834,   1.3475]],

         [[ -5.4871,  -5.6456,   2.3971,  -8.8393],
          [  6.0641,  -2.0258,  -7.5135,   0.3814],
          [ -4.3724,  -1.9445,   6.8157,   6.4477],
          [  2.1722,   4.3881,  -2.5686,  -2.4257]]],


        [[[ -1.7312,  -2.5755,   1.5300,  10.9412],
          [  9.6599,  -6.6948,   4.7075,   4.2106],
          [ -0.8718,  -5.4295,  10.0747, -10.1636],
          [  4.5319,   4.0764,   1.6741,   5.8974]],

         [[  0.9215,  -3.5721,  -1.2668,   6.4006],
          [  0.6660,   4.1247,   3.5245,  -4.6866],
          [ -1.0934,  10.4278,  -0.3531,  -5.8575],
          [ -1.1816,  -6.7908,  -6.1499,  -2.0547]],

         [[ -3.9967,  -4.4300,  -0.2993,   5.8606],
          [ -0.9812,  -1.8121,   3.4439,   7.4879],
          [ -2.4948,  -4.9388,  -1.7896,  -3.9585],
          [  6.3013,   2.1417,   4.0991,  -1.6199]]]]))

        let fwdy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[[[-1.1954e+01,  2.6855e+00],
          [-1.4551e+00, -1.6017e+00]],

         [[ 1.7954e+00,  1.5183e+01],
          [-5.1061e-01, -4.2037e+00]],

         [[-5.7741e+00, -9.1211e-01],
          [-4.1928e+00, -1.1891e+01]]],


        [[[-4.8966e+00, -7.3858e-03],
          [ 6.5086e+00,  6.6678e-01]],

         [[ 2.2310e-01,  7.3424e+00],
          [ 5.5643e-01,  1.2690e+01]],

         [[-5.4125e+00, -3.2977e+00],
          [ 3.1655e+00, -9.4486e+00]]]]))

        let fwdz = Tensor.Conv2D(fwdx, fwdy)
        let fwdzCorrect = Tensor.Create([[[[  69.4275,   57.5256,  204.7637],
          [  72.6434,  -98.7244,   48.0571],
          [  50.4435,  -79.0234,  -50.5811]],

         [[  -6.8532,  110.2038,  -26.9510],
          [ -93.3006,  -57.0783,    0.9066],
          [  36.5820,   87.2137,  -18.4858]]],


        [[[   6.2275, -161.3704,  -84.4986],
          [ -55.9199,   39.6219,    1.1004],
          [   0.9904,   57.8859,  121.5461]],

         [[  46.9451,  -11.9214,  -25.7160],
          [ -36.8064,   22.9777,  -81.6225],
          [  17.8893,   39.8201,  -28.4861]]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[[-222.1388,  -38.6186,  176.5756],
          [ 242.9850, -198.9088,  177.0603],
          [ 137.6601,  -48.8533,   81.2111]],

         [[  26.2820,  175.4338,  -43.1074],
          [-199.0381,   75.3434,  132.5411],
          [ 120.8810,  -18.4522,    8.6456]]],


        [[[  87.8043,   76.5475,  122.6293],
          [-118.5491,  144.9104,   67.6158],
          [   5.6988,  166.2014, -175.9812]],

         [[ 152.4038,  -51.7842,  104.9729],
          [ -63.2891,  -37.7844,   -7.9095],
          [ 159.6389,   17.9089,  178.1622]]]])

        let revx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).ReverseDiff()
        let revy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]]).ReverseDiff()
        let revz = Tensor.Conv2D(revx, revy)
        let revzCorrect = Tensor.Create([[[[  69.4275,   57.5256,  204.7637],
          [  72.6434,  -98.7244,   48.0571],
          [  50.4435,  -79.0234,  -50.5811]],

         [[  -6.8532,  110.2038,  -26.9510],
          [ -93.3006,  -57.0783,    0.9066],
          [  36.5820,   87.2137,  -18.4858]]],


        [[[   6.2275, -161.3704,  -84.4986],
          [ -55.9199,   39.6219,    1.1004],
          [   0.9904,   57.8859,  121.5461]],

         [[  46.9451,  -11.9214,  -25.7160],
          [ -36.8064,   22.9777,  -81.6225],
          [  17.8893,   39.8201,  -28.4861]]]])
        revz.Reverse(Tensor.Create([[[[  0.9103,   0.4812,   0.4156],
          [  5.7374,   4.1146,  -0.3798],
          [ -6.9746,   0.1408,  -0.8381]],

         [[  2.9837,   1.7493,   1.9437],
          [ -7.6868,  -1.0847,  -5.8083],
          [ -5.6574,   3.0264,   2.2271]]],


        [[[  2.9846,  -1.0026,   1.4756],
          [  3.2417,  -0.1431, -12.3301],
          [  9.2809,   4.7381,   1.1553]],

         [[ -1.4849,   3.4750,   1.1084],
          [ -5.1601,   0.4057,  -4.7773],
          [ -4.0470,  -3.2604,   4.7280]]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[[ 1.5744e+00,  6.2329e+00,  3.6429e+00,  6.3518e-01],
          [-1.1657e+00,  9.2745e+01,  6.2671e+01, -1.2355e+00],
          [-6.6796e+01, -1.0047e+02, -6.4138e+01,  1.3308e+01],
          [-2.3897e+01,  9.3344e+01,  2.6230e+00, -6.9200e+00]],

         [[-1.5916e+01,  1.0755e+01,  1.0502e+00,  1.1101e+01],
          [ 3.7602e+01,  2.8869e+01,  6.8584e+01, -3.1641e+01],
          [ 1.6763e+01, -7.7995e+01, -7.4988e+00,  2.2354e+01],
          [-1.6248e+01,  2.8387e+01, -6.3591e+00, -9.6984e+00]],

         [[-3.1077e+01, -3.3924e+01, -2.8451e+01, -9.3088e+00],
          [ 6.1527e+01,  1.0975e+01,  5.5105e+01,  3.3791e+01],
          [ 4.2590e+01, -8.4962e+00, -6.7049e+01, -3.5741e+01],
          [-4.1352e+01, -1.7293e+01,  2.9837e+01,  1.2157e+01]]],


        [[[-8.2185e+00,  5.7007e+01, -2.7805e+01,  1.9566e+01],
          [-3.2923e+01,  8.6503e+01,  9.3691e+00, -1.9134e+02],
          [-7.3624e+01,  1.5387e+02,  8.2900e+01,  1.0073e+02],
          [-5.7140e+01, -7.0859e+01,  2.2811e+01, -3.1745e+01]],

         [[ 2.9070e+00,  1.2004e+01, -6.7632e+00,  1.9838e+01],
          [ 1.6555e+01,  3.3493e+01,  2.9367e+01, -1.5446e+02],
          [-7.9334e+00,  1.2084e+02, -5.3646e-02,  4.2080e+01],
          [-1.0679e+01,  1.2848e+01,  2.9283e+01, -1.9073e+01]],

         [[-4.9491e-01, -3.9616e+01, -2.2836e+01, -1.2852e+01],
          [ 2.5431e+01,  8.4768e+00,  1.2704e+02,  9.8819e+01],
          [-3.4615e+01, -6.0232e+01, -1.0465e+02, -5.7172e+01],
          [-1.5058e+01, -3.2852e+01,  1.3887e+01,  2.7381e+01]]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[[ -45.0676,  -19.8231],
          [ -23.2455, -202.1413]],

         [[ -37.1328,  -86.2777],
          [   4.8285,   34.8047]],

         [[ 119.0452,  -67.8937],
          [ 120.6469,  -34.3602]]],


        [[[  65.0189,   25.5965],
          [ 111.0391,   67.9908]],

         [[  47.3326,  -50.8033],
          [ -16.2907,  -63.8956]],

         [[ -80.6018,  -10.6333],
          [  66.0623,  -79.7451]]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeConv2Dp1 () =
        let fwdx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
          [  4.1074,  12.5312,  -3.2430,   4.4820],
          [-10.0428,   5.0817,   0.4602,  -0.9825],
          [  4.5867,   1.2517,   4.2247,   0.0669]],

         [[  3.6238,  -5.6671,  -4.1270,   1.9999],
          [  4.3497,  -3.8131,  -3.6954,   2.5138],
          [  4.2289,   4.4896,  -0.8233,  10.3737],
          [ -9.1522,  -8.0464,  -2.1834,   1.3475]],

         [[ -5.4871,  -5.6456,   2.3971,  -8.8393],
          [  6.0641,  -2.0258,  -7.5135,   0.3814],
          [ -4.3724,  -1.9445,   6.8157,   6.4477],
          [  2.1722,   4.3881,  -2.5686,  -2.4257]]],


        [[[ -1.7312,  -2.5755,   1.5300,  10.9412],
          [  9.6599,  -6.6948,   4.7075,   4.2106],
          [ -0.8718,  -5.4295,  10.0747, -10.1636],
          [  4.5319,   4.0764,   1.6741,   5.8974]],

         [[  0.9215,  -3.5721,  -1.2668,   6.4006],
          [  0.6660,   4.1247,   3.5245,  -4.6866],
          [ -1.0934,  10.4278,  -0.3531,  -5.8575],
          [ -1.1816,  -6.7908,  -6.1499,  -2.0547]],

         [[ -3.9967,  -4.4300,  -0.2993,   5.8606],
          [ -0.9812,  -1.8121,   3.4439,   7.4879],
          [ -2.4948,  -4.9388,  -1.7896,  -3.9585],
          [  6.3013,   2.1417,   4.0991,  -1.6199]]]]))

        let fwdy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[[[-1.1954e+01,  2.6855e+00],
          [-1.4551e+00, -1.6017e+00]],

         [[ 1.7954e+00,  1.5183e+01],
          [-5.1061e-01, -4.2037e+00]],

         [[-5.7741e+00, -9.1211e-01],
          [-4.1928e+00, -1.1891e+01]]],


        [[[-4.8966e+00, -7.3858e-03],
          [ 6.5086e+00,  6.6678e-01]],

         [[ 2.2310e-01,  7.3424e+00],
          [ 5.5643e-01,  1.2690e+01]],

         [[-5.4125e+00, -3.2977e+00],
          [ 3.1655e+00, -9.4486e+00]]]]))

        let fwdz = Tensor.Conv2D(fwdx, fwdy, padding=1)
        let fwdzCorrect = Tensor.Create([[[[   2.9885,   -4.3019,   -1.1975,  -49.4543,  -24.2013],
          [   6.2745,   69.4275,   57.5256,  204.7637,  -14.6282],
          [ -70.7221,   72.6434,  -98.7244,   48.0571,   -0.4058],
          [  26.7088,   50.4435,  -79.0234,  -50.5811,  -37.6351],
          [ -96.5496,   47.6966,    2.5257,  -20.6500,    8.4670]],

         [[   7.1575,  -29.7318,    6.5727,  -83.8713,   63.0648],
          [  30.3794,   -6.8532,  110.2038,  -26.9510,   17.6584],
          [  -9.0821,  -93.3006,  -57.0783,    0.9066,   28.6689],
          [  59.5862,   36.5820,   87.2137,  -18.4858,  -77.4703],
          [  12.7419,   18.3116, -203.8608,   -2.0445,   24.1051]]],


        [[[  38.7963,   37.5600,   67.4382,   53.8644,   10.4056],
          [-212.9548,    6.2275, -161.3704,  -84.4986,   21.1666],
          [ -61.9209,  -55.9199,   39.6219,    1.1004,  -56.4032],
          [ -44.7484,    0.9904,   57.8859,  121.5461,   -4.2968],
          [ -61.9822, -113.1281,  -45.0625,   42.6192,   18.3062]],

         [[  92.9820,  -57.9407,   36.4209,  -33.5703,  -46.3160],
          [  31.7784,   46.9451,  -11.9214,  -25.7160,   79.4210],
          [   3.6550,  -36.8064,   22.9777,  -81.6225,  -44.4841],
          [  28.2701,   17.8893,   39.8201,  -28.4861,    0.7710],
          [ -20.3688,    0.8945,  -24.6142,  -14.1374,   15.2676]]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[[-1.0527e+00, -9.6677e+00,  2.0051e+01, -2.8588e+01, -2.8092e+01],
          [ 6.6973e+01, -2.2214e+02, -3.8619e+01,  1.7658e+02, -7.2230e+01],
          [ 1.5061e+02,  2.4299e+02, -1.9891e+02,  1.7706e+02, -2.7937e+01],
          [-2.5458e+01,  1.3766e+02, -4.8853e+01,  8.1211e+01, -7.1649e+01],
          [-1.0480e+02,  8.0479e+01, -5.5455e+01, -1.1297e+02,  6.3481e+00]],

         [[-6.6083e+01,  5.6381e+01, -1.9298e+01, -1.2133e+01,  3.3643e+01],
          [-7.0305e+00,  2.6282e+01,  1.7543e+02, -4.3107e+01,  8.7134e+01],
          [ 5.5268e+01, -1.9904e+02,  7.5343e+01,  1.3254e+02,  7.1742e+01],
          [ 9.8615e+01,  1.2088e+02, -1.8452e+01,  8.6456e+00, -1.5476e+02],
          [-6.8246e+01,  4.1057e+01, -1.1768e+02, -1.5195e+01,  2.0304e+01]]],


        [[[ 1.2016e+01,  4.3505e+01, -8.1550e+00, -2.6647e+01, -1.2920e+01],
          [-2.7730e+02,  8.7804e+01,  7.6548e+01,  1.2263e+02, -4.6506e+01],
          [ 1.0181e+02, -1.1855e+02,  1.4491e+02,  6.7616e+01, -5.6487e+01],
          [-6.2205e+01,  5.6988e+00,  1.6620e+02, -1.7598e+02, -1.9307e+01],
          [-3.3957e+01, -9.2942e+01,  9.1539e+00,  1.5654e+02,  5.5646e+01]],

         [[-1.6286e+02, -4.2981e+01, -1.1976e+02, -1.4197e+02,  1.0601e+02],
          [-1.9272e+02,  1.5240e+02, -5.1784e+01,  1.0497e+02,  4.7483e+01],
          [-9.0446e+01, -6.3289e+01, -3.7784e+01, -7.9095e+00, -1.6118e+02],
          [-5.6473e+01,  1.5964e+02,  1.7909e+01,  1.7816e+02,  3.4444e+01],
          [-7.3489e+01, -1.0654e+02, -5.6583e-02,  2.0659e+01,  6.7274e+01]]]])

        let revx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).ReverseDiff()
        let revy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]]).ReverseDiff()
        let revz = Tensor.Conv2D(revx, revy, padding=1)
        let revzCorrect = Tensor.Create([[[[   2.9885,   -4.3019,   -1.1975,  -49.4543,  -24.2013],
          [   6.2745,   69.4275,   57.5256,  204.7637,  -14.6282],
          [ -70.7221,   72.6434,  -98.7244,   48.0571,   -0.4058],
          [  26.7088,   50.4435,  -79.0234,  -50.5811,  -37.6351],
          [ -96.5496,   47.6966,    2.5257,  -20.6500,    8.4670]],

         [[   7.1575,  -29.7318,    6.5727,  -83.8713,   63.0648],
          [  30.3794,   -6.8532,  110.2038,  -26.9510,   17.6584],
          [  -9.0821,  -93.3006,  -57.0783,    0.9066,   28.6689],
          [  59.5862,   36.5820,   87.2137,  -18.4858,  -77.4703],
          [  12.7419,   18.3116, -203.8608,   -2.0445,   24.1051]]],


        [[[  38.7963,   37.5600,   67.4382,   53.8644,   10.4056],
          [-212.9548,    6.2275, -161.3704,  -84.4986,   21.1666],
          [ -61.9209,  -55.9199,   39.6219,    1.1004,  -56.4032],
          [ -44.7484,    0.9904,   57.8859,  121.5461,   -4.2968],
          [ -61.9822, -113.1281,  -45.0625,   42.6192,   18.3062]],

         [[  92.9820,  -57.9407,   36.4209,  -33.5703,  -46.3160],
          [  31.7784,   46.9451,  -11.9214,  -25.7160,   79.4210],
          [   3.6550,  -36.8064,   22.9777,  -81.6225,  -44.4841],
          [  28.2701,   17.8893,   39.8201,  -28.4861,    0.7710],
          [ -20.3688,    0.8945,  -24.6142,  -14.1374,   15.2676]]]])
        revz.Reverse(Tensor.Create([[[[ -0.0818,  -1.7126,  -6.8625,  -1.1005,   6.4032],
          [  1.2647,   2.8299,  -7.0770,  -7.3823, -10.7036],
          [ -0.5949,  -1.6608,   0.1277,  -1.3747,   1.0839],
          [  1.3014,  -6.3608,  -5.8820,  -1.9287,  -3.7376],
          [ -3.1642,  -8.7039,  -2.0229,  -1.3187,   4.2127]],

         [[ -4.7775,   0.0534,  -1.6446,  -1.4996,  -2.7337],
          [ -0.8560,   4.6122,   5.2424,  -2.4503,  -1.2369],
          [ -5.6313,  -1.1085,   0.5474,   4.8368,   2.9536],
          [  4.0118,  -5.4583,  -2.1389,  -6.2162,   1.5350],
          [ -5.2292,  10.6073,  -5.2807,  -0.0235,   2.9736]]],


        [[[  2.4959,   6.3992,   0.2723,   4.3979,   7.2741],
          [ -3.5761,  -1.6745,   2.2457,  -8.5449,  10.0545],
          [  7.9747,  -9.7400,  -8.2683,   0.6409,   2.5845],
          [ -7.5924,   2.2433,  -0.4786,  -5.8168,  -7.2329],
          [  2.7957,   2.0224,   1.2842,  -0.8304,  -1.2040]],

         [[  8.6663,  -6.4984,   0.0181,  -5.5948,  -1.4446],
          [  9.0790,   1.3896,  -1.6444,   4.1883,   6.5606],
          [ -5.8707,  -4.0481,  -1.1043,  -4.2776,   0.1577],
          [ -5.6635,  -2.3297,  -8.7693,   9.0650,   1.4648],
          [ -6.3961,   3.6526,   5.7533,  -5.4021,   0.1755]]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[[  52.5073,   67.9158,  -72.1813, -110.0932],
          [  34.5613,   -2.7875,   24.3253,   41.8013],
          [  44.8780,  -52.7584,  -50.6334,    0.6157],
          [ -54.9037, -102.6098,   -9.6011,   40.4460]],

         [[   4.0912,   24.9880,  -37.6817,  -68.6669],
          [  -0.4027,  -28.2864,  -51.1940,  -12.4506],
          [  83.6321,  -63.4892,  -26.9752,  -56.6132],
          [-123.0175,  -17.2464,  -49.4828,   -6.2429]],

         [[ -87.8671,  -65.1930,   57.9617,   95.2186],
          [  67.5832,   60.6227,  -36.2991,  -76.3125],
          [  16.1409,  100.6101,  139.2297,   82.5894],
          [ -33.6545,   19.6965,  -17.7912,  -68.9333]]],


        [[[-205.5825,  -40.3349,    6.1795, -186.4079],
          [ 145.5407, -138.9059,  -79.5474,   66.9971],
          [-127.4452,  125.1183,   61.3658,  -87.0136],
          [ 112.3364,  -42.2452,  133.2808,   23.4470]],

         [[ -65.1486,   21.6096,   -4.3367, -103.7793],
          [  65.2791, -113.5311,  -54.8015,  -14.8741],
          [ -77.2162,   68.3465,  -82.2631,  -10.6803],
          [   1.3535,   -9.2549,  122.2180,  -63.5231]],

         [[   2.9495,  -23.2417,  -35.1122,  -98.9359],
          [ 109.7330,  121.0604,   98.2866,   66.6759],
          [  13.1614,   32.3627,  -57.0056,    3.9335],
          [ -84.9322, -148.3569,   23.6587,   77.6059]]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[[  60.9107,   -2.4922],
          [  44.1872,  -36.8305]],

         [[-102.1387,  -41.7113],
          [-192.5679, -188.1346]],

         [[  26.3335,   -1.7302],
          [ -96.6882,  -71.5404]]],


        [[[-119.3937,   15.9902],
          [ 113.3007,  -55.3270]],

         [[-105.7772,   -6.5650],
          [ 145.0033,  -17.3124]],

         [[-120.5291,   79.5252],
          [ 102.1578,   65.0787]]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeConv2Ds2p2 () =
        let fwdx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
          [  4.1074,  12.5312,  -3.2430,   4.4820],
          [-10.0428,   5.0817,   0.4602,  -0.9825],
          [  4.5867,   1.2517,   4.2247,   0.0669]],

         [[  3.6238,  -5.6671,  -4.1270,   1.9999],
          [  4.3497,  -3.8131,  -3.6954,   2.5138],
          [  4.2289,   4.4896,  -0.8233,  10.3737],
          [ -9.1522,  -8.0464,  -2.1834,   1.3475]],

         [[ -5.4871,  -5.6456,   2.3971,  -8.8393],
          [  6.0641,  -2.0258,  -7.5135,   0.3814],
          [ -4.3724,  -1.9445,   6.8157,   6.4477],
          [  2.1722,   4.3881,  -2.5686,  -2.4257]]],


        [[[ -1.7312,  -2.5755,   1.5300,  10.9412],
          [  9.6599,  -6.6948,   4.7075,   4.2106],
          [ -0.8718,  -5.4295,  10.0747, -10.1636],
          [  4.5319,   4.0764,   1.6741,   5.8974]],

         [[  0.9215,  -3.5721,  -1.2668,   6.4006],
          [  0.6660,   4.1247,   3.5245,  -4.6866],
          [ -1.0934,  10.4278,  -0.3531,  -5.8575],
          [ -1.1816,  -6.7908,  -6.1499,  -2.0547]],

         [[ -3.9967,  -4.4300,  -0.2993,   5.8606],
          [ -0.9812,  -1.8121,   3.4439,   7.4879],
          [ -2.4948,  -4.9388,  -1.7896,  -3.9585],
          [  6.3013,   2.1417,   4.0991,  -1.6199]]]]))

        let fwdy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[[[-1.1954e+01,  2.6855e+00],
          [-1.4551e+00, -1.6017e+00]],

         [[ 1.7954e+00,  1.5183e+01],
          [-5.1061e-01, -4.2037e+00]],

         [[-5.7741e+00, -9.1211e-01],
          [-4.1928e+00, -1.1891e+01]]],


        [[[-4.8966e+00, -7.3858e-03],
          [ 6.5086e+00,  6.6678e-01]],

         [[ 2.2310e-01,  7.3424e+00],
          [ 5.5643e-01,  1.2690e+01]],

         [[-5.4125e+00, -3.2977e+00],
          [ 3.1655e+00, -9.4486e+00]]]]))

        let fwdz = Tensor.Conv2D(fwdx, fwdy, stride=2, padding=2)
        let fwdzCorrect = Tensor.Create([[[[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,  69.4275, 204.7637,   0.0000],
          [  0.0000,  50.4435, -50.5811,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]],

         [[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,  -6.8532, -26.9510,   0.0000],
          [  0.0000,  36.5820, -18.4858,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]]],


        [[[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,   6.2275, -84.4986,   0.0000],
          [  0.0000,   0.9904, 121.5461,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]],

         [[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,  46.9451, -25.7160,   0.0000],
          [  0.0000,  17.8893, -28.4861,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[[   0.0000,    0.0000,    0.0000,    0.0000],
          [   0.0000, -222.1388,  176.5756,    0.0000],
          [   0.0000,  137.6601,   81.2111,    0.0000],
          [   0.0000,    0.0000,    0.0000,    0.0000]],

         [[   0.0000,    0.0000,    0.0000,    0.0000],
          [   0.0000,   26.2820,  -43.1074,    0.0000],
          [   0.0000,  120.8810,    8.6456,    0.0000],
          [   0.0000,    0.0000,    0.0000,    0.0000]]],


        [[[   0.0000,    0.0000,    0.0000,    0.0000],
          [   0.0000,   87.8043,  122.6293,    0.0000],
          [   0.0000,    5.6988, -175.9812,    0.0000],
          [   0.0000,    0.0000,    0.0000,    0.0000]],

         [[   0.0000,    0.0000,    0.0000,    0.0000],
          [   0.0000,  152.4038,  104.9729,    0.0000],
          [   0.0000,  159.6389,  178.1622,    0.0000],
          [   0.0000,    0.0000,    0.0000,    0.0000]]]])

        let revx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).ReverseDiff()
        let revy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]]).ReverseDiff()
        let revz = Tensor.Conv2D(revx, revy, stride=2, padding=2)
        let revzCorrect = Tensor.Create([[[[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,  69.4275, 204.7637,   0.0000],
          [  0.0000,  50.4435, -50.5811,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]],

         [[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,  -6.8532, -26.9510,   0.0000],
          [  0.0000,  36.5820, -18.4858,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]]],


        [[[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,   6.2275, -84.4986,   0.0000],
          [  0.0000,   0.9904, 121.5461,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]],

         [[  0.0000,   0.0000,   0.0000,   0.0000],
          [  0.0000,  46.9451, -25.7160,   0.0000],
          [  0.0000,  17.8893, -28.4861,   0.0000],
          [  0.0000,   0.0000,   0.0000,   0.0000]]]])
        revz.Reverse(Tensor.Create([[[[ 6.7571e+00,  9.0846e+00, -2.3487e+00,  4.3374e+00],
          [-1.3544e+00,  5.6528e+00,  6.2220e-01,  2.9935e+00],
          [-1.2652e+01,  9.3760e+00, -1.4333e+00,  8.0374e-01],
          [-3.7678e+00,  1.0421e+01,  4.4332e-01,  1.0364e+00]],

         [[-9.2362e-01,  1.8509e+00, -3.8252e+00,  1.0499e+01],
          [-1.7690e+00,  2.6878e+00,  2.1804e-03,  2.0317e-01],
          [ 1.1629e+00, -1.7032e+00,  1.1897e+00,  7.2766e+00],
          [-2.4891e+00,  4.2235e+00,  1.7844e+00,  4.3290e+00]]],


        [[[ 4.9063e-01,  7.7006e-01,  2.9074e-03, -6.1390e+00],
          [ 1.7664e-01,  5.5859e-01,  4.5253e+00,  1.1648e+01],
          [-1.0276e+00, -1.2210e+00,  4.3299e+00,  1.1510e+00],
          [ 6.5332e+00, -1.5466e+00,  6.2008e+00, -6.0223e+00]],

         [[-3.6580e+00, -3.2587e+00,  3.0122e+00,  4.6115e-01],
          [-6.2328e+00, -2.1595e+00, -3.5035e+00, -3.4184e+00],
          [ 1.4125e+00,  6.7146e-01, -9.1552e+00, -6.8517e+00],
          [-6.3924e+00, -8.7000e+00, -6.5166e+00,  8.5122e-01]]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[[-9.0342e+00,  7.9611e+01, -1.3431e+00,  9.6403e+00],
          [ 4.7888e+00, -4.7081e+01, -1.7293e+00, -3.6275e+00],
          [-2.2301e+01,  1.5046e+02,  4.5128e+00, -2.5778e+01],
          [-3.9398e+01, -4.5472e+01,  1.3164e+01,  2.0310e+00]],

         [[-2.1386e+01,  7.0731e+01, -9.1812e-01,  6.7930e+00],
          [ 7.8739e+00, -8.2607e+00,  4.7229e-02,  3.1247e-01],
          [-5.3469e+00,  9.6498e+01, -3.7266e+00, -1.1611e+01],
          [-4.1328e+00,  1.1931e+01,  3.2252e+00, -5.6904e+00]],

         [[-5.0432e+01, -4.3800e+01, -2.9084e+00, -3.7971e+00],
          [ 2.2026e+01,  1.8243e+01,  6.4640e-01,  3.4581e-01],
          [-2.8204e+01, -5.1166e+01, -4.0518e+00,  4.5811e+00],
          [-7.7079e-01, -4.6156e+00,  5.7449e+00,  5.9661e+00]]],


        [[[-3.7725e+00,  1.5114e+01, -1.3948e+01,  8.0633e+01],
          [-1.8160e+01,  8.1865e+00, -3.9618e+01, -7.7511e+00],
          [ 3.4383e+00, -2.0938e+01, -2.0237e+01,  9.4495e+01],
          [ 8.5857e+00,  3.5413e+00, -8.2496e+01,  2.3306e+01]],

         [[ 9.7438e+00, -1.2053e+00,  1.0530e+01,  3.7514e+01],
          [-5.9890e+00,  9.2728e+00, -9.4771e+00,  1.6914e+01],
          [-1.5022e+00, -1.1048e+01,  3.8448e+01,  1.6285e+01],
          [ 1.7929e+00, -3.4245e+00, -2.5261e+01,  4.0326e+01]],

         [[ 1.6840e+01,  4.1279e+00,  1.0517e+01, -1.5345e+01],
          [-1.2507e+01, -1.1924e+01, -1.6607e+01, -1.7405e+01],
          [-3.7334e-01,  5.0954e+00,  6.2284e+01,  5.5519e+00],
          [ 2.8228e+00,  3.1462e+00, -5.1026e+01, -4.9500e+01]]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[[ -92.7970,  -19.9241],
          [-103.8568,   27.1854]],

         [[  18.3723,   51.3441],
          [ -23.5677,  136.5338]],

         [[ -63.8686,  -33.2020],
          [  -6.3042,  171.1903]]],


        [[[  78.0537,  -24.2271],
          [  35.3763,   21.2273]],

         [[  -9.2828,   37.4667],
          [ -17.5716,  -65.7624]],

         [[  39.3125,   32.0967],
          [ -47.9975,   34.2678]]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    [<Test>]
    member this.TestDerivativeConv2Ds2p2d3 () =
        let fwdx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]])
        let fwdx = fwdx.ForwardDiff(Tensor.Create([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
          [  4.1074,  12.5312,  -3.2430,   4.4820],
          [-10.0428,   5.0817,   0.4602,  -0.9825],
          [  4.5867,   1.2517,   4.2247,   0.0669]],

         [[  3.6238,  -5.6671,  -4.1270,   1.9999],
          [  4.3497,  -3.8131,  -3.6954,   2.5138],
          [  4.2289,   4.4896,  -0.8233,  10.3737],
          [ -9.1522,  -8.0464,  -2.1834,   1.3475]],

         [[ -5.4871,  -5.6456,   2.3971,  -8.8393],
          [  6.0641,  -2.0258,  -7.5135,   0.3814],
          [ -4.3724,  -1.9445,   6.8157,   6.4477],
          [  2.1722,   4.3881,  -2.5686,  -2.4257]]],


        [[[ -1.7312,  -2.5755,   1.5300,  10.9412],
          [  9.6599,  -6.6948,   4.7075,   4.2106],
          [ -0.8718,  -5.4295,  10.0747, -10.1636],
          [  4.5319,   4.0764,   1.6741,   5.8974]],

         [[  0.9215,  -3.5721,  -1.2668,   6.4006],
          [  0.6660,   4.1247,   3.5245,  -4.6866],
          [ -1.0934,  10.4278,  -0.3531,  -5.8575],
          [ -1.1816,  -6.7908,  -6.1499,  -2.0547]],

         [[ -3.9967,  -4.4300,  -0.2993,   5.8606],
          [ -0.9812,  -1.8121,   3.4439,   7.4879],
          [ -2.4948,  -4.9388,  -1.7896,  -3.9585],
          [  6.3013,   2.1417,   4.0991,  -1.6199]]]]))

        let fwdy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]])
        let fwdy = fwdy.ForwardDiff(Tensor.Create([[[[-1.1954e+01,  2.6855e+00],
          [-1.4551e+00, -1.6017e+00]],

         [[ 1.7954e+00,  1.5183e+01],
          [-5.1061e-01, -4.2037e+00]],

         [[-5.7741e+00, -9.1211e-01],
          [-4.1928e+00, -1.1891e+01]]],


        [[[-4.8966e+00, -7.3858e-03],
          [ 6.5086e+00,  6.6678e-01]],

         [[ 2.2310e-01,  7.3424e+00],
          [ 5.5643e-01,  1.2690e+01]],

         [[-5.4125e+00, -3.2977e+00],
          [ 3.1655e+00, -9.4486e+00]]]]))

        let fwdz = Tensor.Conv2D(fwdx, fwdy, stride=2, padding=2, dilation=3)
        let fwdzCorrect = Tensor.Create([[[[-14.6076,  16.4301,  21.7161],
          [ 75.2237, 171.5346,  -5.3082],
          [  5.6034,  25.6748, -22.2133]],

         [[ 19.5075, -47.7868, -33.1086],
          [ 42.3128, -77.9976, 102.6402],
          [ 39.4306,  14.2866, -74.9929]]],


        [[[  6.0893,   9.7672,   1.4466],
          [ 16.2573, -69.7829,  22.9131],
          [-29.0139,  63.6238,  22.9648]],

         [[ -2.1946,  38.6560,  -1.0571],
          [ 59.6885, -29.8678, -25.2544],
          [  4.1600, -55.7119,  22.2137]]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[[-192.4834,  -40.2855,    1.3868],
          [-249.6178,  258.1780,  -22.2914],
          [ 213.6440,  208.5909,   -3.8136]],

         [[-109.4333,   48.9210, -117.9083],
          [  86.2960,   44.9616,  102.1127],
          [  54.5835,   67.3117,  -64.1436]]],


        [[[ 123.7054, -174.0053,  -10.9364],
          [ -86.5387,  196.1334,   92.6273],
          [  41.4041, -196.3940,   54.1555]],

         [[   3.0167,  128.8499,   66.0478],
          [  25.9553,  143.8179,   65.0520],
          [  64.2762,   28.9000,   69.2620]]]])

        let revx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
          [ -4.7983,   3.5090,  -6.9395,  -0.1943],
          [ -3.4793,  -4.3857,  -4.2665,   0.2690],
          [ -4.9017,   0.4501,   8.7836,   0.8665]],

         [[ -1.1159,   4.5843,   1.4441,   2.6304],
          [ -1.3113,   2.3928,   2.8665,   2.8928],
          [  2.4570,   5.5207,   4.3569,   3.0159],
          [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

         [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
          [  0.1634,   8.4872,   2.0171,   0.4871],
          [ -4.7028,  -2.1992,   5.4033,   6.1017],
          [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


        [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
          [ -1.2660,  -1.9108,  -0.3337,   1.1744],
          [ -1.8766,  -1.5132,  -3.0659,   6.0395],
          [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

         [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
          [ -1.0585,  -4.2271,  -0.6372,   4.9192],
          [  0.1994,  -0.9833,   2.8143,  -2.2687],
          [ -3.2098,   0.3120,   1.9338,   5.3132]],

         [[  3.6207,  -2.5295,   2.7143,  -0.8815],
          [  7.1561,  -5.2819,   0.5426,   6.1291],
          [  3.4090,  -0.8490,  -4.4021,  -1.1141],
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).ReverseDiff()
        let revy = Tensor.Create([[[[-2.1628, 15.5045],
          [-2.8062, -5.8116]],

         [[-1.4585, 10.9059],
          [ 0.0661,  0.5168]],

         [[-4.6428, -6.0905],
          [ 1.0177,  0.5360]]],


        [[[ 1.1875, -2.9886],
          [ 7.6836, -5.2942]],

         [[-4.8894,  3.3792],
          [ 2.7905, -4.1603]],

         [[-8.9989, -3.4869],
          [ 6.0547,  5.6603]]]]).ReverseDiff()
        let revz = Tensor.Conv2D(revx, revy, stride=2, padding=2, dilation=3)
        let revzCorrect = Tensor.Create([[[[-14.6076,  16.4301,  21.7161],
          [ 75.2237, 171.5346,  -5.3082],
          [  5.6034,  25.6748, -22.2133]],

         [[ 19.5075, -47.7868, -33.1086],
          [ 42.3128, -77.9976, 102.6402],
          [ 39.4306,  14.2866, -74.9929]]],


        [[[  6.0893,   9.7672,   1.4466],
          [ 16.2573, -69.7829,  22.9131],
          [-29.0139,  63.6238,  22.9648]],

         [[ -2.1946,  38.6560,  -1.0571],
          [ 59.6885, -29.8678, -25.2544],
          [  4.1600, -55.7119,  22.2137]]]])
        revz.Reverse(Tensor.Create([[[[-4.9500, -0.1235, -1.4549],
          [ 1.0173, -3.8754,  8.1473],
          [-1.1222, -2.0444, -3.8435]],

         [[ 2.1598, -8.1827,  1.4084],
          [-6.0393,  3.0670,  0.2767],
          [-6.9600,  2.8638,  1.4368]]],


        [[[-7.1100,  1.2635,  2.4884],
          [ 0.8497,  0.5392,  1.4672],
          [-2.3189, -5.4578, -0.3853]],

         [[-4.2101,  6.2027,  3.0791],
          [ 3.7285, -0.2306, -0.2753],
          [ 0.1230, -4.0682, -3.4253]]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[[ 12.0238,  33.8209, -17.2926, -69.2517],
          [-62.5262,  17.3330,  14.9041,  44.0385],
          [  7.8226,   3.4009,  10.0191, -40.2569],
          [ 34.4406,  26.0614, -20.7367,   6.2848]],

         [[ -9.3435,  -9.3136, -13.2356, -31.9008],
          [-22.8417, -11.5434,   3.8338,  33.9789],
          [-11.0206, -35.7580,  -1.4193, -12.6192],
          [  8.3021,  25.6510,   1.3110, -14.7624]],

         [[ -9.6068,  14.8629, -40.3168,  12.9088],
          [-49.6693,   9.5719,   7.0467, -46.3823],
          [-16.2796,  31.1040,   4.9152,   2.4657],
          [ 14.6258, -33.6386,   9.9669,  15.2829]]],


        [[[ -1.4400,   2.0322,  -3.5001,   9.0489],
          [ 44.1135,  63.6098,  16.6760, -40.1814],
          [  6.9733, -36.3203,  -3.2342, -72.4627],
          [ -3.2851, -24.6777,  -6.2321,  -1.9125]],

         [[  0.3412,  21.8664,  -0.7940,   5.1009],
          [ 17.3920,  13.8411,   8.7568, -25.1523],
          [ 27.8511, -24.8737,  17.3097, -73.2700],
          [ -0.6079, -15.0725,  -0.6711,   1.2381]],

         [[ -0.4279, -18.1763,  -4.3348,  -2.4797],
          [ 38.8413, -27.6409,  21.1756,  35.7860],
          [ 61.9495,  13.6942,  32.6135,  47.4267],
          [ -0.8477,  21.5595,  -0.1735,  -1.0164]]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[[ 1.4095e+01, -5.9958e+01],
          [ 9.4394e+01, -1.1459e+01]],

         [[-1.2466e+01, -4.6630e+00],
          [-2.3808e+01,  4.1206e+01]],

         [[-5.1847e+01,  1.3497e+00],
          [ 5.3702e+01,  2.1704e+01]]],


        [[[ 4.5069e+00,  1.3558e+01],
          [ 8.9010e+00,  1.3353e+00]],

         [[ 1.7762e+00, -3.7820e+01],
          [-5.1715e+00, -2.4090e+01]],

         [[-7.5394e+00,  1.7613e+01],
          [ 3.7940e+01,  8.9327e-02]]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

    // [<Test>]
    // member this.TestDerivativeConv2Ds23p32d23 () =
    //     let fwdx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
    //       [ -4.7983,   3.5090,  -6.9395,  -0.1943],
    //       [ -3.4793,  -4.3857,  -4.2665,   0.2690],
    //       [ -4.9017,   0.4501,   8.7836,   0.8665]],

    //      [[ -1.1159,   4.5843,   1.4441,   2.6304],
    //       [ -1.3113,   2.3928,   2.8665,   2.8928],
    //       [  2.4570,   5.5207,   4.3569,   3.0159],
    //       [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

    //      [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
    //       [  0.1634,   8.4872,   2.0171,   0.4871],
    //       [ -4.7028,  -2.1992,   5.4033,   6.1017],
    //       [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


    //     [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
    //       [ -1.2660,  -1.9108,  -0.3337,   1.1744],
    //       [ -1.8766,  -1.5132,  -3.0659,   6.0395],
    //       [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

    //      [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
    //       [ -1.0585,  -4.2271,  -0.6372,   4.9192],
    //       [  0.1994,  -0.9833,   2.8143,  -2.2687],
    //       [ -3.2098,   0.3120,   1.9338,   5.3132]],

    //      [[  3.6207,  -2.5295,   2.7143,  -0.8815],
    //       [  7.1561,  -5.2819,   0.5426,   6.1291],
    //       [  3.4090,  -0.8490,  -4.4021,  -1.1141],
    //       [  3.1586,   1.6269,   4.5772,  -4.8104]]]])
    //     let fwdx = fwdx.ForwardDiff(Tensor.Create([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
    //       [  4.1074,  12.5312,  -3.2430,   4.4820],
    //       [-10.0428,   5.0817,   0.4602,  -0.9825],
    //       [  4.5867,   1.2517,   4.2247,   0.0669]],

    //      [[  3.6238,  -5.6671,  -4.1270,   1.9999],
    //       [  4.3497,  -3.8131,  -3.6954,   2.5138],
    //       [  4.2289,   4.4896,  -0.8233,  10.3737],
    //       [ -9.1522,  -8.0464,  -2.1834,   1.3475]],

    //      [[ -5.4871,  -5.6456,   2.3971,  -8.8393],
    //       [  6.0641,  -2.0258,  -7.5135,   0.3814],
    //       [ -4.3724,  -1.9445,   6.8157,   6.4477],
    //       [  2.1722,   4.3881,  -2.5686,  -2.4257]]],


    //     [[[ -1.7312,  -2.5755,   1.5300,  10.9412],
    //       [  9.6599,  -6.6948,   4.7075,   4.2106],
    //       [ -0.8718,  -5.4295,  10.0747, -10.1636],
    //       [  4.5319,   4.0764,   1.6741,   5.8974]],

    //      [[  0.9215,  -3.5721,  -1.2668,   6.4006],
    //       [  0.6660,   4.1247,   3.5245,  -4.6866],
    //       [ -1.0934,  10.4278,  -0.3531,  -5.8575],
    //       [ -1.1816,  -6.7908,  -6.1499,  -2.0547]],

    //      [[ -3.9967,  -4.4300,  -0.2993,   5.8606],
    //       [ -0.9812,  -1.8121,   3.4439,   7.4879],
    //       [ -2.4948,  -4.9388,  -1.7896,  -3.9585],
    //       [  6.3013,   2.1417,   4.0991,  -1.6199]]]]))

    //     let fwdy = Tensor.Create([[[[-2.1628, 15.5045],
    //       [-2.8062, -5.8116]],

    //      [[-1.4585, 10.9059],
    //       [ 0.0661,  0.5168]],

    //      [[-4.6428, -6.0905],
    //       [ 1.0177,  0.5360]]],


    //     [[[ 1.1875, -2.9886],
    //       [ 7.6836, -5.2942]],

    //      [[-4.8894,  3.3792],
    //       [ 2.7905, -4.1603]],

    //      [[-8.9989, -3.4869],
    //       [ 6.0547,  5.6603]]]])
    //     let fwdy = fwdy.ForwardDiff(Tensor.Create([[[[-1.1954e+01,  2.6855e+00],
    //       [-1.4551e+00, -1.6017e+00]],

    //      [[ 1.7954e+00,  1.5183e+01],
    //       [-5.1061e-01, -4.2037e+00]],

    //      [[-5.7741e+00, -9.1211e-01],
    //       [-4.1928e+00, -1.1891e+01]]],


    //     [[[-4.8966e+00, -7.3858e-03],
    //       [ 6.5086e+00,  6.6678e-01]],

    //      [[ 2.2310e-01,  7.3424e+00],
    //       [ 5.5643e-01,  1.2690e+01]],

    //      [[-5.4125e+00, -3.2977e+00],
    //       [ 3.1655e+00, -9.4486e+00]]]]))

    //     let fwdz = Tensor.Conv2D(fwdx, fwdy, stride=[2;3], padding=[3;2], dilation=[2;3])
    //     let fwdzCorrect = Tensor.Create([[[[   0.0000,    0.0000],
    //       [ -14.6076,   -1.0516],
    //       [  37.1832,  -38.3543],
    //       [  19.1313,  -71.5200]],

    //      [[   0.0000,    0.0000],
    //       [  19.5075,   85.0262],
    //       [   3.1274,   18.8062],
    //       [ -17.7834, -153.0469]]],


    //     [[[   0.0000,    0.0000],
    //       [   6.0893,   -0.2928],
    //       [  -5.8958,   54.1834],
    //       [-104.2242,    5.6230]],

    //      [[   0.0000,    0.0000],
    //       [  -2.1946,  -58.4576],
    //       [  51.1217,   28.2248],
    //       [  14.2171,  -23.6501]]]])
    //     let fwdzd = fwdz.Derivative
    //     let fwdzdCorrect = Tensor.Create([[[[   0.0000,    0.0000],
    //       [-192.4834,  -79.3922],
    //       [   8.7453, -156.1497],
    //       [  18.6200,  -74.9842]],

    //      [[   0.0000,    0.0000],
    //       [-109.4333,  124.4153],
    //       [ -17.4147,   50.3533],
    //       [ -27.9149,  -67.3966]]],


    //     [[[   0.0000,    0.0000],
    //       [ 123.7054,   44.3007],
    //       [-148.8877,   55.1096],
    //       [ -37.5733,   57.6518]],

    //      [[   0.0000,    0.0000],
    //       [   3.0167,  -82.4102],
    //       [  29.8377,   14.8331],
    //       [ -45.6256,   40.8957]]]])

    //     let revx = Tensor.Create([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
    //       [ -4.7983,   3.5090,  -6.9395,  -0.1943],
    //       [ -3.4793,  -4.3857,  -4.2665,   0.2690],
    //       [ -4.9017,   0.4501,   8.7836,   0.8665]],

    //      [[ -1.1159,   4.5843,   1.4441,   2.6304],
    //       [ -1.3113,   2.3928,   2.8665,   2.8928],
    //       [  2.4570,   5.5207,   4.3569,   3.0159],
    //       [ -3.4414,   8.1672,  -3.8442,  -1.4704]],

    //      [[ -0.1417,   1.0840,  -3.7151,  -1.2461],
    //       [  0.1634,   8.4872,   2.0171,   0.4871],
    //       [ -4.7028,  -2.1992,   5.4033,   6.1017],
    //       [ -2.7881,  12.6291,   3.3191,  -1.7654]]],


    //     [[[ -7.0889,  -2.6943, -10.4957,  -4.1073],
    //       [ -1.2660,  -1.9108,  -0.3337,   1.1744],
    //       [ -1.8766,  -1.5132,  -3.0659,   6.0395],
    //       [ -0.4991,  -6.3026,  -2.8313,  -1.7206]],

    //      [[ -8.4025,   0.4552,  -0.0573,  -3.3758],
    //       [ -1.0585,  -4.2271,  -0.6372,   4.9192],
    //       [  0.1994,  -0.9833,   2.8143,  -2.2687],
    //       [ -3.2098,   0.3120,   1.9338,   5.3132]],

    //      [[  3.6207,  -2.5295,   2.7143,  -0.8815],
    //       [  7.1561,  -5.2819,   0.5426,   6.1291],
    //       [  3.4090,  -0.8490,  -4.4021,  -1.1141],
    //       [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).ReverseDiff()
    //     let revy = Tensor.Create([[[[-2.1628, 15.5045],
    //       [-2.8062, -5.8116]],

    //      [[-1.4585, 10.9059],
    //       [ 0.0661,  0.5168]],

    //      [[-4.6428, -6.0905],
    //       [ 1.0177,  0.5360]]],


    //     [[[ 1.1875, -2.9886],
    //       [ 7.6836, -5.2942]],

    //      [[-4.8894,  3.3792],
    //       [ 2.7905, -4.1603]],

    //      [[-8.9989, -3.4869],
    //       [ 6.0547,  5.6603]]]]).ReverseDiff()
    //     let revz = Tensor.Conv2D(revx, revy, stride=[2;3], padding=[3;2], dilation=[2;3])
    //     let revzCorrect = Tensor.Create([[[[   0.0000,    0.0000],
    //       [ -14.6076,   -1.0516],
    //       [  37.1832,  -38.3543],
    //       [  19.1313,  -71.5200]],

    //      [[   0.0000,    0.0000],
    //       [  19.5075,   85.0262],
    //       [   3.1274,   18.8062],
    //       [ -17.7834, -153.0469]]],


    //     [[[   0.0000,    0.0000],
    //       [   6.0893,   -0.2928],
    //       [  -5.8958,   54.1834],
    //       [-104.2242,    5.6230]],

    //      [[   0.0000,    0.0000],
    //       [  -2.1946,  -58.4576],
    //       [  51.1217,   28.2248],
    //       [  14.2171,  -23.6501]]]])
    //     revz.Reverse(Tensor.Create([[[[ 9.3434,  2.6928],
    //       [ 1.0891, -3.7427],
    //       [-1.4629,  2.3710],
    //       [-6.0087,  0.0640]],

    //      [[ 0.8308,  2.2070],
    //       [-1.5825, -2.8429],
    //       [-5.6184, -4.6618],
    //       [-3.7099,  7.1608]]],


    //     [[[ 2.3090,  8.8259],
    //       [-4.7131, -0.8071],
    //       [ 0.6332,  2.8465],
    //       [ 2.4269,  4.5666]],

    //      [[ 1.6529,  0.7295],
    //       [ 5.1375, 10.5018],
    //       [-6.1957,  4.4722],
    //       [-5.1692,  7.7524]]]]))            
    //     let revxd = revx.Derivative
    //     let revxdCorrect = Tensor.Create([[[[  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000, 174.2975,   0.0000,   0.0000],
    //       [  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000, -24.2263,   0.0000,   0.0000]],

    //      [[  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000,  63.0568,   0.0000,   0.0000],
    //       [  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000, -28.0555,   0.0000,   0.0000]],

    //      [[  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000,  28.4857,   0.0000,   0.0000],
    //       [  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000,   2.3057,   0.0000,   0.0000]]],


    //     [[[  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000, 105.5149,   0.0000,   0.0000],
    //       [  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000, 171.8324,   0.0000,   0.0000]],

    //      [[  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000, -74.8343,   0.0000,   0.0000],
    //       [  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000,  93.7225,   0.0000,   0.0000]],

    //      [[  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000,  20.4351,   0.0000,   0.0000],
    //       [  0.0000,   0.0000,   0.0000,   0.0000],
    //       [  0.0000,  94.8045,   0.0000,   0.0000]]]])
    //     let revyd = revy.Derivative
    //     let revydCorrect = Tensor.Create([[[[-7.9097e+00, -1.0028e+01],
    //       [ 2.5043e+01,  1.3009e+01]],

    //      [[-3.0224e+00,  5.3923e-02],
    //       [ 2.9515e+01,  4.7731e+01]],

    //      [[ 1.3413e+01,  1.6094e+01],
    //       [-1.3050e+00,  5.1869e+01]]],


    //     [[[ 2.1210e+01,  1.8179e+01],
    //       [-5.6601e+01, -1.2165e+01]],

    //      [[-3.0691e+01, -2.4564e+01],
    //       [-1.7412e+01, -5.2216e+01]],

    //      [[-4.5804e+01, -5.3832e+01],
    //       [ 5.4477e+01, -8.4577e+01]]]])

    //     Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
    //     Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
    //     Assert.True(revz.ApproximatelyEqual(revzCorrect))
    //     Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
    //     Assert.True(revyd.ApproximatelyEqual(revydCorrect))

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdaCorrect, revxda)
        Assert.AreEqual(revxdbCorrect, revxdb)
        Assert.AreEqual(revxdcCorrect, revxdc)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

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
    member this.TestDerivativeDilateT () =
        let fwdx = Tensor.Create([[1.;2.];[3.;4.]]).ForwardDiff(Tensor.Create([[10.;20.];[30.;40.]]))
        let fwdz = fwdx.Dilate([|2; 2|])
        let fwdzCorrect =  Tensor.Create([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect =  Tensor.Create([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]])

        let revx = Tensor.Create([[1.;2.];[3.;4.]]).ReverseDiff()
        let revz = revx.Dilate([|2; 2|])
        let revzCorrect =  Tensor.Create([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]])
        revz.Reverse(Tensor.Create([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[10.;20.];[30.;40.]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    [<Test>]
    member this.TestDerivativeUndilateT () =
        let fwdx = Tensor.Create([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]]).ForwardDiff(Tensor.Create([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]]))
        let fwdz = fwdx.Undilate([|2; 2|])
        let fwdzCorrect =  Tensor.Create([[1.;2.];[3.;4.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect =  Tensor.Create([[10.;20.];[30.;40.]])

        let revx = Tensor.Create([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]]).ReverseDiff()
        let revz = revx.Undilate([|2; 2|])
        let revzCorrect =  Tensor.Create([[1.;2.];[3.;4.]])
        revz.Reverse(Tensor.Create([[10.;20.];[30.;40.]]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]])

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

[<TestFixture>]
module TestOps =

    type Tensor with

        static member mulx (a,b) = 
            Tensor.Op
                { new BinaryOp with 
                    member _.Compute(a,b) = a.MulTT(b)
                    member _.GradForwardTT(fab,ap,ad,bp,bd) = (Tensor.mulx(ad, bp)) + Tensor.mulx(ap, bd)
                    member _.GradForwardTC(fab,_a,ad,b) = Tensor.mulx(ad, b)
                    member _.GradForwardCT(fab,a,_b,bd) = Tensor.mulx(a, bd)
                    member _.GradReverseTT(t,a,b) = Tensor.mulx(t.Derivative, b.Primal), Tensor.mulx(t.Derivative, a.Primal)
                    member _.GradReverseTC(t,_a,b) = Tensor.mulx(t.Derivative, b)
                    member _.GradReverseCT(t,a,_b) = Tensor.mulx(t.Derivative, a) }
                (a,b)
    [<Test>]
    let ``test mulx extension``() = 
        let fwdx = Tensor.Create([1.; 2.; 3.]).ForwardDiff(Tensor.Create([2.; 3.; 4.]))
        let fwdy = Tensor.Create([5.; 6.; 7.]).ForwardDiff(Tensor.Create([2.; 2.; 3.]))
        let fwdz = Tensor.mulx(fwdx, fwdy)
        let fwdzCorrect = Tensor.Create([5.; 12.; 21.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([12.; 22.; 37.])

        let revx = Tensor.Create([1.; 2.; 3.]).ReverseDiff()
        let revy = Tensor.Create([5.; 6.; 7.]).ReverseDiff()
        let revz = Tensor.mulx(revx, revy)
        let revzCorrect = Tensor.Create([5.; 12.; 21.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([25.; 30.; 35.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([5.; 10.; 15.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)
        Assert.AreEqual(revydCorrect, revyd)

    type Tensor with
        member a.sinx() = 
            Tensor.Op
               { new UnaryOp with 
                    member _.Compute(a) = a.SinT()
                    member _.GradForward(_fa,a,da) = da * a.cosx()
                    member _.GradReverse (fa,a) = fa.Derivative * a.Primal.cosx() }
               a

        member a.cosx() = 
            Tensor.Op
               { new UnaryOp with 
                    member _.Compute(a) = a.CosT()
                    member _.GradForward(_fa,a,da) = -da * a.sinx()
                    member _.GradReverse(fa,a) = -fa.Derivative * a.Primal.sinx() }
                a


    [<Test>]
    let ``test sinx extension``() = 
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ForwardDiff(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.sinx()
        let fwdzCorrect = Tensor.Create([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.0022; 0.0237; 1.4571; 1.0510; 0.3840])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).ReverseDiff()
        let revz = revx.sinx()
        let revzCorrect = Tensor.Create([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([2.9194;  0.4080;  4.8988;  4.1774; -3.3228])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))

    type Tensor with
        static member conv1dx(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int) = 
            let stride = defaultArg stride 1
            let padding = defaultArg padding 0
            let dilation = defaultArg dilation 1
            let mutable b = b
            if dilation > 1 then
                b <- b.Dilate([|1;1;dilation|])
            Tensor.Op
                { new BinaryOp with 
                    member _.Compute(a,b) = a.Conv1D(b, stride, padding)
                    member _.GradForwardTT(_fab,a,da,b,db) = Tensor.conv1dx(da, b, stride, padding) + Tensor.conv1dx(a, db, stride, padding)
                    member _.GradForwardTC(_fab,_a,da,b) = Tensor.conv1dx(da, b, stride, padding)
                    member _.GradForwardCT(_fab,a,_b,db) = Tensor.conv1dx(a, db, stride, padding)
                    member _.GradReverseTT(fab,a,b) = 
                        let batchSize = fab.Shape.[0]
                        let outputChannels = fab.Shape.[1]
                        let inputChannels = a.Shape.[1]
                        let inputLength = a.Shape.[2]
                        let kernelLength = b.Shape.[2]
                        let mutable tderivative = fab.Derivative
                        if stride > 1 then
                            tderivative <- tderivative.Dilate([|1;1;stride|])
                        let bFlipped = b.Primal.Flip([|2|])
                        let mutable aderivative = Tensor.ZerosLike(a)
                        for k=0 to outputChannels-1 do
                            let b = bFlipped.[k].View([|inputChannels; 1; kernelLength|])
                            let dBounds: int[,] = array2D [[0; batchSize-1; 1]; [k; k; 1]; [0; tderivative.Shape.[2]-1; 1]]
                            let d = tderivative.GetSlice(dBounds).View([|batchSize; 1; -1|])
                            let mutable c = Tensor.conv1dx(d, b, padding=kernelLength-1)
                            if padding > 0 then
                                let cBounds = array2D [[0; batchSize-1; 1]; [0; inputChannels-1; 1]; [padding; c.Shape.[2]-1-padding; 1]]
                                c <- c.GetSlice(cBounds).View([|batchSize; inputChannels; -1|])
                            aderivative <- aderivative + c
                        let mutable bderivative = Tensor.ZerosLike(b)
                        for n=0 to batchSize-1 do
                            let aa = a.Primal.[n].View([|inputChannels; 1; inputLength|]) 
                            let d = tderivative.[n]
                            for k=0 to outputChannels-1 do
                                let dd = d.[k].View([|1; 1; tderivative.Shape.[2]|])
                                let c = Tensor.conv1dx(aa, dd, padding=padding).View([|1; inputChannels; kernelLength|])
                                bderivative <- Tensor.AddSlice(bderivative, [|k; 0; 0|], c)
                        (aderivative, bderivative)
                    member _.GradReverseTC(t,_a,b) = 
                        let batchSize = t.Shape.[0]
                        let outputChannels = t.Shape.[1]
                        let inputChannels = a.Shape.[1]
                        let kernelLength = b.Shape.[2]
                        let mutable tderivative = t.Derivative
                        if stride > 1 then
                            tderivative <- tderivative.Dilate([|1;1;stride|])
                        let bFlipped = b.Flip([|2|])
                        let mutable aderivative = Tensor.ZerosLike(a)
                        for k=0 to outputChannels-1 do
                            let b = bFlipped.[k].View([|inputChannels; 1; kernelLength|])
                            let dBounds = array2D [[0; batchSize-1; 1]; [k; k; 1]; [0; tderivative.Shape.[2]-1; 1]]
                            let d = tderivative.GetSlice(dBounds).View([|batchSize; 1; -1|])
                            let mutable c = Tensor.conv1dx(d, b, padding=kernelLength-1)
                            if padding > 0 then
                                let cBounds = array2D [[0; batchSize-1; 1]; [0; inputChannels-1; 1]; [padding; c.Shape.[2]-1-padding; 1]]
                                c <- c.GetSlice(cBounds).View([|batchSize; inputChannels; -1|])
                            aderivative <- aderivative + c
                        aderivative
                    member _.GradReverseCT(t,a,_b) = 
                        let batchSize = t.Shape.[0]
                        let outputChannels = t.Shape.[1]
                        let inputChannels = a.Shape.[1]
                        let inputLength = a.Shape.[2]
                        let kernelLength = b.Shape.[2]
                        let mutable tderivative = t.Derivative
                        if stride > 1 then
                            tderivative <- tderivative.Dilate([|1;1;stride|])
                        let mutable bderivative = Tensor.ZerosLike(b)
                        for n=0 to batchSize-1 do
                            let aa = a.[n].View([|inputChannels; 1; inputLength|])
                            let d = tderivative.[n]
                            for k=0 to outputChannels-1 do
                                let dd = d.[k].View([|1; 1; tderivative.Shape.[2]|])
                                let c = Tensor.conv1dx(aa, dd, padding=padding).View([|1; inputChannels; kernelLength|])
                                bderivative <- Tensor.AddSlice(bderivative, [|k; 0; 0|], c)
                        bderivative }
                (a,b)

    [<Test>]
    let ``test conv1dx extension``() = 
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

        let fwdz = Tensor.conv1dx(fwdx, fwdy, stride=1)
        let fwdzCorrect = Tensor.Create([[[ 143.3192;  108.0332;   11.2241];
                                         [  -5.9062;    4.6091;    6.0273]];

                                        [[  27.3032;   97.9855; -133.8372];
                                         [  -1.4792;   45.6659;   29.8705]]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[[ 111.2865;  -40.3692;   -1.8573];
                                         [   -1.9154;   43.3470;   29.3626]];

                                        [[ -168.6758;  -43.1578;   25.4470];
                                         [ -149.6851;   23.1963;  -50.1932]]])

        let revx = Tensor.Create([[[ 2.8564;  0.0424;  7.0984; -2.5130];
                                 [-1.1502;  0.1410;  2.5438;  4.4798];
                                 [ 0.4381; -4.3649;  2.5502;  2.5141]];

                                [[-2.8894; -7.1729; -7.1368;  1.1060];
                                 [-1.3253;  0.0257; -2.8552; -0.4933];
                                 [ 4.7305; -5.6787;  3.4658;  4.5768]]]).ReverseDiff()
        let revy = Tensor.Create([[[ 0.6355; -5.8100];
                                 [ 0.6244;  6.0336];
                                 [ 4.8205;  1.1716]];

                                [[-8.2315; -3.0400];
                                 [-2.2282; -2.9084];
                                 [-0.9613;  1.0958]]]).ReverseDiff()
        let revz = Tensor.conv1dx(revx, revy, stride=1)
        let revzCorrect = Tensor.Create([[[ -1.3005; -43.8321;  62.9678];
                                         [-26.6931; -22.6506; -69.1848]];

                                        [[ 55.3161;  -3.6187;   6.3480];
                                         [ 37.6982;  98.2438;  64.8643]]])
        revz.Reverse(Tensor.Create([[[ 4.5763;  2.7538;  2.0173];
                                     [-2.7543;  7.9257; -1.3670]];

                                    [[ 1.7997; -1.2354;  4.6313];
                                     [-4.0646;  0.0384;  4.1437]]]))            
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([[[ 25.5806; -81.7051; -27.5597;  -7.5648];
                                         [  8.9949;  19.6812;  -2.1304;  16.1472];
                                         [ 24.7076;   7.9984;  22.9497;   0.8655]];

                                        [[ 34.6019;   0.7992; -24.1050; -39.5052];
                                         [ 10.1808;  21.8231; -13.9067;  15.8920];
                                         [ 12.5828;  -8.3376;  16.9365;   9.9666]]])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([[[ -1.8835;  15.7019];
                                         [-15.3840;  17.9761];
                                         [ 26.7091;  -1.1857]];

                                        [[-35.3382;  93.0419];
                                         [ -5.6351;  11.3910];
                                         [-44.3729;  70.9775]]])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
        Assert.True(revyd.ApproximatelyEqual(revydCorrect))

