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
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdy = dsharp.tensor([5.; 6.; 7.]).forwardDiff(dsharp.tensor([2.; 2.; 3.]))
        let fwdz = fwdx + fwdy
        let fwdzCorrect = dsharp.tensor([6.; 8.; 10.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([4.; 5.; 7.])

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revy = dsharp.tensor([5.; 6.; 7.]).reverseDiff()
        let revz = revx + revy
        let revzCorrect = dsharp.tensor([6.; 8.; 10.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([5.; 5.; 5.])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([5.; 5.; 5.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)
        Assert.AreEqual(revydCorrect, revyd)

    [<Test>]
    member this.TestDerivativeAddT2T1 () =
        let fwdx = dsharp.tensor([[1.; 2.]; [3.; 4.]]).forwardDiff(dsharp.tensor([[2.; 3.]; [4.; 5.]]))
        let fwdy = dsharp.tensor([5.; 6.]).forwardDiff(dsharp.tensor([2.; 3.]))
        let fwdz = fwdx + fwdy
        let fwdzCorrect = dsharp.tensor([[6.; 8.]; [8.; 10.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[4.; 6.]; [6.; 8.]])

        let revx = dsharp.tensor([[1.; 2.]; [3.; 4.]]).reverseDiff()
        let revy = dsharp.tensor([5.; 6.]).reverseDiff()
        let revz = revx + revy
        let revzCorrect = dsharp.tensor([[6.; 8.]; [8.; 10.]])
        revz.reverse(dsharp.tensor([[2.; 3.]; [4.; 5.]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[2.; 3.]; [4.; 5.]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([6.; 8.])

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
    member this.TestDerivativeExpand () =
        let fwdx = dsharp.tensor([[1.]; [2.]]).forwardDiff(dsharp.tensor([[5.]; [6.]])) // 2x1
        let fwdz = fwdx.expand([2;2;2]) // 2x2x2 = [[[1.;1]; [2.;2]]; [[1.;1]; [2.;2]]]
        let fwdzCorrect = dsharp.tensor([[[1.;1.]; [2.;2.]]; [[1.;1.]; [2.;2.]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor ([[[5., 5.], [6., 6.]], [[5., 5.], [6., 6.]]])

        (* Python:
        import torch 
        t1 = torch.tensor([[1.], [2.]], requires_grad=True)
        revz = t1.expand([2,2,2])
        revz.backward(torch.tensor([[[3.,3.], [6.,6.]], [[3.,3.], [6.,6.]]]))
        t1.grad
        --> tensor([[12.],[24.]])
        *)
        let revx = dsharp.tensor([[1.]; [2.]]).reverseDiff()
        let revz = revx.expand([2;2;2])
        let revzCorrect = dsharp.tensor([[[1.;1.]; [2.;2.]]; [[1.;1.]; [2.;2.]]])
        revz.reverse(dsharp.tensor([[[3.;3.]; [6.;6.]]; [[3.;3.]; [6.;6.]]]))
        let revxd = revx.derivative
        // Note: The 4x'3' accumulate to the first entry, the 4x'6' accumulate to the second entry
        let revxdCorrect = dsharp.tensor [[12.], [24.]]

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd,fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd,revxdCorrect)

    [<Test>]
    member this.TestAddWithBroadcastSystematic () =
        // This is a somewhat adhoc extra test to do a whole range of additiosn
        // with broadcast, mainly to check that not problems occur in taking the
        // derivatives.
        //
        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 + 1  (broadcast --> 2x3x4)
        // 2x3x4 + 4  (broadcast --> 2x3x4)
        // 2x3x4 + 1x1  (broadcast --> 2x3x4)
        // 2x3x4 + 3x1  (broadcast --> 2x3x4)
        // 2x3x4 + 1x4  (broadcast --> 2x3x4)
        // etc.
        let t1a = dsharp.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
                                  [ [13.; 14.; 15.; 16.]; [17.; 18.; 19.; 20.]; [21.; 22.; 23.; 24.] ]  ])
        
        // Get all the interesting shapes that broadcast into t1a
        let shapes = 
            [ for i1 in [0;1;2] do
                for i2 in [0;1;3] do
                  for i3 in [0;1;4] do 
                    if i1 <> 2 || i2 <> 3 || i3 <> 4 then
                        [| if i1 <> 0 && i2 <> 0 && i3 <> 0 then yield i1
                           if i2 <> 0 && i3 <> 0 then yield i2
                           if i3 <> 0 then yield i3 |] ]
            |> List.distinct

        // For each shape, create a broadcasting addition and take forward and reverse derivatives
        for shape in shapes do 
            let t1b = dsharp.tensor( Util.arrayND shape (fun is -> double (Array.sum is) + 2.0))
            let t1a_deriv = t1a + 1.0
            let t1b_delta = dsharp.tensor( Util.arrayND shape (fun is -> double (Array.sum is) - 2.0))
            let fwda = t1a.forwardDiff(t1a_deriv)
            let fwdb = t1b.forwardDiff(t1b_delta)
            let fwdz = fwda + fwdb
            let fwdzd = fwdz.derivative

            let revx = t1a.reverseDiff()
            let revy = t1b.reverseDiff()
            let revz = revx + revy
            let revz_grad = t1a - 1.0
            revz.reverse(revz_grad)
            let revxd = revx.derivative
            let revyd = revy.derivative

            // In the simple case of broadcasting a constant, check the result against the non-broadcast case
            if t1b.sum() = dsharp.tensor(2.0) then 
                let t1c = dsharp.tensor( Util.arrayND [| 2;3;4 |] (fun _idxs -> 2.0))
                let t1c_deriv = dsharp.tensor( Util.arrayND [| 2;3;4 |] (fun _idxs -> -2.0))
                let fwda = t1a.forwardDiff(t1a_deriv)
                let fwdc = t1c.forwardDiff(t1c_deriv)
                let fwdz2 = fwda + fwdc
                let fwdzd2 = fwdz2.derivative

                let revx2 = t1a.reverseDiff()
                let revy2 = t1c.reverseDiff()
                let revz2 = revx2 + revy2
                revz2.reverse(revz_grad)
                let revxd2 = revx2.derivative
                let revyd2 = revy2.derivative
                Assert.AreEqual(fwdzd,fwdzd2)
                Assert.AreEqual(revxd,revxd2)
                // note the difference in shape here, and the need to summate down
                Assert.AreEqual(revyd.sum(),revyd2.sum())

    [<Test>]
    member this.TestDerivativeSubTT () =
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdy = dsharp.tensor([5.; 6.; 7.]).forwardDiff(dsharp.tensor([2.; 2.; 3.]))
        let fwdz = fwdx - fwdy
        let fwdzCorrect = dsharp.tensor([-4.; -4.; -4.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.; 1.; 1.])

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revy = dsharp.tensor([5.; 6.; 7.]).reverseDiff()
        let revz = revx - revy
        let revzCorrect = dsharp.tensor([-4.; -4.; -4.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([5.; 5.; 5.])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([-5.; -5.; -5.])

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
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdy = dsharp.tensor([5.; 6.; 7.]).forwardDiff(dsharp.tensor([2.; 2.; 3.]))
        let fwdz = fwdx * fwdy
        let fwdzCorrect = dsharp.tensor([5.; 12.; 21.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([12.; 22.; 37.])

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revy = dsharp.tensor([5.; 6.; 7.]).reverseDiff()
        let revz = revx * revy
        let revzCorrect = dsharp.tensor([5.; 12.; 21.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([25.; 30.; 35.])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([5.; 10.; 15.])

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
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdy = dsharp.tensor([5.; 6.; 7.]).forwardDiff(dsharp.tensor([2.; 2.; 3.]))
        let fwdz = fwdx / fwdy
        let fwdzCorrect = dsharp.tensor([0.2; 0.333333; 0.428571])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.32; 0.388889; 0.387755])

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revy = dsharp.tensor([5.; 6.; 7.]).reverseDiff()
        let revz = revx / revy
        let revzCorrect = dsharp.tensor([0.2; 0.333333; 0.428571])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([1.; 0.833333; 0.714286])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([-0.2; -0.277778; -0.306122])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))
    
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
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdy = dsharp.tensor([5.; 6.; 7.]).forwardDiff(dsharp.tensor([2.; 2.; 3.]))
        let fwdz = fwdx ** fwdy
        let fwdzCorrect = dsharp.tensor([1.; 64.; 2187.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([10.; 664.723; 27620.])

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revy = dsharp.tensor([5.; 6.; 7.]).reverseDiff()
        let revz = revx ** revy
        let revzCorrect = dsharp.tensor([1.; 64.; 2187.])
        revz.reverse(dsharp.tensor([5.; 15.; 25.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([25.; 2880.; 127575.])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([0.; 665.421; 60066.6])

        Assert.True(fwdz.allclose(fwdzCorrect,0.1))
        Assert.True(fwdzd.allclose(fwdzdCorrect,0.1))
        Assert.True(revz.allclose(revzCorrect,0.1))
        Assert.True(revxd.allclose(revxdCorrect,0.1))
        Assert.True(revyd.allclose(revydCorrect,0.1))
    
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
        let fwdx = dsharp.tensor([[[  0.1264;   5.3183;   6.6905; -10.6416];
                                 [ 13.8060;   4.5253;   2.8568;  -3.2037];
                                 [ -0.5796;  -2.7937;  -3.3662;  -1.3017]];

                                [[ -2.8910;   3.9349;  -4.3892;  -2.6051];
                                 [  4.2547;   2.6049;  -9.8226;  -5.4543];
                                 [ -0.9674;   1.0070;  -4.6518;   7.1702]]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[-4.3197; -6.5898; -6.2003;  2.1058];
                                 [ 7.0684; -3.7964;  4.4218;  3.9533];
                                 [-7.1559; -7.6799; -9.5234; -3.9351]];

                                [[-0.2089; -7.8695;  6.5383;  5.1090];
                                 [-3.8272;  7.6264;  6.8205;  5.7346];
                                 [ 6.5570;  7.7248;  6.3494; -2.9007]]]))

        let fwdy = dsharp.tensor([[[ 4.0332e+00;  6.3036e+00];
                                 [ 8.4410e+00; -5.7543e+00];
                                 [-5.6937e-03; -6.7241e+00]];

                                [[-2.2619e+00;  1.2082e+00];
                                 [-1.2203e-01; -4.9373e+00];
                                 [-4.1881e+00; -3.4198e+00]]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[-1.5107; -0.0610];
                                 [-0.2609;  5.9220];
                                 [ 2.8221; -5.7314]];

                                [[ 5.0064;  3.8631];
                                 [-4.6264; -7.9380];
                                 [ 8.2204; -1.9833]]]))

        let fwdz = fwdx.conv1d(fwdy, stride=1)
        let fwdzCorrect = dsharp.tensor([[[ 143.3192;  108.0332;   11.2241];
                                         [  -5.9062;    4.6091;    6.0273]];

                                        [[  27.3032;   97.9855; -133.8372];
                                         [  -1.4792;   45.6659;   29.8705]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[ 111.2865;  -40.3692;   -1.8573];
                                         [   -1.9154;   43.3470;   29.3626]];

                                        [[ -168.6758;  -43.1578;   25.4470];
                                         [ -149.6851;   23.1963;  -50.1932]]])

        let revx = dsharp.tensor([[[ 2.8564;  0.0424;  7.0984; -2.5130];
                                 [-1.1502;  0.1410;  2.5438;  4.4798];
                                 [ 0.4381; -4.3649;  2.5502;  2.5141]];

                                [[-2.8894; -7.1729; -7.1368;  1.1060];
                                 [-1.3253;  0.0257; -2.8552; -0.4933];
                                 [ 4.7305; -5.6787;  3.4658;  4.5768]]]).reverseDiff()
        let revy = dsharp.tensor([[[ 0.6355; -5.8100];
                                 [ 0.6244;  6.0336];
                                 [ 4.8205;  1.1716]];

                                [[-8.2315; -3.0400];
                                 [-2.2282; -2.9084];
                                 [-0.9613;  1.0958]]]).reverseDiff()
        let revz = revx.conv1d(revy, stride=1)
        let revzCorrect = dsharp.tensor([[[ -1.3005; -43.8321;  62.9678];
                                         [-26.6931; -22.6506; -69.1848]];

                                        [[ 55.3161;  -3.6187;   6.3480];
                                         [ 37.6982;  98.2438;  64.8643]]])
        revz.reverse(dsharp.tensor([[[ 4.5763;  2.7538;  2.0173];
                                     [-2.7543;  7.9257; -1.3670]];

                                    [[ 1.7997; -1.2354;  4.6313];
                                     [-4.0646;  0.0384;  4.1437]]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[ 25.5806; -81.7051; -27.5597;  -7.5648];
                                         [  8.9949;  19.6812;  -2.1304;  16.1472];
                                         [ 24.7076;   7.9984;  22.9497;   0.8655]];

                                        [[ 34.6019;   0.7992; -24.1050; -39.5052];
                                         [ 10.1808;  21.8231; -13.9067;  15.8920];
                                         [ 12.5828;  -8.3376;  16.9365;   9.9666]]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[ -1.8835;  15.7019];
                                         [-15.3840;  17.9761];
                                         [ 26.7091;  -1.1857]];

                                        [[-35.3382;  93.0419];
                                         [ -5.6351;  11.3910];
                                         [-44.3729;  70.9775]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeConv1Dp1 () =
        let fwdx = dsharp.tensor([[[ 2.0028; -8.1570;  8.1037; -6.6905];
                             [ 3.6960; -3.8631; -7.0608; -1.4756];
                             [ 0.8208; -1.9973;  1.9964; -0.8280]];

                            [[-0.9567;  0.2317; -1.7773; -1.1823];
                             [ 5.1062;  0.2814;  6.3003;  1.3638];
                             [-4.9674;  3.9325;  3.8709; -0.6739]]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[-5.6993;  4.2450; 16.2727; -6.0774];
                                                 [ 2.2534; -0.2354;  6.3848;  4.8030];
                                                 [-3.0135;  4.5033; -1.8186; -8.0432]];

                                                [[ 1.0174;  4.6637;  0.7299; -2.4792];
                                                 [-4.0121;  5.3963; -0.1097;  9.4151];
                                                 [11.4479;  9.9700;  4.8665;  0.8840]]]))

        let fwdy = dsharp.tensor([[[-1.7830; -1.9625];
                             [-5.0868;  3.1041];
                             [ 7.7795;  1.4873]];

                            [[-1.3655;  1.6386];
                             [-6.1317;  3.5536];
                             [ 5.2382;  9.9893]]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[ 7.7903; -0.8083];
                                                 [-4.3881; -1.4926];
                                                 [-1.7475; -7.8380]];

                                                [[-0.5209; -3.6855];
                                                 [ 6.9068;  1.8811];
                                                 [ 0.0273; -0.1305]]]))

        let fwdz = fwdx.conv1d(fwdy, padding=1)
        let fwdzCorrect = dsharp.tensor([[[  8.7631; -14.9407; -16.1941;  44.3169;  12.9940];
                                         [ 24.6148; -68.1444;  32.4942;  18.2088;  13.8465]];

                                        [[ 10.3392; -56.6450;  57.5502;   6.7860; -10.0721];
                                         [-33.0434; -15.3614;  76.7016; -19.7505; -10.2782]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[  0.1285; -1.1446; -40.9217;  43.9575;  -120.3672];
                                         [ -31.9704; 76.8472;  -15.4720;  -175.0760;  -70.0134]];

                                        [[ 34.6626; 77.4712;  2.6527;  28.4661; -50.6120];
                                         [115.5451; 244.3823;  82.0146; 114.9505; -39.6976]]])

        let revx = dsharp.tensor([[[ 2.0028; -8.1570;  8.1037; -6.6905];
                             [ 3.6960; -3.8631; -7.0608; -1.4756];
                             [ 0.8208; -1.9973;  1.9964; -0.8280]];

                            [[-0.9567;  0.2317; -1.7773; -1.1823];
                             [ 5.1062;  0.2814;  6.3003;  1.3638];
                             [-4.9674;  3.9325;  3.8709; -0.6739]]]).reverseDiff()
        let revy = dsharp.tensor([[[-1.7830; -1.9625];
                             [-5.0868;  3.1041];
                             [ 7.7795;  1.4873]];

                            [[-1.3655;  1.6386];
                             [-6.1317;  3.5536];
                             [ 5.2382;  9.9893]]]).reverseDiff()
        let revz = revx.conv1d(revy, padding=1)
        let revzCorrect = dsharp.tensor([[[  8.7631; -14.9407; -16.1941;  44.3169;  12.9940];
                                         [ 24.6148; -68.1444;  32.4942;  18.2088;  13.8465]];

                                        [[ 10.3392; -56.6450;  57.5502;   6.7860; -10.0721];
                                         [-33.0434; -15.3614;  76.7016; -19.7505; -10.2782]]])
        revz.reverse(dsharp.tensor([[[-3.7189; -0.4834;  1.2958; -6.2053;  3.5560];
                                     [ 5.8734;  0.3692; -6.7996;  5.7922;  3.0245]];

                                    [[-1.5334;  1.5764; -5.1078;  3.8610;  3.4756];
                                     [-7.4071;  6.3234; -3.9537;  5.0018;  3.8255]]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[ 17.2804;   8.5280; -10.5300;  11.1987];
                                         [  9.5227;  34.9135; -24.0920; -35.3127];
                                         [ 51.3131; -22.5681; -83.9293;  92.1382]];

                                        [[-20.5736;  21.7742; -10.1688; -10.8016];
                                         [-77.8735;  77.5891; -80.2143; -11.3768];
                                         [-30.8856;   5.0636;   9.1459; 102.7838]]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[-99.2786;  54.8582];
                                         [ 67.4523; -46.1721];
                                         [-33.6315;  -2.9197]];

                                        [[ 62.5286; -75.4390];
                                         [ 50.1777;   5.6146];
                                         [ -7.2318;  28.6983]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeConv1Ds2p2 () =
        let fwdx = dsharp.tensor([[[  0.1264;   5.3183;   6.6905; -10.6416];
                                 [ 13.8060;   4.5253;   2.8568;  -3.2037];
                                 [ -0.5796;  -2.7937;  -3.3662;  -1.3017]];

                                [[ -2.8910;   3.9349;  -4.3892;  -2.6051];
                                 [  4.2547;   2.6049;  -9.8226;  -5.4543];
                                 [ -0.9674;   1.0070;  -4.6518;   7.1702]]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[-4.3197; -6.5898; -6.2003;  2.1058];
                                 [ 7.0684; -3.7964;  4.4218;  3.9533];
                                 [-7.1559; -7.6799; -9.5234; -3.9351]];

                                [[-0.2089; -7.8695;  6.5383;  5.1090];
                                 [-3.8272;  7.6264;  6.8205;  5.7346];
                                 [ 6.5570;  7.7248;  6.3494; -2.9007]]]))

        let fwdy = dsharp.tensor([[[ 4.0332e+00;  6.3036e+00];
                                 [ 8.4410e+00; -5.7543e+00];
                                 [-5.6937e-03; -6.7241e+00]];

                                [[-2.2619e+00;  1.2082e+00];
                                 [-1.2203e-01; -4.9373e+00];
                                 [-4.1881e+00; -3.4198e+00]]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[-1.5107; -0.0610];
                                 [-0.2609;  5.9220];
                                 [ 2.8221; -5.7314]];

                                [[ 5.0064;  3.8631];
                                 [-4.6264; -7.9380];
                                 [ 8.2204; -1.9833]]]))

        let fwdz = fwdx.conv1d(fwdy, stride=2, padding=2)
        let fwdzCorrect = dsharp.tensor([[[   0.0000;  143.3192;   11.2241;    0.0000];
                                          [   0.0000;   -5.9062;    6.0273;    0.0000]];

                                         [[   0.0000;   27.3032; -133.8372;    0.0000];
                                          [   0.0000;   -1.4792;   29.8705;    0.0000]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[   0.0000;  111.2865;   -1.8573;    0.0000];
                                          [    0.0000;   -1.9154;   29.3626;    0.0000]];

                                         [[   0.0000;  -168.6758;   25.4470;    0.0000];
                                          [   0.0000;  -149.6851;  -50.1932;    0.0000]]])

        let revx = dsharp.tensor([[[  4.4675;  -3.3205;  -1.5695;   2.6373];
                                     [ -2.0373;  -1.6156;  -5.4200;   2.1263];
                                     [ -7.6023;  -3.8521;   4.1061; -11.9378]];

                                    [[ -3.6480;  -6.2680;  10.2511;   8.2932];
                                     [  6.7741;   1.4493;   0.0978;   1.8473];
                                     [  1.7488;   5.7890;  -3.9845; -10.2116]]]).reverseDiff()
        let revy = dsharp.tensor([[[ 0.5392; -7.2312];
                                 [-6.4932;  6.0252];
                                 [ 5.4071; -1.3692]];

                                [[ 2.3730; -3.1319];
                                 [-4.3207;  2.2916];
                                 [-2.1185;  5.0338]]]).reverseDiff()
        let revz = revx.conv1d(revy, stride=2, padding=2)
        let revzCorrect = dsharp.tensor([[[  0.0000;  -5.9184;  66.6342;   0.0000];
                                         [  0.0000;  22.8156; -52.4840;   0.0000]];

                                        [[  0.0000;   9.6349; -51.5099;   0.0000];
                                         [  0.0000;  10.4620; -40.7989;   0.0000]]])
        revz.reverse(dsharp.tensor([[[ -3.2046;  -1.7019;   5.4695;  -1.0924];
                                     [  2.3244;   2.3524;   1.4251;   5.4270]];

                                    [[ -1.2459;   3.5582;   0.4258;  -9.7433];
                                     [-10.7600;  -1.3447;   2.6181;  -1.3003]]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[  4.6644;   4.9396;   6.3311; -44.0143];
                                         [  0.8869;  -4.8637; -41.6721;  36.2206];
                                         [-14.1861;  14.1717;  26.5551;  -0.3149]];

                                        [[ -1.2721; -21.5189;   6.4422; -11.2788];
                                         [-17.2943;  18.3576; -14.0771;   8.5654];
                                         [ 22.0885; -11.6407;  -3.2439;  12.5958]]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[-24.8029;   1.3047];
                                         [ -2.0322;  20.3232];
                                         [ 39.9225; -42.4877]];

                                        [[ 40.0164;  26.0883];
                                         [-21.3696;   2.1173];
                                         [-24.8156; -60.5938]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeConv1Ds2p2d3 () =
        let fwdx = dsharp.tensor([[[  4.4675;  -3.3205;  -1.5695;   2.6373];
                                 [ -2.0373;  -1.6156;  -5.4200;   2.1263];
                                 [ -7.6023;  -3.8521;   4.1061; -11.9378]];

                                [[ -3.6480;  -6.2680;  10.2511;   8.2932];
                                 [  6.7741;   1.4493;   0.0978;   1.8473];
                                 [  1.7488;   5.7890;  -3.9845; -10.2116]]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[ -2.4789;  -2.3435;  -1.7153;  -9.8687];
                                                     [  9.5786; -10.2393;   8.3291;  -8.8992];
                                                     [-10.1198;  -1.4206;   5.4935;   0.2305]];

                                                    [[ -5.6670;  -5.2314;  -0.1757;  -0.0272];
                                                     [ -2.3740;  -0.1860;   1.9684;  -1.5754];
                                                     [  1.5551;  -1.1761;   7.5176;  -1.9207]]]))

        let fwdy = dsharp.tensor([[[ 0.5392; -7.2312];
                                 [-6.4932;  6.0252];
                                 [ 5.4071; -1.3692]];

                                [[ 2.3730; -3.1319];
                                 [-4.3207;  2.2916];
                                 [-2.1185;  5.0338]]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[ -7.0064;   0.3474];
                                                 [  1.8052;   3.9392];
                                                 [ -6.8035;  -4.4947]];

                                                [[-10.6156;  -2.6311];
                                                 [  8.8583;   3.0635];
                                                 [  5.1788;  -4.6292]]]))

        let fwdz = fwdx.conv1d(fwdy, stride=2, padding=2, dilation=3)
        let fwdzCorrect = dsharp.tensor([[[  19.5508;  -15.3841;   56.5490];
                                         [ -12.6935;  -27.9701;   10.9953]];

                                        [[  46.1316;  -71.3546;  -16.6521];
                                         [  52.0922; -114.7731;   32.3440]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[  -33.0063;  -21.1314;   -52.0266];
                                         [ -1.6565;  -64.1607;  -61.7825]];

                                        [[  15.8308;  96.0475;  -16.7656];
                                         [  4.1709; 119.2216;  -153.4385]]])

        let revx = dsharp.tensor([[[  4.4675;  -3.3205;  -1.5695;   2.6373];
                                 [ -2.0373;  -1.6156;  -5.4200;   2.1263];
                                 [ -7.6023;  -3.8521;   4.1061; -11.9378]];

                                [[ -3.6480;  -6.2680;  10.2511;   8.2932];
                                 [  6.7741;   1.4493;   0.0978;   1.8473];
                                 [  1.7488;   5.7890;  -3.9845; -10.2116]]]).reverseDiff()
        let revy = dsharp.tensor([[[ 0.5392; -7.2312];
                                 [-6.4932;  6.0252];
                                 [ 5.4071; -1.3692]];

                                [[ 2.3730; -3.1319];
                                 [-4.3207;  2.2916];
                                 [-2.1185;  5.0338]]]).reverseDiff()
        let revz = revx.conv1d(revy, stride=2, padding=2, dilation=3)
        let revzCorrect = dsharp.tensor([[[  19.5508;  -15.3841;   56.5490];
                                         [ -12.6935;  -27.9701;   10.9953]];

                                        [[  46.1316;  -71.3546;  -16.6521];
                                         [  52.0922; -114.7731;   32.3440]]])
        revz.reverse(dsharp.tensor([[[  2.9290;  -8.6265;   2.4380];
                                     [-11.2377;  -8.5424;   3.5779]];

                                    [[ -2.1010;  -5.1271;   2.4158];
                                     [  7.3964;  -1.2906;   4.3965]]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[-24.9225;  14.0146;   9.8048;  89.1337];
                                         [ 92.9230;  -8.1042; -31.2896; -71.5521];
                                         [-28.5474; -60.5786;   5.6028; -31.1895]];

                                        [[ -5.8272;  -7.9718;  11.7355;  41.1170];
                                         [ 38.8675;   4.2906; -34.6827; -33.8492];
                                         [-24.9887;  40.1087;   3.7486;   0.5234]]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[  1.1031; -61.8278];
                                         [-30.1347; -35.5914];
                                         [ 57.0000; 131.8920]];

                                        [[  5.9986; -42.2784];
                                         [-10.3015;   8.3273];
                                         [ 59.8580; 201.2624]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    //TODO: add test for Conv1DTTConst
    //TODO: add test for Conv1DTConstT

    [<Test>]
    member this.TestDerivativeConv2D () =
        let fwdx = dsharp.tensor([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
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
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
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

        let fwdy = dsharp.tensor([[[[-2.1628, 15.5045],
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
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[-1.1954e+01,  2.6855e+00],
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

        let fwdz = fwdx.conv2d(fwdy)
        let fwdzCorrect = dsharp.tensor([[[[  69.4275,   57.5256,  204.7637],
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
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[-222.1388,  -38.6186,  176.5756],
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

        let revx = dsharp.tensor([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
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
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).reverseDiff()
        let revy = dsharp.tensor([[[[-2.1628, 15.5045],
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
          [ 6.0547,  5.6603]]]]).reverseDiff()
        let revz = revx.conv2d(revy)
        let revzCorrect = dsharp.tensor([[[[  69.4275,   57.5256,  204.7637],
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
        revz.reverse(dsharp.tensor([[[[  0.9103,   0.4812,   0.4156],
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
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[ 1.5744e+00,  6.2329e+00,  3.6429e+00,  6.3518e-01],
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
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[ -45.0676,  -19.8231],
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

        Assert.True(fwdz.allclose(fwdzCorrect, 0.05))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.05))
        Assert.True(revz.allclose(revzCorrect, 0.05))
        Assert.True(revxd.allclose(revxdCorrect, 0.05))
        Assert.True(revyd.allclose(revydCorrect, 0.05))

    [<Test>]
    member this.TestDerivativeConv2Dp1 () =
        let fwdx = dsharp.tensor([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
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
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
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

        let fwdy = dsharp.tensor([[[[-2.1628, 15.5045],
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
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[-1.1954e+01,  2.6855e+00],
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

        let fwdz = fwdx.conv2d(fwdy, padding=1)
        let fwdzCorrect = dsharp.tensor([[[[   2.9885,   -4.3019,   -1.1975,  -49.4543,  -24.2013],
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
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[-1.0527e+00, -9.6677e+00,  2.0051e+01, -2.8588e+01, -2.8092e+01],
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

        let revx = dsharp.tensor([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
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
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).reverseDiff()
        let revy = dsharp.tensor([[[[-2.1628, 15.5045],
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
          [ 6.0547,  5.6603]]]]).reverseDiff()
        let revz = revx.conv2d(revy, padding=1)
        let revzCorrect = dsharp.tensor([[[[   2.9885,   -4.3019,   -1.1975,  -49.4543,  -24.2013],
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
        revz.reverse(dsharp.tensor([[[[ -0.0818,  -1.7126,  -6.8625,  -1.1005,   6.4032],
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
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[  52.5073,   67.9158,  -72.1813, -110.0932],
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
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[  60.9107,   -2.4922],
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

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeConv2Ds2p2 () =
        let fwdx = dsharp.tensor([[[[ 0.4834,  0.6182, -1.7554, -0.5000],
          [ 0.4145, -1.8404, -1.3268, -0.1491],
          [-0.5398,  0.3185,  0.6443, -0.4982],
          [-0.2872, -0.3839, -0.6973,  0.0694]],

         [[ 1.0968,  0.9643, -0.7724, -1.6023],
          [-0.5834,  1.1545, -1.2427,  0.4320],
          [-1.0401,  1.0573, -0.8101, -0.3413],
          [ 0.7388,  1.5844,  1.2583, -0.5581]],

         [[ 0.0346,  0.2731, -0.4538,  2.5557],
          [-0.0170, -1.6569, -0.9249, -0.9113],
          [-1.4229,  1.1463, -1.1280, -0.1395],
          [-2.2696,  0.7478, -1.0832,  2.2226]]],


        [[[-0.8449,  0.5820, -0.4157,  1.0947],
          [ 1.3660,  1.0608,  0.6365, -0.1920],
          [ 0.6311,  1.6463, -0.2425, -1.0097],
          [-0.3608, -1.7502,  1.7420,  0.5930]],

         [[-1.6549,  2.2352,  1.1925,  0.7442],
          [-0.0709,  0.3418, -0.4357, -0.2820],
          [ 0.5562,  0.3125,  0.0196,  0.9572],
          [ 1.2896,  0.0771, -0.6986, -0.7867]],

         [[-0.0746,  0.4294, -0.2819, -0.4537],
          [-1.3862, -0.0826,  0.3576, -0.1390],
          [-0.8496, -0.3607, -1.3422, -0.2171],
          [-1.0049, -0.2201, -0.6854,  0.1075]]]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[-1.5002, -0.0945, -0.0888, -0.4832],
          [-0.4096,  0.5563,  2.7051,  0.2993],
          [ 0.0865, -0.8459,  0.0942, -0.7664],
          [-1.2281,  0.4562,  0.8160,  1.7656]],

         [[ 0.0926,  1.8626, -1.4714,  0.7599],
          [-2.1331,  0.4889,  0.2131, -0.6327],
          [ 1.6363, -1.0074, -0.1512,  0.5428],
          [-0.3064, -2.0972,  0.3547,  1.0570]],

         [[ 0.0464,  0.1123,  0.4548,  2.8014],
          [-0.1400,  1.5593, -0.6376, -0.2120],
          [ 0.1604,  0.7083, -2.7408, -0.5854],
          [ 0.1344, -1.6082,  0.8238, -0.1565]]],


        [[[-1.0191,  0.2514,  0.2823,  0.5765],
          [ 1.0055, -1.2449, -0.3048, -0.2980],
          [ 0.5053, -1.9267,  0.5766,  0.9301],
          [-1.1695,  0.3153,  0.8527, -0.3253]],

         [[-1.3888,  0.9471, -0.4777, -1.2275],
          [-1.0541, -0.9903, -1.5533,  0.1210],
          [-0.5436,  2.1711,  0.2939, -0.3888],
          [ 0.1004, -0.1512,  1.9490,  1.8398]],

         [[-0.7521,  0.5800, -0.9237, -0.8863],
          [ 0.3444,  0.2013,  1.0172,  1.8263],
          [-1.1695, -1.2351, -0.7672,  0.7108],
          [-0.3632,  0.8239, -0.8944,  0.1580]]]]))

        let fwdy = dsharp.tensor([[[[ 1.3889,  0.4028],
          [ 0.3151, -0.8122]],

         [[ 0.0136, -1.2223],
          [-0.9280,  0.6786]],

         [[-0.2875,  0.5234],
          [ 0.5449, -2.3525]]],


        [[[ 0.7302,  0.1386],
          [ 0.7884, -1.2391]],

         [[ 0.3385, -0.5488],
          [-1.2420, -0.4406]],

         [[ 0.4992, -0.1227],
          [-0.5066, -1.0926]]]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[ 5.8915e-01, -6.7759e-01],
          [-1.4154e+00,  9.7490e-01]],

         [[-1.0435e-01,  2.2350e+00],
          [ 2.4124e-03, -6.3847e-01]],

         [[ 1.2695e+00,  4.7261e-01],
          [-6.6061e-01, -1.0496e+00]]],


        [[[-1.7298e+00,  2.9588e-01],
          [-1.1972e+00,  5.2415e-01]],

         [[ 2.4734e+00,  2.0884e-01],
          [-3.8946e-01, -4.2185e-01]],

         [[ 2.0299e-01,  4.3039e-02],
          [-3.9840e-01,  1.2968e+00]]]]))

        let fwdz = fwdx.conv2d(fwdy, stride=2, padding=2)
        let fwdzCorrect = dsharp.tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  6.7284,  3.5657,  0.0000],
          [ 0.0000, -3.3040, -6.2900,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  4.9065,  0.6824,  0.0000],
          [ 0.0000, -3.1670, -4.0639,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]],


        [[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -4.1419, -0.0949,  0.0000],
          [ 0.0000,  1.3546, -2.0856,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -1.9204,  1.1343,  0.0000],
          [ 0.0000,  1.3242,  0.6037,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -5.6554, -0.3408,  0.0000],
          [ 0.0000,  3.7282, -1.9959,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -3.8840,  2.8273,  0.0000],
          [ 0.0000,  2.0030, -4.0214,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]],


        [[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  3.6465, -1.1720,  0.0000],
          [ 0.0000, -7.7013,  1.1858,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -1.1995,  3.1140,  0.0000],
          [ 0.0000, -4.0499, -2.1539,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]]])

        let revx = dsharp.tensor([[[[ 0.4834,  0.6182, -1.7554, -0.5000],
          [ 0.4145, -1.8404, -1.3268, -0.1491],
          [-0.5398,  0.3185,  0.6443, -0.4982],
          [-0.2872, -0.3839, -0.6973,  0.0694]],

         [[ 1.0968,  0.9643, -0.7724, -1.6023],
          [-0.5834,  1.1545, -1.2427,  0.4320],
          [-1.0401,  1.0573, -0.8101, -0.3413],
          [ 0.7388,  1.5844,  1.2583, -0.5581]],

         [[ 0.0346,  0.2731, -0.4538,  2.5557],
          [-0.0170, -1.6569, -0.9249, -0.9113],
          [-1.4229,  1.1463, -1.1280, -0.1395],
          [-2.2696,  0.7478, -1.0832,  2.2226]]],


        [[[-0.8449,  0.5820, -0.4157,  1.0947],
          [ 1.3660,  1.0608,  0.6365, -0.1920],
          [ 0.6311,  1.6463, -0.2425, -1.0097],
          [-0.3608, -1.7502,  1.7420,  0.5930]],

         [[-1.6549,  2.2352,  1.1925,  0.7442],
          [-0.0709,  0.3418, -0.4357, -0.2820],
          [ 0.5562,  0.3125,  0.0196,  0.9572],
          [ 1.2896,  0.0771, -0.6986, -0.7867]],

         [[-0.0746,  0.4294, -0.2819, -0.4537],
          [-1.3862, -0.0826,  0.3576, -0.1390],
          [-0.8496, -0.3607, -1.3422, -0.2171],
          [-1.0049, -0.2201, -0.6854,  0.1075]]]]).reverseDiff()
        let revy = dsharp.tensor([[[[ 1.3889,  0.4028],
          [ 0.3151, -0.8122]],

         [[ 0.0136, -1.2223],
          [-0.9280,  0.6786]],

         [[-0.2875,  0.5234],
          [ 0.5449, -2.3525]]],


        [[[ 0.7302,  0.1386],
          [ 0.7884, -1.2391]],

         [[ 0.3385, -0.5488],
          [-1.2420, -0.4406]],

         [[ 0.4992, -0.1227],
          [-0.5066, -1.0926]]]]).reverseDiff()
        let revz = revx.conv2d(revy, stride=2, padding=2)
        let revzCorrect = dsharp.tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  6.7284,  3.5657,  0.0000],
          [ 0.0000, -3.3040, -6.2900,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000,  4.9065,  0.6824,  0.0000],
          [ 0.0000, -3.1670, -4.0639,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]],


        [[[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -4.1419, -0.0949,  0.0000],
          [ 0.0000,  1.3546, -2.0856,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]],

         [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.0000, -1.9204,  1.1343,  0.0000],
          [ 0.0000,  1.3242,  0.6037,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000]]]])
        revz.reverse(dsharp.tensor([[[[ 1.7740, -0.1396, -0.9889,  0.6326],
          [ 1.1137, -0.0517, -0.4521,  1.1363],
          [-0.0234, -1.3553,  0.8126,  0.3698],
          [-0.0928, -0.7908, -1.2760,  1.9422]],

         [[ 0.8845, -0.2722,  0.7297, -1.0148],
          [ 0.1432,  2.0478,  1.0663,  0.3389],
          [ 0.6543,  0.3726,  1.0108,  1.2081],
          [-2.5648, -1.1497,  0.0553,  1.0618]]],


        [[[ 0.5235, -1.0919,  3.1056,  0.3805],
          [ 1.4381, -2.2029,  0.6346,  0.1696],
          [ 0.1067, -1.4993,  0.8540,  1.2161],
          [ 0.6368,  0.1843,  0.9232,  0.8174]],

         [[-0.1157,  0.6536, -1.3355, -0.2433],
          [ 0.8911,  0.2206, -0.1410,  0.2908],
          [-0.6419, -0.0444, -0.5832,  2.2924],
          [-0.2105, -2.6944, -0.5024, -0.6681]]]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[ 1.4236,  0.2631,  0.1507, -0.0343],
           [ 1.5983, -2.4954,  0.6983, -0.9540],
           [-1.6103, -0.4943,  1.8668,  0.4675],
           [-0.1332,  0.6390,  1.0529, -1.9124]],
 
          [[ 0.6924, -1.0606,  0.3547, -0.0325],
           [-2.4954, -0.9374, -0.9048, -0.7767],
           [ 0.1077,  1.4521,  0.3532, -1.5480],
           [ 0.7950, -1.0838, -2.0095,  0.1061]],
 
          [[ 1.0371, -0.2783,  0.6623, -0.3674],
           [-1.0655, -2.1158, -0.7865, -0.1015],
           [ 0.5756, -0.7550,  0.2709,  0.3013],
           [-0.9272,  2.7812, -0.0692, -3.0161]]],
 
 
         [[[-2.8985, -0.8568,  0.7784,  0.2361],
           [-0.5201,  1.5157,  0.0887, -0.3406],
           [-2.1149, -0.6102,  0.7603,  0.2632],
           [-0.5074,  1.2728, -0.1907,  0.0290]],
 
          [[ 0.0447,  2.5716, -0.0391, -0.6982],
           [ 1.7703, -1.5921, -0.4137,  0.4928],
           [-0.0354,  1.8571, -0.1858, -0.7238],
           [ 1.4466, -0.9979, -0.0682,  0.8365]],
 
          [[ 0.7435, -1.1800, -0.2528,  0.3494],
           [-1.3121,  4.9413,  0.4172, -1.3387],
           [ 0.4089, -0.7792, -0.5366,  0.5185],
           [-0.7945,  3.5758,  0.7608, -1.3719]]]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[ 2.4678, -4.5606],
           [-0.1756,  1.4112]],
 
          [[ 4.6288, -5.1385],
           [-2.0372, -4.5751]],
 
          [[ 1.3283, -3.7151],
           [ 6.8165,  1.8061]]],
 
 
         [[[-0.4462,  0.8376],
           [-2.1659, -4.0076]],
 
          [[-0.3534,  0.1312],
           [-0.5767,  3.4215]],
 
          [[-1.2396,  3.8720],
           [-2.8735, -1.8911]]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeConv2Ds2p2d3 () =
        let fwdx = dsharp.tensor([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
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
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[  1.2671,  -6.4862,   3.6131,   3.9654],
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

        let fwdy = dsharp.tensor([[[[-2.1628, 15.5045],
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
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[-1.1954e+01,  2.6855e+00],
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

        let fwdz = fwdx.conv2d(fwdy, stride=2, padding=2, dilation=3)
        let fwdzCorrect = dsharp.tensor([[[[-14.6076,  16.4301,  21.7161],
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
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[-192.4834,  -40.2855,    1.3868],
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

        let revx = dsharp.tensor([[[[ -0.6265,   1.5129,  -0.4967,   8.2343],
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
          [  3.1586,   1.6269,   4.5772,  -4.8104]]]]).reverseDiff()
        let revy = dsharp.tensor([[[[-2.1628, 15.5045],
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
          [ 6.0547,  5.6603]]]]).reverseDiff()
        let revz = revx.conv2d(revy, stride=2, padding=2, dilation=3)
        let revzCorrect = dsharp.tensor([[[[-14.6076,  16.4301,  21.7161],
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
        revz.reverse(dsharp.tensor([[[[-4.9500, -0.1235, -1.4549],
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
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[ 12.0238,  33.8209, -17.2926, -69.2517],
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
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[ 1.4095e+01, -5.9958e+01],
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

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

////////////
/// 

    [<Test>]
    member this.TestDerivativeConv3D () =
        let fwdx = dsharp.tensor([[[[ 2.0403e+00,  5.0188e-01,  4.6880e-01,  8.0736e-01],
                                     [-6.1190e-01,  6.1642e-01, -4.0588e-01, -2.9679e-01],
                                     [-5.6210e-01,  3.6843e-01, -6.6630e-02, -1.3918e+00],
                                     [-1.2988e+00,  9.6719e-01, -3.3539e-01,  8.7715e-01]],

                                    [[-1.7863e+00, -1.1244e+00, -2.1417e-02,  6.4124e-01],
                                     [ 7.5028e-01,  2.2587e-01, -1.2390e-01, -8.4495e-02],
                                     [-1.1291e+00,  1.5644e+00, -2.0280e+00, -9.2168e-01],
                                     [-9.2567e-01,  3.9768e-01,  1.0377e+00,  5.0193e-01]],

                                    [[-5.3238e-01, -8.4971e-02,  5.3398e-01, -1.0695e+00],
                                     [ 5.6227e-01,  2.3256e-01,  6.6780e-01, -7.1462e-01],
                                     [-6.6682e-01, -3.5299e-01, -6.0286e-01, -1.0693e+00],
                                     [ 1.2855e+00, -5.9239e-02, -1.6507e-01, -7.1905e-01]],

                                    [[-4.1638e-01,  7.6894e-01, -8.3663e-01,  8.2333e-01],
                                     [-1.4869e+00, -1.5159e+00,  8.6893e-01, -4.0507e-01],
                                     [ 1.6423e+00,  1.1892e+00,  9.8311e-01, -4.7513e-01],
                                     [ 1.4261e+00, -1.6494e+00,  8.3231e-02,  3.5143e-01]]],


                                   [[[ 1.6732e+00, -2.3141e+00, -2.7201e-01,  4.8099e-02],
                                     [ 1.4185e-01, -2.7953e-01,  2.0087e-01,  2.5665e+00],
                                     [ 2.0306e+00,  1.3222e+00,  2.3076e-01,  4.5952e-01],
                                     [ 8.8091e-01, -7.6203e-01,  1.4536e-03,  1.3817e-01]],

                                    [[-1.8129e-01,  3.7236e-01,  4.3555e-01,  1.0214e+00],
                                     [ 1.7297e-01, -3.5313e-01,  2.8694e+00, -4.7409e-01],
                                     [-6.3609e-01,  3.4134e+00, -4.9251e-01, -3.8600e-01],
                                     [ 6.8581e-02,  1.0088e+00,  3.0463e-01, -5.7993e-01]],

                                    [[ 7.7506e-01,  1.5062e-01, -2.9680e-02, -1.9979e+00],
                                     [ 6.7832e-01,  1.3433e+00,  1.0491e+00,  9.5303e-02],
                                     [-1.4113e+00, -3.0230e-01, -3.2206e-01,  3.3161e-01],
                                     [-1.0122e+00,  5.1443e-01,  6.5048e-02, -4.2270e-02]],

                                    [[ 1.2150e+00, -1.4316e+00, -2.9044e-01, -7.3760e-01],
                                     [ 3.5693e-01,  1.0187e+00,  1.1133e+00, -4.1039e-01],
                                     [-1.7768e+00, -2.2549e-01,  2.7584e-01, -1.2234e+00],
                                     [-2.9351e-01, -5.3639e-01, -1.2375e+00,  8.3979e-03]]]]).unsqueeze(0)
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[-0.2972, -1.2066, -0.6269,  0.9434],
                                                       [-0.9555,  0.3641,  0.1879,  1.2518],
                                                       [ 0.4894, -1.0205,  2.2927, -0.7572],
                                                       [ 1.1067,  0.5710, -0.0511, -1.1728]],

                                                      [[ 0.0590,  1.5141,  1.4791,  1.3218],
                                                       [-0.0762, -0.2851, -0.6395,  3.1052],
                                                       [ 1.6596, -1.5428, -0.1131, -1.1990],
                                                       [-2.3374, -0.5068, -0.7178,  0.6640]],

                                                      [[ 1.0926,  0.3023, -1.6971,  0.9061],
                                                       [ 0.8218, -1.2368,  0.1982, -0.9901],
                                                       [ 0.3584, -0.9222, -0.8179, -0.4034],
                                                       [ 0.8289, -0.6201, -0.1906,  1.0632]],

                                                      [[ 0.6155,  1.5985, -0.0755,  0.6376],
                                                       [-0.7542,  0.0373, -0.0905,  0.5350],
                                                       [-1.6389,  2.0249, -0.0807, -0.7581],
                                                       [-0.4853,  0.7108,  0.9363,  1.1280]]],


                                                     [[[-0.2131, -0.8801, -0.2641, -0.0649],
                                                       [ 0.7594,  2.1863, -0.7348,  0.4229],
                                                       [ 0.1049, -0.6827,  0.3937, -0.0045],
                                                       [ 0.1832,  0.5384, -0.4187,  0.9574]],

                                                      [[-2.6329, -0.0135,  0.3173, -1.8305],
                                                       [-1.9374, -0.1485,  1.4946,  1.2563],
                                                       [ 1.9370,  0.8723, -0.3913, -0.3136],
                                                       [-1.3977,  1.0717, -0.5465,  1.3868]],

                                                      [[ 0.0996, -0.2846, -0.9976, -0.5384],
                                                       [ 0.6077,  0.5662, -0.0110, -1.1148],
                                                       [ 0.3858, -1.5971, -0.6918,  0.1975],
                                                       [ 0.8372, -0.2214,  0.6310, -0.4999]],

                                                      [[ 1.0194, -0.0401,  0.8514, -0.7894],
                                                       [-1.8992, -0.6169,  1.0967, -0.8021],
                                                       [-0.2553,  1.1071,  0.2514,  0.2893],
                                                       [ 0.5300,  0.1334,  0.0045,  1.7511]]]]).unsqueeze(0))

        let fwdy = dsharp.tensor([[[[-0.5868, -0.6268,  0.2067],
                                     [ 0.0902, -0.2625,  0.4332],
                                     [-2.3743,  0.4579,  1.1151]],

                                    [[-0.6703, -0.4771,  1.5989],
                                     [-0.8629,  0.0367, -1.7918],
                                     [-0.1023,  0.0615, -1.3259]],

                                    [[ 0.5963,  0.3167,  0.8568],
                                     [ 1.0630, -0.2076, -1.6126],
                                     [-0.6459,  1.4887, -1.4647]]],


                                   [[[-0.6016,  0.8268,  1.3840],
                                     [-0.2750, -0.2897,  0.9044],
                                     [-1.8141, -0.2568,  0.3517]],

                                    [[ 0.4624, -0.5173, -0.7067],
                                     [-0.3159,  0.7693,  0.0949],
                                     [ 0.2051,  1.2193, -1.5660]],

                                    [[-0.0875,  0.5780, -0.2825],
                                     [ 0.2239,  0.7976,  1.5523],
                                     [ 0.6226, -0.4116,  1.0639]]]]).unsqueeze(0)
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[-1.7070, -2.1382, -0.9192],
                                                       [ 0.2977, -1.4472, -0.4086],
                                                       [-1.4684, -1.4817, -1.6733]],

                                                      [[ 0.3169,  0.7824,  1.2211],
                                                       [-0.4838, -0.1465,  1.0617],
                                                       [ 1.0696,  1.3210, -1.8589]],

                                                      [[ 0.8873,  1.8525,  0.4656],
                                                       [ 0.3368, -1.9705, -2.1460],
                                                       [ 0.2832, -1.2236, -0.1256]]],


                                                     [[[-0.7593, -1.1835,  0.3072],
                                                       [-0.4075,  0.5990,  0.0690],
                                                       [ 0.1857,  1.1791,  0.5331]],

                                                      [[ 0.7346, -0.0147, -0.5703],
                                                       [-0.9621,  0.0674, -0.8416],
                                                       [ 0.1213,  1.1037, -1.2360]],

                                                      [[ 1.4888, -1.2166, -0.4875],
                                                       [-0.4658,  1.8283,  1.9934],
                                                       [-1.2205,  0.0902, -2.0896]]]]).unsqueeze(0))

        let fwdz = fwdx.conv3d(fwdy)
        let fwdzCorrect = dsharp.tensor([[[[ 3.1109,  6.7899],
                                              [ 4.3064,  4.1053]],
                                   
                                             [[ 5.0324, -8.8943],
                                              [-0.1298,  1.2862]]]]).unsqueeze(0)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[11.2710, 12.5367],
                                                [-1.7060, -5.5142]],
                                     
                                               [[16.1629,  7.7531],
                                                [ 8.8851,  7.2768]]]]).unsqueeze(0)

        let revx = dsharp.tensor([[[[ 2.0403e+00,  5.0188e-01,  4.6880e-01,  8.0736e-01],
                                     [-6.1190e-01,  6.1642e-01, -4.0588e-01, -2.9679e-01],
                                     [-5.6210e-01,  3.6843e-01, -6.6630e-02, -1.3918e+00],
                                     [-1.2988e+00,  9.6719e-01, -3.3539e-01,  8.7715e-01]],

                                    [[-1.7863e+00, -1.1244e+00, -2.1417e-02,  6.4124e-01],
                                     [ 7.5028e-01,  2.2587e-01, -1.2390e-01, -8.4495e-02],
                                     [-1.1291e+00,  1.5644e+00, -2.0280e+00, -9.2168e-01],
                                     [-9.2567e-01,  3.9768e-01,  1.0377e+00,  5.0193e-01]],

                                    [[-5.3238e-01, -8.4971e-02,  5.3398e-01, -1.0695e+00],
                                     [ 5.6227e-01,  2.3256e-01,  6.6780e-01, -7.1462e-01],
                                     [-6.6682e-01, -3.5299e-01, -6.0286e-01, -1.0693e+00],
                                     [ 1.2855e+00, -5.9239e-02, -1.6507e-01, -7.1905e-01]],

                                    [[-4.1638e-01,  7.6894e-01, -8.3663e-01,  8.2333e-01],
                                     [-1.4869e+00, -1.5159e+00,  8.6893e-01, -4.0507e-01],
                                     [ 1.6423e+00,  1.1892e+00,  9.8311e-01, -4.7513e-01],
                                     [ 1.4261e+00, -1.6494e+00,  8.3231e-02,  3.5143e-01]]],


                                   [[[ 1.6732e+00, -2.3141e+00, -2.7201e-01,  4.8099e-02],
                                     [ 1.4185e-01, -2.7953e-01,  2.0087e-01,  2.5665e+00],
                                     [ 2.0306e+00,  1.3222e+00,  2.3076e-01,  4.5952e-01],
                                     [ 8.8091e-01, -7.6203e-01,  1.4536e-03,  1.3817e-01]],

                                    [[-1.8129e-01,  3.7236e-01,  4.3555e-01,  1.0214e+00],
                                     [ 1.7297e-01, -3.5313e-01,  2.8694e+00, -4.7409e-01],
                                     [-6.3609e-01,  3.4134e+00, -4.9251e-01, -3.8600e-01],
                                     [ 6.8581e-02,  1.0088e+00,  3.0463e-01, -5.7993e-01]],

                                    [[ 7.7506e-01,  1.5062e-01, -2.9680e-02, -1.9979e+00],
                                     [ 6.7832e-01,  1.3433e+00,  1.0491e+00,  9.5303e-02],
                                     [-1.4113e+00, -3.0230e-01, -3.2206e-01,  3.3161e-01],
                                     [-1.0122e+00,  5.1443e-01,  6.5048e-02, -4.2270e-02]],

                                    [[ 1.2150e+00, -1.4316e+00, -2.9044e-01, -7.3760e-01],
                                     [ 3.5693e-01,  1.0187e+00,  1.1133e+00, -4.1039e-01],
                                     [-1.7768e+00, -2.2549e-01,  2.7584e-01, -1.2234e+00],
                                     [-2.9351e-01, -5.3639e-01, -1.2375e+00,  8.3979e-03]]]]).unsqueeze(0).reverseDiff()
        let revy = dsharp.tensor([[[[-0.5868, -0.6268,  0.2067],
                                     [ 0.0902, -0.2625,  0.4332],
                                     [-2.3743,  0.4579,  1.1151]],

                                    [[-0.6703, -0.4771,  1.5989],
                                     [-0.8629,  0.0367, -1.7918],
                                     [-0.1023,  0.0615, -1.3259]],

                                    [[ 0.5963,  0.3167,  0.8568],
                                     [ 1.0630, -0.2076, -1.6126],
                                     [-0.6459,  1.4887, -1.4647]]],


                                   [[[-0.6016,  0.8268,  1.3840],
                                     [-0.2750, -0.2897,  0.9044],
                                     [-1.8141, -0.2568,  0.3517]],

                                    [[ 0.4624, -0.5173, -0.7067],
                                     [-0.3159,  0.7693,  0.0949],
                                     [ 0.2051,  1.2193, -1.5660]],

                                    [[-0.0875,  0.5780, -0.2825],
                                     [ 0.2239,  0.7976,  1.5523],
                                     [ 0.6226, -0.4116,  1.0639]]]]).unsqueeze(0).reverseDiff()
        let revz = revx.conv3d(revy)
        let revzCorrect = dsharp.tensor([[[[ 3.1109,  6.7899],
                                              [ 4.3064,  4.1053]],
                                   
                                             [[ 5.0324, -8.8943],
                                              [-0.1298,  1.2862]]]]).unsqueeze(0)
        revz.reverse(dsharp.tensor([[[[-0.1342,  1.0524],
                                        [ 2.0821,  0.8130]],
                             
                                       [[ 0.4078, -0.4072],
                                        [ 0.6948, -2.6370]]]]).unsqueeze(0))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[ 0.0787, -0.5334, -0.6874,  0.2176],
                                              [-1.2338, -1.6520, -0.4136,  0.6239],
                                              [ 0.5063, -3.0335,  1.0207,  1.5257],
                                              [-4.9435, -0.9768,  2.6941,  0.9066]],
                                   
                                             [[-0.1494, -0.6581, -0.3771,  1.5985],
                                              [-1.6507, -1.4833,  5.3006, -1.3073],
                                              [-2.6887, -0.0076, -2.1965, -4.4485],
                                              [-1.8626,  6.6241, -3.1435, -4.0186]],
                                   
                                             [[-0.3534,  0.6635,  1.0647,  0.2506],
                                              [ 0.2814,  4.0933,  3.6627, -4.4874],
                                              [ 1.6588,  1.9203, -3.6706,  2.4124],
                                              [-1.4160,  2.8871, -2.9229,  2.3058]],
                                   
                                             [[ 0.2432, -0.1137,  0.2205, -0.3489],
                                              [ 0.8479, -1.8701, -0.8130, -1.6027],
                                              [ 0.4751, -2.0773, -1.7766,  4.8490],
                                              [-0.4488,  2.7376, -4.9433,  3.8625]]],
                                   
                                   
                                            [[[ 0.0807, -0.7440,  0.6844,  1.4565],
                                              [-1.2156,  0.9820,  3.1275,  2.0769],
                                              [-0.3290, -2.7013,  1.3300,  1.1053],
                                              [-3.7771, -2.0095,  0.5235,  0.2859]],
                                   
                                             [[-0.3074,  1.1382, -0.2218, -1.3073],
                                              [ 0.4751,  1.0177, -1.8271, -4.4925],
                                              [-1.6161,  2.5550,  3.9565, -4.0990],
                                              [-0.8332,  7.3108, -1.3479, -2.2006]],
                                   
                                             [[ 0.2003, -0.5690,  0.5687, -0.0095],
                                              [-0.0199,  0.1245,  1.1116,  3.2289],
                                              [ 0.2468,  4.3344,  0.2067,  2.7692],
                                              [ 1.4388, -0.0447, -2.4227,  4.9946]],
                                   
                                             [[-0.0357,  0.2714, -0.3506,  0.1150],
                                              [ 0.0305,  0.8666, -1.4122,  0.1128],
                                              [ 0.4095, -0.4576, -0.4234, -4.5267],
                                              [ 0.4325, -1.9277,  1.8246, -2.8056]]]]).unsqueeze(0)
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[-0.8636,  1.4133, -0.4328],
                                              [-4.8358,  6.7805, -0.5227],
                                              [-4.2442,  0.6252, -2.4955]],
                                   
                                             [[ 0.3971, -1.3536,  3.3528],
                                              [-0.3399,  2.6152, -2.0798],
                                              [ 1.1150, -0.1769,  3.8419]],
                                   
                                             [[ 3.8232, -1.0897,  0.6076],
                                              [-3.4902, -3.2919, -0.5109],
                                              [ 7.8721, -2.1253, -2.2471]]],
                                   
                                   
                                            [[[-1.7659, -8.2318,  5.5972],
                                              [-4.2392,  5.5472,  5.5671],
                                              [-1.9285, -0.0298,  2.2652]],
                                   
                                             [[-2.3271,  0.2461,  7.8844],
                                              [ 0.6021, 10.5337, -2.9324],
                                              [ 2.1283,  1.5654, -0.2871]],
                                   
                                             [[ 1.1990,  0.9047,  2.2008],
                                              [-2.7707, -0.8894,  3.5974],
                                              [-1.2403,  3.5118,  0.2221]]]]).unsqueeze(0)

        Assert.True(fwdz.allclose(fwdzCorrect, 0.05))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.05))
        Assert.True(revz.allclose(revzCorrect, 0.05))
        Assert.True(revxd.allclose(revxdCorrect, 0.05))
        Assert.True(revyd.allclose(revydCorrect, 0.05))

    [<Test>]
    member this.TestDerivativeConv3Dp1 () =
        let fwdx = dsharp.tensor([[[[ 2.0403e+00,  5.0188e-01,  4.6880e-01,  8.0736e-01],
                                     [-6.1190e-01,  6.1642e-01, -4.0588e-01, -2.9679e-01],
                                     [-5.6210e-01,  3.6843e-01, -6.6630e-02, -1.3918e+00],
                                     [-1.2988e+00,  9.6719e-01, -3.3539e-01,  8.7715e-01]],

                                    [[-1.7863e+00, -1.1244e+00, -2.1417e-02,  6.4124e-01],
                                     [ 7.5028e-01,  2.2587e-01, -1.2390e-01, -8.4495e-02],
                                     [-1.1291e+00,  1.5644e+00, -2.0280e+00, -9.2168e-01],
                                     [-9.2567e-01,  3.9768e-01,  1.0377e+00,  5.0193e-01]],

                                    [[-5.3238e-01, -8.4971e-02,  5.3398e-01, -1.0695e+00],
                                     [ 5.6227e-01,  2.3256e-01,  6.6780e-01, -7.1462e-01],
                                     [-6.6682e-01, -3.5299e-01, -6.0286e-01, -1.0693e+00],
                                     [ 1.2855e+00, -5.9239e-02, -1.6507e-01, -7.1905e-01]],

                                    [[-4.1638e-01,  7.6894e-01, -8.3663e-01,  8.2333e-01],
                                     [-1.4869e+00, -1.5159e+00,  8.6893e-01, -4.0507e-01],
                                     [ 1.6423e+00,  1.1892e+00,  9.8311e-01, -4.7513e-01],
                                     [ 1.4261e+00, -1.6494e+00,  8.3231e-02,  3.5143e-01]]],


                                   [[[ 1.6732e+00, -2.3141e+00, -2.7201e-01,  4.8099e-02],
                                     [ 1.4185e-01, -2.7953e-01,  2.0087e-01,  2.5665e+00],
                                     [ 2.0306e+00,  1.3222e+00,  2.3076e-01,  4.5952e-01],
                                     [ 8.8091e-01, -7.6203e-01,  1.4536e-03,  1.3817e-01]],

                                    [[-1.8129e-01,  3.7236e-01,  4.3555e-01,  1.0214e+00],
                                     [ 1.7297e-01, -3.5313e-01,  2.8694e+00, -4.7409e-01],
                                     [-6.3609e-01,  3.4134e+00, -4.9251e-01, -3.8600e-01],
                                     [ 6.8581e-02,  1.0088e+00,  3.0463e-01, -5.7993e-01]],

                                    [[ 7.7506e-01,  1.5062e-01, -2.9680e-02, -1.9979e+00],
                                     [ 6.7832e-01,  1.3433e+00,  1.0491e+00,  9.5303e-02],
                                     [-1.4113e+00, -3.0230e-01, -3.2206e-01,  3.3161e-01],
                                     [-1.0122e+00,  5.1443e-01,  6.5048e-02, -4.2270e-02]],

                                    [[ 1.2150e+00, -1.4316e+00, -2.9044e-01, -7.3760e-01],
                                     [ 3.5693e-01,  1.0187e+00,  1.1133e+00, -4.1039e-01],
                                     [-1.7768e+00, -2.2549e-01,  2.7584e-01, -1.2234e+00],
                                     [-2.9351e-01, -5.3639e-01, -1.2375e+00,  8.3979e-03]]]]).unsqueeze(0)
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[-0.2972, -1.2066, -0.6269,  0.9434],
                                                       [-0.9555,  0.3641,  0.1879,  1.2518],
                                                       [ 0.4894, -1.0205,  2.2927, -0.7572],
                                                       [ 1.1067,  0.5710, -0.0511, -1.1728]],

                                                      [[ 0.0590,  1.5141,  1.4791,  1.3218],
                                                       [-0.0762, -0.2851, -0.6395,  3.1052],
                                                       [ 1.6596, -1.5428, -0.1131, -1.1990],
                                                       [-2.3374, -0.5068, -0.7178,  0.6640]],

                                                      [[ 1.0926,  0.3023, -1.6971,  0.9061],
                                                       [ 0.8218, -1.2368,  0.1982, -0.9901],
                                                       [ 0.3584, -0.9222, -0.8179, -0.4034],
                                                       [ 0.8289, -0.6201, -0.1906,  1.0632]],

                                                      [[ 0.6155,  1.5985, -0.0755,  0.6376],
                                                       [-0.7542,  0.0373, -0.0905,  0.5350],
                                                       [-1.6389,  2.0249, -0.0807, -0.7581],
                                                       [-0.4853,  0.7108,  0.9363,  1.1280]]],


                                                     [[[-0.2131, -0.8801, -0.2641, -0.0649],
                                                       [ 0.7594,  2.1863, -0.7348,  0.4229],
                                                       [ 0.1049, -0.6827,  0.3937, -0.0045],
                                                       [ 0.1832,  0.5384, -0.4187,  0.9574]],

                                                      [[-2.6329, -0.0135,  0.3173, -1.8305],
                                                       [-1.9374, -0.1485,  1.4946,  1.2563],
                                                       [ 1.9370,  0.8723, -0.3913, -0.3136],
                                                       [-1.3977,  1.0717, -0.5465,  1.3868]],

                                                      [[ 0.0996, -0.2846, -0.9976, -0.5384],
                                                       [ 0.6077,  0.5662, -0.0110, -1.1148],
                                                       [ 0.3858, -1.5971, -0.6918,  0.1975],
                                                       [ 0.8372, -0.2214,  0.6310, -0.4999]],

                                                      [[ 1.0194, -0.0401,  0.8514, -0.7894],
                                                       [-1.8992, -0.6169,  1.0967, -0.8021],
                                                       [-0.2553,  1.1071,  0.2514,  0.2893],
                                                       [ 0.5300,  0.1334,  0.0045,  1.7511]]]]).unsqueeze(0))

        let fwdy = dsharp.tensor([[[[-0.5868, -0.6268,  0.2067],
                                     [ 0.0902, -0.2625,  0.4332],
                                     [-2.3743,  0.4579,  1.1151]],

                                    [[-0.6703, -0.4771,  1.5989],
                                     [-0.8629,  0.0367, -1.7918],
                                     [-0.1023,  0.0615, -1.3259]],

                                    [[ 0.5963,  0.3167,  0.8568],
                                     [ 1.0630, -0.2076, -1.6126],
                                     [-0.6459,  1.4887, -1.4647]]],


                                   [[[-0.6016,  0.8268,  1.3840],
                                     [-0.2750, -0.2897,  0.9044],
                                     [-1.8141, -0.2568,  0.3517]],

                                    [[ 0.4624, -0.5173, -0.7067],
                                     [-0.3159,  0.7693,  0.0949],
                                     [ 0.2051,  1.2193, -1.5660]],

                                    [[-0.0875,  0.5780, -0.2825],
                                     [ 0.2239,  0.7976,  1.5523],
                                     [ 0.6226, -0.4116,  1.0639]]]]).unsqueeze(0)
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[-1.7070, -2.1382, -0.9192],
                                                       [ 0.2977, -1.4472, -0.4086],
                                                       [-1.4684, -1.4817, -1.6733]],

                                                      [[ 0.3169,  0.7824,  1.2211],
                                                       [-0.4838, -0.1465,  1.0617],
                                                       [ 1.0696,  1.3210, -1.8589]],

                                                      [[ 0.8873,  1.8525,  0.4656],
                                                       [ 0.3368, -1.9705, -2.1460],
                                                       [ 0.2832, -1.2236, -0.1256]]],


                                                     [[[-0.7593, -1.1835,  0.3072],
                                                       [-0.4075,  0.5990,  0.0690],
                                                       [ 0.1857,  1.1791,  0.5331]],

                                                      [[ 0.7346, -0.0147, -0.5703],
                                                       [-0.9621,  0.0674, -0.8416],
                                                       [ 0.1213,  1.1037, -1.2360]],

                                                      [[ 1.4888, -1.2166, -0.4875],
                                                       [-0.4658,  1.8283,  1.9934],
                                                       [-1.2205,  0.0902, -2.0896]]]]).unsqueeze(0))

        let fwdz = fwdx.conv3d(fwdy, padding=1)
        let fwdzCorrect = dsharp.tensor([[[[  2.9555,  -2.2637,  -7.1829,   5.6339],
                                            [ -3.3115,  11.7124,   2.7917,   2.6118],
                                            [  5.5319,   3.0030,   3.2099,  -2.7804],
                                            [ -1.4804,  -0.1157,  -6.4439,  -0.0716]],
                                 
                                           [[  2.4783,  -2.6479,   5.6216,  -1.2882],
                                            [-10.3388,   3.1109,   6.7899,  -6.1003],
                                            [ -1.3145,   4.3064,   4.1053,   5.3012],
                                            [  2.6878,  -4.5237,  -0.6728,   0.6796]],
                                 
                                           [[ -1.4721,  -4.1515,   4.6180,  -9.2384],
                                            [  9.8664,   5.0324,  -8.8943,   5.2075],
                                            [ -1.5404,  -0.1298,   1.2862,  -3.2419],
                                            [  8.5308,   2.7561,  -6.2106,   1.8973]],
                                 
                                           [[  0.9938,  -2.9158,  -5.2227,  -3.0340],
                                            [  3.2490,   2.0787,   2.2262,  -2.4861],
                                            [ -0.0842,   0.3416,  -3.8301,  -2.1084],
                                            [  4.0825,  -1.9845,  -1.1269,   2.3267]]]]).unsqueeze(0)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[  2.2253,  -1.9130,  -8.2497,   5.2579],
                                              [ -3.4888,   4.0274,   7.8024,  -2.4420],
                                              [  5.6041,   1.3396,  -0.2006,   4.3691],
                                              [  1.6976,  -1.4672,  -3.4178,  -7.1506]],
                                   
                                             [[ -7.8148, -11.8660, -10.5471,  -2.5008],
                                              [ -4.3332,  11.2710,  12.5367,  -8.6581],
                                              [  2.8854,  -1.7060,  -5.5142,  10.1307],
                                              [ -8.0015,  -1.0182,  -5.2705,  -3.2682]],
                                   
                                             [[ -1.1798,   6.1692,   4.6521,  -2.8876],
                                              [  7.1007,  16.1629,   7.7531,  -9.5081],
                                              [-21.3943,   8.8851,   7.2768,   2.3644],
                                              [ 12.1507,  -7.9583,  -3.8605,   8.3011]],
                                   
                                             [[ -1.1825, -16.1171,   6.6346,  -0.5791],
                                              [-10.4037,  -0.9133,   9.9003,   9.4461],
                                              [ -6.1232,  -6.2640,  -5.8346,   5.5872],
                                              [  1.5468,   3.2443,   3.2940,   7.3158]]]]).unsqueeze(0)

        let revx = dsharp.tensor([[[[ 2.0403e+00,  5.0188e-01,  4.6880e-01,  8.0736e-01],
                                     [-6.1190e-01,  6.1642e-01, -4.0588e-01, -2.9679e-01],
                                     [-5.6210e-01,  3.6843e-01, -6.6630e-02, -1.3918e+00],
                                     [-1.2988e+00,  9.6719e-01, -3.3539e-01,  8.7715e-01]],

                                    [[-1.7863e+00, -1.1244e+00, -2.1417e-02,  6.4124e-01],
                                     [ 7.5028e-01,  2.2587e-01, -1.2390e-01, -8.4495e-02],
                                     [-1.1291e+00,  1.5644e+00, -2.0280e+00, -9.2168e-01],
                                     [-9.2567e-01,  3.9768e-01,  1.0377e+00,  5.0193e-01]],

                                    [[-5.3238e-01, -8.4971e-02,  5.3398e-01, -1.0695e+00],
                                     [ 5.6227e-01,  2.3256e-01,  6.6780e-01, -7.1462e-01],
                                     [-6.6682e-01, -3.5299e-01, -6.0286e-01, -1.0693e+00],
                                     [ 1.2855e+00, -5.9239e-02, -1.6507e-01, -7.1905e-01]],

                                    [[-4.1638e-01,  7.6894e-01, -8.3663e-01,  8.2333e-01],
                                     [-1.4869e+00, -1.5159e+00,  8.6893e-01, -4.0507e-01],
                                     [ 1.6423e+00,  1.1892e+00,  9.8311e-01, -4.7513e-01],
                                     [ 1.4261e+00, -1.6494e+00,  8.3231e-02,  3.5143e-01]]],


                                   [[[ 1.6732e+00, -2.3141e+00, -2.7201e-01,  4.8099e-02],
                                     [ 1.4185e-01, -2.7953e-01,  2.0087e-01,  2.5665e+00],
                                     [ 2.0306e+00,  1.3222e+00,  2.3076e-01,  4.5952e-01],
                                     [ 8.8091e-01, -7.6203e-01,  1.4536e-03,  1.3817e-01]],

                                    [[-1.8129e-01,  3.7236e-01,  4.3555e-01,  1.0214e+00],
                                     [ 1.7297e-01, -3.5313e-01,  2.8694e+00, -4.7409e-01],
                                     [-6.3609e-01,  3.4134e+00, -4.9251e-01, -3.8600e-01],
                                     [ 6.8581e-02,  1.0088e+00,  3.0463e-01, -5.7993e-01]],

                                    [[ 7.7506e-01,  1.5062e-01, -2.9680e-02, -1.9979e+00],
                                     [ 6.7832e-01,  1.3433e+00,  1.0491e+00,  9.5303e-02],
                                     [-1.4113e+00, -3.0230e-01, -3.2206e-01,  3.3161e-01],
                                     [-1.0122e+00,  5.1443e-01,  6.5048e-02, -4.2270e-02]],

                                    [[ 1.2150e+00, -1.4316e+00, -2.9044e-01, -7.3760e-01],
                                     [ 3.5693e-01,  1.0187e+00,  1.1133e+00, -4.1039e-01],
                                     [-1.7768e+00, -2.2549e-01,  2.7584e-01, -1.2234e+00],
                                     [-2.9351e-01, -5.3639e-01, -1.2375e+00,  8.3979e-03]]]]).unsqueeze(0).reverseDiff()
        let revy = dsharp.tensor([[[[-0.5868, -0.6268,  0.2067],
                                     [ 0.0902, -0.2625,  0.4332],
                                     [-2.3743,  0.4579,  1.1151]],

                                    [[-0.6703, -0.4771,  1.5989],
                                     [-0.8629,  0.0367, -1.7918],
                                     [-0.1023,  0.0615, -1.3259]],

                                    [[ 0.5963,  0.3167,  0.8568],
                                     [ 1.0630, -0.2076, -1.6126],
                                     [-0.6459,  1.4887, -1.4647]]],


                                   [[[-0.6016,  0.8268,  1.3840],
                                     [-0.2750, -0.2897,  0.9044],
                                     [-1.8141, -0.2568,  0.3517]],

                                    [[ 0.4624, -0.5173, -0.7067],
                                     [-0.3159,  0.7693,  0.0949],
                                     [ 0.2051,  1.2193, -1.5660]],

                                    [[-0.0875,  0.5780, -0.2825],
                                     [ 0.2239,  0.7976,  1.5523],
                                     [ 0.6226, -0.4116,  1.0639]]]]).unsqueeze(0).reverseDiff()
        let revz = revx.conv3d(revy, padding=1)
        let revzCorrect = dsharp.tensor([[[[  2.9555,  -2.2637,  -7.1829,   5.6339],
                                            [ -3.3115,  11.7124,   2.7917,   2.6118],
                                            [  5.5319,   3.0030,   3.2099,  -2.7804],
                                            [ -1.4804,  -0.1157,  -6.4439,  -0.0716]],
                                 
                                           [[  2.4783,  -2.6479,   5.6216,  -1.2882],
                                            [-10.3388,   3.1109,   6.7899,  -6.1003],
                                            [ -1.3145,   4.3064,   4.1053,   5.3012],
                                            [  2.6878,  -4.5237,  -0.6728,   0.6796]],
                                 
                                           [[ -1.4721,  -4.1515,   4.6180,  -9.2384],
                                            [  9.8664,   5.0324,  -8.8943,   5.2075],
                                            [ -1.5404,  -0.1298,   1.2862,  -3.2419],
                                            [  8.5308,   2.7561,  -6.2106,   1.8973]],
                                 
                                           [[  0.9938,  -2.9158,  -5.2227,  -3.0340],
                                            [  3.2490,   2.0787,   2.2262,  -2.4861],
                                            [ -0.0842,   0.3416,  -3.8301,  -2.1084],
                                            [  4.0825,  -1.9845,  -1.1269,   2.3267]]]]).unsqueeze(0)
        revz.reverse(dsharp.tensor([[[[ 1.0787, -0.1694,  0.8318,  0.0683],
                                        [ 0.2137,  0.4235, -0.4706,  0.2851],
                                        [-0.0614,  0.6621,  0.0898,  0.3021],
                                        [ 0.2139,  1.1688,  0.1729,  0.1205]],
                             
                                       [[ 0.3361,  0.4322,  1.8931, -0.1921],
                                        [-1.5294, -1.1524, -0.3630,  0.4818],
                                        [ 1.5874,  0.6508, -0.2494, -0.5331],
                                        [-0.2771,  0.3350,  0.1326,  0.4606]],
                             
                                       [[ 0.1003, -0.5249, -0.0857,  2.2667],
                                        [ 0.3799,  0.4602,  1.4079, -0.1913],
                                        [-0.8060,  0.0085, -0.1624,  0.1486],
                                        [ 0.5855, -0.1068,  0.7961, -0.8168]],
                             
                                       [[-1.2982, -0.6645, -0.0568, -0.5870],
                                        [ 0.3615, -0.6582,  0.3083, -0.5198],
                                        [ 0.0950,  0.9003, -0.1936, -1.7822],
                                        [ 1.0543,  0.4372,  1.5583,  0.1493]]]]).unsqueeze(0))         
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[ 1.3856, -1.3797,  0.3651, -1.8831],
                                              [-2.6399, -6.2095,  2.1083,  1.7764],
                                              [ 0.1656, -1.7059, -2.9380,  0.2971],
                                              [-1.7876,  2.0877, -1.0662, -0.9915]],
                                   
                                             [[ 0.4759, -5.5183, -2.6502, -6.0993],
                                              [ 3.9518,  1.8707, -1.5257, -0.7668],
                                              [-0.0566, -3.3216,  1.3925,  4.0728],
                                              [-0.1417, -1.1870, -5.3870,  0.5377]],
                                   
                                             [[-0.3745, -1.2785, -3.2856, -0.0732],
                                              [ 0.5012, -2.2447,  6.3254, -4.5481],
                                              [-0.4594, -2.6480, -0.8333,  2.8201],
                                              [ 0.0881,  0.1293,  4.7713, -2.3660]],
                                   
                                             [[ 0.6104,  4.2032,  4.8441,  1.6347],
                                              [ 0.5691, -0.2685,  3.2872,  1.1856],
                                              [-0.9135,  1.1983,  2.1151,  0.5687],
                                              [-1.8652, -2.0462, -3.0898, -3.2949]]],
                                   
                                   
                                            [[[ 0.1811, -4.0713, -1.6113,  1.9802],
                                              [ 2.4596, -2.9796,  0.4670, -1.8031],
                                              [ 1.9348,  1.4304, -2.0844,  1.3830],
                                              [-1.7438,  2.2461,  0.7382,  0.3725]],
                                   
                                             [[ 1.4418,  3.5523,  3.6253,  2.7389],
                                              [-1.1937, -0.2722, -4.2476, -1.2932],
                                              [-0.6440, -0.4399,  4.3180,  0.9542],
                                              [ 2.6711, -0.1826,  1.0631,  1.3045]],
                                   
                                             [[ 1.0945, -0.3299, -0.5288,  4.1216],
                                              [ 1.2479, -1.2035,  2.2876,  3.4172],
                                              [ 2.1519,  2.5535,  3.7250, -0.7841],
                                              [-2.9743,  2.6455,  5.3739,  2.1657]],
                                   
                                             [[-1.1384, -0.6356,  0.4688,  0.7608],
                                              [-1.2947,  1.8075,  2.4030,  1.2020],
                                              [-0.4118, -1.2795,  1.0230, -2.9383],
                                              [ 1.7536,  0.8931, -0.3636, -1.2568]]]]).unsqueeze(0)
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[-3.8544, -7.7605, -2.6786],
                                              [ 2.2065, 10.6375, -1.0804],
                                              [ 3.3410, -3.3944, -2.1287]],
                                   
                                             [[ 2.4894,  8.5181,  0.3189],
                                              [-5.8240,  5.8649,  0.6173],
                                              [-0.1392, -4.3105,  1.1929]],
                                   
                                             [[ 0.6404,  3.3322,  0.3633],
                                              [-9.4428, -1.5023, -6.1540],
                                              [ 5.5108,  8.7631,  2.3608]]],
                                   
                                   
                                            [[[ 0.7244, -1.9559,  7.1328],
                                              [-3.8623,  8.9849, -1.6474],
                                              [ 7.2135, -6.4818, -2.6631]],
                                   
                                             [[ 0.1007,  3.1437, -5.9348],
                                              [-0.6806,  1.5860,  3.7250],
                                              [ 1.1818,  2.5798, -4.7944]],
                                   
                                             [[-3.0799,  2.7416,  1.4940],
                                              [ 1.2951, -0.8951, -4.8984],
                                              [ 2.4404,  7.5955,  0.4889]]]]).unsqueeze(0)

        Assert.True(fwdz.allclose(fwdzCorrect, 0.05))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.05))
        Assert.True(revz.allclose(revzCorrect, 0.05))
        Assert.True(revxd.allclose(revxdCorrect, 0.05))
        Assert.True(revyd.allclose(revydCorrect, 0.05))


    [<Test>]
    member this.TestDerivativeConv3Ds2p2 () =
        let fwdx = dsharp.tensor([[[[-1.5904, -0.8058,  0.4822, -0.3421],
           [ 2.3982, -0.7756,  1.7063, -0.9767],
           [-0.9279, -0.1570,  0.1906, -0.4191],
           [-1.4358, -0.6180,  0.2677, -0.6742]],

          [[ 0.4756, -1.3139,  2.5448, -2.8427],
           [ 2.0189, -2.0797,  0.6925, -1.2299],
           [-1.0635,  0.4433,  0.1820,  0.0827],
           [ 1.4230, -0.5275,  0.1589, -2.8224]],

          [[-0.3935, -1.7893, -0.0471,  0.7907],
           [-0.6793, -0.4471, -0.9305, -0.7797],
           [ 0.5104,  1.9798, -1.1192, -0.0919],
           [ 1.0709, -0.1363, -1.8121, -1.4039]],

          [[ 0.2945, -0.1779,  1.6510,  1.6800],
           [ 1.3777,  2.0716, -0.2836,  1.1209],
           [-1.3167,  0.1428,  0.1593,  0.5809],
           [-0.4858, -2.5794,  0.9739, -0.7222]]],


         [[[-1.6964,  1.7475,  0.6552, -1.1695],
           [ 0.5890, -1.0370,  0.6101, -0.6485],
           [ 0.7730,  1.1991,  1.5889, -0.9303],
           [-1.2792,  1.9116,  0.3753,  0.2457]],

          [[-1.3765,  0.7872,  0.5773, -0.6554],
           [ 0.2688, -0.1965, -0.5030,  0.6641],
           [-0.7249,  0.8578,  0.0953,  0.0374],
           [ 1.7075, -0.2019,  2.9123, -1.2923]],

          [[-0.6997,  0.1398,  1.5686, -2.7572],
           [-1.1310, -0.7520,  0.4269,  0.5536],
           [-0.6281,  1.5412, -1.0439,  1.2544],
           [-0.0201, -0.0932, -0.4438, -0.8729]],

          [[-1.0440,  0.7288, -0.3181,  0.4614],
           [-0.3361,  1.1943, -0.0648,  1.2685],
           [-0.5980, -0.1370,  1.2596,  0.9503],
           [ 0.3746, -0.0476,  0.6285, -0.7984]]]]).unsqueeze(0)
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[-1.3015, -0.1880,  1.3592, -1.1958],
           [-2.7080, -0.1565,  1.2310, -0.3947],
           [ 0.6691, -0.1534, -1.1670,  0.3617],
           [-0.3833,  1.3008,  0.3536, -1.3922]],

          [[-0.0125, -0.9405, -0.7559, -1.3247],
           [-0.2974, -0.3436, -1.8449, -1.0550],
           [-0.1813,  1.2735,  0.3520,  0.1704],
           [-0.4180,  0.0847,  1.8343,  0.4186]],

          [[ 1.2303, -1.0069, -0.8112,  1.0470],
           [-0.5389,  1.3146,  0.5313, -0.1912],
           [ 0.2771,  0.8542,  0.7820,  0.0474],
           [ 0.7872,  1.1479,  0.3659,  0.4458]],

          [[-0.3232, -0.9842, -0.9212,  2.6036],
           [ 0.3252,  0.9271,  1.1186, -1.6097],
           [-0.4261,  0.7227, -0.1925, -0.5133],
           [-0.8449, -1.1080,  0.9662, -0.0929]]],


         [[[-1.5141,  0.6547,  2.1970, -0.0409],
           [-2.3183,  1.4426, -0.3854, -0.7114],
           [-0.0887, -1.1452, -0.2280,  1.5571],
           [ 0.9241, -1.8740,  0.3112, -0.2049]],

          [[ 1.1093, -0.3449, -0.3255,  0.9099],
           [-0.7125, -0.4374,  1.8616,  0.6667],
           [ 0.5224,  1.4306,  0.8016, -0.0313],
           [ 0.5159, -0.4320, -0.1219,  0.7793]],

          [[ 1.0529, -0.3624, -0.6578,  1.1890],
           [ 0.0752,  0.0917, -1.4923,  2.8585],
           [ 0.5680, -1.8759,  2.0905,  2.1641],
           [ 0.2412,  1.3570,  0.2070, -1.0763]],

          [[ 1.1773,  2.0413,  0.4166,  0.5245],
           [ 0.2943, -0.1179,  2.1536,  0.4256],
           [-0.7390, -0.8965,  0.1503,  1.5269],
           [ 1.2729, -0.2600, -1.7034,  0.8674]]]]).unsqueeze(0))

        let fwdy = dsharp.tensor([[[[-0.4092,  0.1349,  0.6251],
           [ 1.5807, -0.5210,  0.3522],
           [ 0.3967,  0.5225, -0.9222]],

          [[-1.7646,  0.4058,  0.5206],
           [-0.8032,  1.0944, -0.1925],
           [ 0.7191, -0.9140,  1.0677]],

          [[-0.5459,  1.9029, -0.2896],
           [-0.0429, -1.2130,  1.0489],
           [-0.8698, -0.9374,  0.3327]]],


         [[[-0.8415, -0.2588, -1.8338],
           [-0.3497, -0.1282,  1.2967],
           [ 0.8212, -0.9989,  2.1417]],

          [[-0.2980,  2.1933, -0.7749],
           [ 1.3312,  0.1169, -1.3034],
           [ 1.3853,  0.6447,  0.6835]],

          [[ 0.0552,  2.2518, -1.4007],
           [-0.5206,  0.5875, -0.9891],
           [-0.5678, -0.2258,  1.3448]]]]).unsqueeze(0)
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[ 0.2063, -1.5832, -0.5495],
           [-1.7204,  0.2235, -0.8021],
           [-0.0909,  0.3673,  1.2532]],

          [[-1.5376,  1.2636,  0.7637],
           [ 0.2897, -1.3912, -0.7593],
           [ 0.8666,  0.3668,  1.9164]],

          [[-0.6670, -1.6203,  0.8436],
           [ 0.3539, -0.0813, -0.3490],
           [-0.0755,  1.9532, -0.8412]]],


         [[[-0.6086, -1.3540,  0.1197],
           [ 0.2755, -0.8959,  0.3564],
           [ 0.6376, -0.8714,  1.4792]],

          [[ 0.4008,  1.7383, -1.8348],
           [ 0.4232,  0.3102,  0.5869],
           [-1.2214, -0.2971, -1.3363]],

          [[-0.6806, -0.4961,  0.1877],
           [ 1.2847,  0.8804,  1.7529],
           [-2.2667,  0.1669,  0.3584]]]]).unsqueeze(0))

        let fwdz = fwdx.conv3d(fwdy, stride=2, padding=2)
        let fwdzCorrect = dsharp.tensor([[[[-2.8106,  3.7489, -0.2067],
           [ 5.5005,  5.6719, -3.5636],
           [-1.0549,  3.1799, -2.1535]],

          [[-3.6717,  4.5025,  5.5560],
           [ 6.0061, -4.2598, -6.9959],
           [-4.7782,  3.6403,  4.3581]],

          [[-1.5348,  2.5410,  3.9453],
           [-3.1639, -2.1265, -2.0348],
           [ 1.2052,  0.3915, -0.0322]]]]).unsqueeze(0)

        printfn "%A %A %A" fwdx.shape fwdy.shape fwdz.shape
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[-1.7395,  7.9394, -3.6845],
           [ 1.6466,  4.5153, -5.6526],
           [-3.7645, -5.3068,  6.9903]],

          [[-1.1439,  3.3253,  1.8431],
           [-7.6165,  1.4332,  7.0343],
           [ 1.9058, -3.1705,  7.0434]],

          [[ 2.0113,  9.3525,  2.3664],
           [-3.6221,  1.2941,  5.5802],
           [-1.9656, -2.2882,  4.4366]]]]).unsqueeze(0)

        let revx = dsharp.tensor([[[[-1.5904, -0.8058,  0.4822, -0.3421],
           [ 2.3982, -0.7756,  1.7063, -0.9767],
           [-0.9279, -0.1570,  0.1906, -0.4191],
           [-1.4358, -0.6180,  0.2677, -0.6742]],

          [[ 0.4756, -1.3139,  2.5448, -2.8427],
           [ 2.0189, -2.0797,  0.6925, -1.2299],
           [-1.0635,  0.4433,  0.1820,  0.0827],
           [ 1.4230, -0.5275,  0.1589, -2.8224]],

          [[-0.3935, -1.7893, -0.0471,  0.7907],
           [-0.6793, -0.4471, -0.9305, -0.7797],
           [ 0.5104,  1.9798, -1.1192, -0.0919],
           [ 1.0709, -0.1363, -1.8121, -1.4039]],

          [[ 0.2945, -0.1779,  1.6510,  1.6800],
           [ 1.3777,  2.0716, -0.2836,  1.1209],
           [-1.3167,  0.1428,  0.1593,  0.5809],
           [-0.4858, -2.5794,  0.9739, -0.7222]]],


         [[[-1.6964,  1.7475,  0.6552, -1.1695],
           [ 0.5890, -1.0370,  0.6101, -0.6485],
           [ 0.7730,  1.1991,  1.5889, -0.9303],
           [-1.2792,  1.9116,  0.3753,  0.2457]],

          [[-1.3765,  0.7872,  0.5773, -0.6554],
           [ 0.2688, -0.1965, -0.5030,  0.6641],
           [-0.7249,  0.8578,  0.0953,  0.0374],
           [ 1.7075, -0.2019,  2.9123, -1.2923]],

          [[-0.6997,  0.1398,  1.5686, -2.7572],
           [-1.1310, -0.7520,  0.4269,  0.5536],
           [-0.6281,  1.5412, -1.0439,  1.2544],
           [-0.0201, -0.0932, -0.4438, -0.8729]],

          [[-1.0440,  0.7288, -0.3181,  0.4614],
           [-0.3361,  1.1943, -0.0648,  1.2685],
           [-0.5980, -0.1370,  1.2596,  0.9503],
           [ 0.3746, -0.0476,  0.6285, -0.7984]]]]).unsqueeze(0).reverseDiff()
        let revy = dsharp.tensor([[[[-0.4092,  0.1349,  0.6251],
           [ 1.5807, -0.5210,  0.3522],
           [ 0.3967,  0.5225, -0.9222]],

          [[-1.7646,  0.4058,  0.5206],
           [-0.8032,  1.0944, -0.1925],
           [ 0.7191, -0.9140,  1.0677]],

          [[-0.5459,  1.9029, -0.2896],
           [-0.0429, -1.2130,  1.0489],
           [-0.8698, -0.9374,  0.3327]]],


         [[[-0.8415, -0.2588, -1.8338],
           [-0.3497, -0.1282,  1.2967],
           [ 0.8212, -0.9989,  2.1417]],

          [[-0.2980,  2.1933, -0.7749],
           [ 1.3312,  0.1169, -1.3034],
           [ 1.3853,  0.6447,  0.6835]],

          [[ 0.0552,  2.2518, -1.4007],
           [-0.5206,  0.5875, -0.9891],
           [-0.5678, -0.2258,  1.3448]]]]).unsqueeze(0).reverseDiff()
        let revz = revx.conv3d(revy, stride=2, padding=2)
        let revzCorrect = dsharp.tensor([[[[-2.8106,  3.7489, -0.2067],
           [ 5.5005,  5.6719, -3.5636],
           [-1.0549,  3.1799, -2.1535]],

          [[-3.6717,  4.5025,  5.5560],
           [ 6.0061, -4.2598, -6.9959],
           [-4.7782,  3.6403,  4.3581]],

          [[-1.5348,  2.5410,  3.9453],
           [-3.1639, -2.1265, -2.0348],
           [ 1.2052,  0.3915, -0.0322]]]]).unsqueeze(0)
        revz.reverse(dsharp.tensor([[[[-7.6066e-01,  1.5309e+00,  2.7315e+00],
           [-3.6182e-01,  6.5514e-01,  8.5998e-01],
           [ 3.0530e-01,  1.5119e+00,  4.9778e-01]],

          [[ 3.1924e-01,  9.3710e-01, -3.7149e-02],
           [-1.3388e+00, -1.5496e+00, -8.9303e-01],
           [-2.9413e-01, -1.2695e-01, -3.4554e-01]],

          [[ 1.3347e+00,  1.1509e+00, -2.8595e+00],
           [-2.2276e-03, -4.7472e-01,  6.3146e-01],
           [ 1.5889e+00,  9.0104e-01,  8.1894e-01]]]]).unsqueeze(0))         
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[-1.9630,  0.0921, -4.0079, -1.0640],
            [-3.3286,  0.0127, -1.3071, -0.5779],
            [-1.1160,  1.4361, -0.1028, -0.3721],
            [-0.0489, -1.7678,  0.9736, -0.4238]],
 
           [[ 3.0522, -1.4853,  1.7429, -0.3284],
            [ 1.5024, -1.6959,  1.0156, -0.9773],
            [-2.4729,  1.3648, -1.7530,  0.6760],
            [ 0.1586, -0.1389,  0.3020, -0.3782]],
 
           [[-0.0567, -3.2900, -1.4704, -3.0733],
            [-2.0890,  2.1271, -0.7561,  0.7543],
            [ 1.4952,  1.0846,  1.4030,  0.6200],
            [ 1.6808, -0.3155,  1.4935, -0.0075]],
 
           [[ 3.0891, -1.2445, -2.1890,  2.8698],
            [ 0.3817, -0.5195, -0.4158,  0.6911],
            [-1.1065,  0.7995, -1.0288, -0.2448],
            [-1.0296,  0.9861, -0.8312,  0.8963]]],
 
 
          [[[ 3.8632,  0.5945,  5.2074,  1.5879],
            [-1.1774,  0.5836, -2.7929,  0.6198],
            [-4.6964,  4.8374, -5.2262,  1.9082],
            [-1.4261,  0.9046, -1.7983,  0.3368]],
 
           [[ 3.0157, -2.7946,  2.0560, -1.9826],
            [-0.3178, -0.1812,  0.8310, -0.1044],
            [-2.7959, -1.2775, -2.0948, -1.3336],
            [ 0.2144, -0.0148, -0.2945, -0.0404]],
 
           [[ 5.8942, -4.7278,  3.8586,  0.6904],
            [ 2.2940, -0.8496,  1.1612, -0.6056],
            [-4.5823,  0.3051, -4.2579, -1.4191],
            [ 2.1023, -0.1901,  1.1875, -0.3080]],
 
           [[ 2.6497, -0.2992, -2.9950, -0.4586],
            [-0.6290, -0.0555,  1.4593,  0.0738],
            [-2.1590,  1.6702, -0.3920,  2.2033],
            [-0.8716,  0.1054, -0.0843,  0.0958]]]]).unsqueeze(0)
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[ 1.7862,  4.7762,  1.4564],
            [-5.9343,  0.8328, -4.9542],
            [-1.5079, -5.4431,  0.8418]],
 
           [[-3.0906,  6.2394, -7.0235],
            [-4.4557,  2.1720, -3.9774],
            [-1.8200, -6.5961,  5.8988]],
 
           [[-0.9614,  0.5791,  1.4879],
            [ 3.3747, -0.7264,  2.4830],
            [-1.7200, -7.3232,  3.2898]]],
 
 
          [[[ 1.2983, -0.8858, -1.8550],
            [-0.9998,  1.7663, -2.0372],
            [-9.8826,  8.7592, -2.0562]],
 
           [[ 2.4640, -0.1563,  1.4874],
            [-0.2195, -0.2790,  0.7407],
            [ 0.5143, -0.4162, -1.4317]],
 
           [[ 1.5357,  3.1054,  2.5047],
            [ 0.6906,  2.7597,  1.2784],
            [ 2.2572, -3.8087,  6.7598]]]]).unsqueeze(0)

        Assert.True(fwdz.allclose(fwdzCorrect, 0.05, 0.05))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.05, 0.05))
        Assert.True(revz.allclose(revzCorrect, 0.05, 0.05))
        Assert.True(revxd.allclose(revxdCorrect, 0.05, 0.05))
        Assert.True(revyd.allclose(revydCorrect, 0.05, 0.05))

    [<Test>]
    member this.TestDerivativeConv3Ds2p2d3 () =
        let fwdx = dsharp.tensor([[[[ 1.9019e-01, -6.0187e-01,  1.6493e+00,  1.7904e+00,  6.4850e-01],
           [-7.6622e-01,  9.0741e-01, -2.0115e-01, -3.1550e-01, -2.8159e-01],
           [-2.9039e-01, -2.2963e-01, -2.2106e-03,  1.0760e+00,  4.7973e-01],
           [-8.2379e-01,  1.3691e+00,  8.3987e-01, -2.3692e-01, -3.7774e-03],
           [-3.5065e-01, -2.4607e-01,  1.7664e+00, -1.8098e+00, -8.6175e-01]],

          [[ 1.3153e+00,  7.5605e-01, -1.0409e-01,  5.6346e-01,  6.5540e-01],
           [ 1.1977e+00,  9.6769e-01, -1.0510e+00,  3.6388e-01,  1.1678e-01],
           [-4.0814e-01, -1.8352e+00, -4.7508e-01, -1.0568e+00, -1.6195e+00],
           [-6.1626e-01,  1.4576e-01,  3.4039e-01, -7.1435e-01,  5.5371e-01],
           [ 1.1659e-01,  4.4883e-01,  2.2412e-01, -1.3702e-01,  1.7223e+00]],

          [[ 2.4527e-01, -1.9598e-01, -8.7821e-01,  2.9173e-01,  1.2561e+00],
           [-2.0942e+00,  2.5240e+00,  6.7556e-01, -8.5924e-01,  8.7535e-01],
           [ 2.3768e+00,  9.9838e-01, -6.1532e-01,  2.6833e-01, -1.2786e+00],
           [ 1.8762e-01, -6.8857e-01,  6.2736e-01, -4.3622e-01,  3.7992e-01],
           [-7.4648e-01,  2.1041e-01,  9.0266e-01, -1.1191e+00,  9.3862e-02]],

          [[ 3.0181e-01,  4.0373e-02, -8.7220e-01, -5.2011e-01,  4.6089e-01],
           [ 2.8674e-01, -3.0049e-01, -1.7049e+00, -1.4587e+00,  2.8076e-01],
           [-2.0561e-02,  1.4840e-01, -1.1705e-01, -1.5750e+00, -4.9979e-01],
           [-3.4979e-02, -2.2382e+00, -8.0744e-01,  4.8958e-01, -6.9414e-03],
           [ 3.3803e-01,  5.2228e-01,  4.6876e-02, -1.1296e-01, -6.3076e-01]],

          [[-3.5729e-01,  1.2579e+00, -5.5797e-01, -1.2824e+00,  2.8992e-01],
           [ 8.2230e-01, -4.5214e-01,  3.2368e-01,  9.4724e-02,  2.4407e+00],
           [-1.2704e+00,  2.7714e-01,  7.2789e-02,  9.2973e-01, -3.1997e-01],
           [-1.3208e-01, -1.4231e+00, -2.7735e-01,  2.7211e-01, -4.2336e-01],
           [ 7.5354e-01, -2.2061e-01, -1.4393e-01,  6.0575e-01, -1.1262e+00]]],


         [[[ 5.3742e-02,  8.2642e-01, -6.2055e-01,  1.2531e+00, -1.6536e-01],
           [-3.9541e-01, -2.0237e-01, -1.1699e+00,  3.5179e-02, -1.0149e+00],
           [ 3.6397e-01, -9.1534e-02, -7.6868e-01, -1.1445e+00, -6.4953e-01],
           [-4.5394e-01, -3.9051e-01,  7.5944e-01,  9.2531e-01,  1.2107e+00],
           [-4.4119e-01, -1.4383e+00,  3.0592e-01,  1.8999e-01, -7.1820e-01]],

          [[-1.1630e+00, -1.5975e-01, -5.4233e-01,  3.9724e-02, -2.4968e+00],
           [-4.3015e-01,  9.0982e-01,  8.6209e-01,  9.1329e-01,  4.3185e-01],
           [ 1.2126e+00,  7.7428e-01, -7.8957e-01, -2.0635e-01, -8.9979e-01],
           [-5.9508e-01, -8.7973e-01, -2.6213e+00,  1.6183e+00,  6.9877e-01],
           [-7.5514e-01, -2.3081e-01, -1.2845e-01,  1.5114e+00,  9.5189e-01]],

          [[-7.4949e-01,  3.9437e-01, -1.8863e-01, -2.3308e-01,  4.4665e-02],
           [ 7.5541e-01,  1.1475e-01,  8.4598e-01, -1.3091e+00, -5.0016e-01],
           [ 1.6062e+00, -1.5293e+00, -9.4639e-01, -1.1926e-01,  2.0189e-01],
           [-1.6043e-01, -2.3845e-02,  7.0491e-02,  1.1347e+00, -6.8573e-01],
           [ 3.2799e-01, -1.2632e+00,  7.7024e-01,  1.0526e+00, -1.9441e-01]],

          [[ 6.3536e-01,  6.8994e-02,  1.1031e+00,  2.5976e+00, -5.9173e-02],
           [ 2.8109e-01, -9.9978e-01, -1.0297e+00, -6.6330e-01, -2.3533e+00],
           [-3.6825e-01,  1.2059e+00, -6.7255e-01,  9.0993e-03, -1.1551e+00],
           [-8.4553e-01,  2.2455e-01, -1.6792e+00, -1.0454e+00, -6.4601e-01],
           [ 2.3051e+00,  5.4523e-01,  7.4099e-01, -2.1058e-01,  1.2212e+00]],

          [[-2.6009e-01, -4.9484e-01, -2.9092e+00, -5.8243e-02,  4.9593e-01],
           [ 9.6673e-01,  2.1564e-01,  1.3181e+00,  1.6915e+00,  2.0366e+00],
           [ 6.5330e-01, -1.7003e+00,  1.7339e-03,  3.0407e-01,  5.9578e-01],
           [-2.0100e+00,  9.2629e-01,  1.1525e-01,  3.4881e-01,  2.9752e-01],
           [-1.2822e+00,  4.7037e-01, -2.6658e+00,  6.3094e-01, -2.9173e-01]]]]).unsqueeze(0)
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[-1.4453e-01, -2.5193e-01, -1.6063e+00,  8.3997e-01, -4.8692e-01],
           [ 1.5610e+00,  8.5250e-01,  8.3792e-02,  6.0423e-01,  4.1669e-01],
           [ 1.5127e-01,  1.7195e+00, -5.6499e-01,  4.7642e-01, -9.1223e-01],
           [-2.8026e+00, -1.4448e+00,  2.9707e-01, -7.8099e-01, -5.5547e-01],
           [-2.0645e-01, -2.1528e-02, -6.6557e-01,  2.0319e+00, -1.7598e-01]],

          [[-4.5192e-01,  3.3469e+00,  1.1584e+00,  1.8616e+00,  7.0418e-01],
           [ 2.6886e-01, -1.8859e+00,  4.2206e-01,  1.1869e+00, -1.1695e+00],
           [-6.5267e-01, -1.2055e+00, -6.6474e-01, -1.6786e+00,  5.4507e-01],
           [ 1.6234e+00, -1.4175e+00,  1.3881e+00, -4.4556e-01,  6.8025e-01],
           [-1.0575e+00,  6.7716e-01, -6.2238e-01,  3.0351e-01,  9.3022e-01]],

          [[-1.5888e+00,  1.1878e-01, -1.0490e+00, -5.5190e-01,  3.7196e-01],
           [-1.0593e+00, -2.6790e+00,  4.0773e-01,  2.7493e-01, -2.3959e+00],
           [ 5.0254e-01,  5.9713e-01,  1.8516e-01,  1.1820e+00,  9.0747e-01],
           [ 2.5278e-01,  1.1615e+00,  5.9526e-01,  5.7415e-01,  4.4239e-01],
           [ 4.9694e-01,  8.8886e-01,  1.0924e+00, -1.1091e-01, -6.6711e-01]],

          [[ 5.1861e-01, -2.1958e+00, -1.2359e+00, -6.5801e-01,  6.6661e-01],
           [ 7.7109e-01,  7.9878e-01, -7.1244e-01,  4.6042e-01,  5.0998e-01],
           [ 1.0394e+00, -2.1884e+00,  1.7715e+00,  1.1771e+00, -1.1386e+00],
           [ 9.1426e-01,  6.6502e-01, -1.5491e-01, -1.2896e-01,  3.8778e-02],
           [ 1.4875e+00,  1.2684e+00,  1.3639e+00, -1.4334e-01,  1.4346e-01]],

          [[ 9.1115e-02,  1.8266e-01,  2.6634e-01, -8.5348e-01,  2.7726e+00],
           [-5.3104e-01,  1.3018e+00, -2.4243e-01,  8.1912e-01,  1.8386e+00],
           [ 5.4448e-01,  3.9288e-01, -1.0538e-01,  2.1827e-01, -2.7060e-01],
           [-2.5510e+00, -5.7999e-01,  7.9808e-01,  1.1592e+00, -8.0411e-01],
           [ 3.5636e-01,  3.9579e-01,  1.2808e+00, -2.5381e-01,  3.5814e-01]]],


         [[[-8.8678e-01, -4.3899e-01,  1.9736e-01, -5.6528e-02, -1.3694e-01],
           [-1.2242e+00, -8.0438e-01,  1.2799e+00, -9.2258e-01,  1.4839e+00],
           [-7.0899e-01,  6.8547e-01, -6.7400e-01,  4.6569e-01,  1.0021e+00],
           [-1.2812e+00, -4.9874e-01, -1.4080e-01,  8.8901e-01, -1.6594e-01],
           [-2.1021e-01,  2.3140e+00,  1.1562e+00,  1.2981e+00,  2.6842e-01]],

          [[ 8.2059e-01,  4.9553e-01,  1.3334e+00, -2.8390e-03,  1.8374e+00],
           [ 5.0186e-01,  1.8674e+00,  1.5143e+00,  2.4900e-01,  3.3684e-01],
           [ 2.3027e+00,  5.1881e-01, -9.0161e-01,  6.7463e-01,  5.9619e-01],
           [ 9.3840e-02,  1.4859e+00,  4.7002e-01,  6.1050e-01, -2.6531e+00],
           [ 5.0611e-01, -2.5369e-01, -8.6410e-01,  9.5455e-01, -3.2866e-01]],

          [[-5.3852e-01,  1.6611e+00, -1.7849e+00, -1.1149e+00,  2.8358e-01],
           [ 8.4228e-01, -9.9380e-03,  2.0940e+00, -6.8380e-02,  1.7631e+00],
           [ 1.5040e+00,  3.0895e-02,  1.5179e+00, -1.9171e-01, -1.2796e+00],
           [-2.3414e+00, -1.7697e+00,  4.3267e-01, -3.9850e-01,  1.1852e+00],
           [ 1.0079e+00, -1.5429e+00,  9.4237e-02,  6.0558e-01, -1.2924e+00]],

          [[-4.8632e-01, -1.5202e-01,  4.3728e-02, -6.6836e-01,  3.9981e-01],
           [-4.9614e-01, -1.1898e+00,  5.4824e-01, -6.1036e-01,  6.6130e-01],
           [ 1.7354e-01,  1.1656e+00, -7.0485e-01, -9.7515e-01, -3.0136e-01],
           [-8.3849e-01, -7.2108e-01, -7.6947e-01, -1.1760e-01, -7.5628e-01],
           [-2.9505e-01, -5.3810e-01,  7.4298e-01,  7.6034e-01, -1.8879e-01]],

          [[ 1.2792e+00, -3.6138e-01,  2.1446e-01,  4.1714e-01, -6.2706e-01],
           [ 1.6621e-01, -3.0893e-01,  1.2822e+00, -1.1141e+00,  6.9796e-01],
           [ 1.0342e+00,  3.8890e-01,  7.7467e-01,  1.0495e+00, -6.6030e-01],
           [-3.7067e-01,  4.5176e-01, -2.2020e+00,  1.4143e+00, -3.8839e-01],
           [ 1.6477e+00, -9.5105e-01,  8.0099e-01,  6.4814e-01, -1.1899e+00]]]]).unsqueeze(0))

        let fwdy = dsharp.tensor([[[[ 0.6996, -0.3127, -0.8337],
           [-0.4648, -1.0491,  0.2662],
           [ 0.1741, -0.1338, -1.1185]],

          [[-0.5431, -1.9117, -0.3445],
           [ 0.1090, -0.0702, -3.2707],
           [-0.7626, -0.4830,  0.4755]],

          [[ 1.8418,  1.5318,  0.4656],
           [ 1.1695,  0.5450, -1.1726],
           [-0.0227,  0.6563, -1.5468]]],


         [[[-1.5093,  0.5981,  1.2025],
           [ 0.8754,  1.2323,  0.6609],
           [ 1.8901,  0.3092, -0.8540]],

          [[ 0.4275,  0.3249, -0.4688],
           [-0.6980,  0.1173, -0.1754],
           [-0.0808,  1.3685,  2.0843]],

          [[-1.7598, -1.0454, -0.4667],
           [ 0.3086, -0.5486,  0.4746],
           [ 0.7247,  1.6422, -1.2026]]]]).unsqueeze(0)
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[ 0.9516, -1.0160, -0.3549],
           [ 0.1355, -0.8438, -1.2124],
           [-1.1963,  0.0505,  0.2252]],

          [[-1.0647,  0.2369,  0.0776],
           [ 0.3038, -0.5541,  1.5491],
           [-0.1983, -0.3025,  0.3942]],

          [[-0.3266,  0.8587, -0.5190],
           [-0.0456,  0.1023, -0.1999],
           [-2.0708, -0.0640,  1.2942]]],


         [[[ 0.5228,  1.3411,  0.7442],
           [-0.6761, -1.2487,  1.9835],
           [ 0.2896,  0.3069, -0.2862]],

          [[ 0.7288,  0.7760, -0.0615],
           [ 0.1509, -0.9155,  0.2590],
           [-0.4899,  0.8724, -0.5018]],

          [[-0.2203,  0.3000, -0.6040],
           [ 0.1434,  0.0557, -0.7613],
           [ 0.6402, -0.1453,  0.8314]]]]).unsqueeze(0))

        let fwdz = fwdx.conv3d(fwdy, stride=2, padding=2, dilation=3)
        let fwdzCorrect = dsharp.tensor([[[[ 2.3117,  3.4901],
           [-0.8979, -4.6094]],

          [[ 1.3561, -1.0276],
           [-1.0418,  3.9914]]]]).unsqueeze(0)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[ -3.2099,   2.7830],
           [ -4.1220, -14.4437]],

          [[ -6.0255,  -2.7025],
           [  8.3708,   6.1605]]]]).unsqueeze(0)

        let revx = dsharp.tensor([[[[ 1.9019e-01, -6.0187e-01,  1.6493e+00,  1.7904e+00,  6.4850e-01],
           [-7.6622e-01,  9.0741e-01, -2.0115e-01, -3.1550e-01, -2.8159e-01],
           [-2.9039e-01, -2.2963e-01, -2.2106e-03,  1.0760e+00,  4.7973e-01],
           [-8.2379e-01,  1.3691e+00,  8.3987e-01, -2.3692e-01, -3.7774e-03],
           [-3.5065e-01, -2.4607e-01,  1.7664e+00, -1.8098e+00, -8.6175e-01]],

          [[ 1.3153e+00,  7.5605e-01, -1.0409e-01,  5.6346e-01,  6.5540e-01],
           [ 1.1977e+00,  9.6769e-01, -1.0510e+00,  3.6388e-01,  1.1678e-01],
           [-4.0814e-01, -1.8352e+00, -4.7508e-01, -1.0568e+00, -1.6195e+00],
           [-6.1626e-01,  1.4576e-01,  3.4039e-01, -7.1435e-01,  5.5371e-01],
           [ 1.1659e-01,  4.4883e-01,  2.2412e-01, -1.3702e-01,  1.7223e+00]],

          [[ 2.4527e-01, -1.9598e-01, -8.7821e-01,  2.9173e-01,  1.2561e+00],
           [-2.0942e+00,  2.5240e+00,  6.7556e-01, -8.5924e-01,  8.7535e-01],
           [ 2.3768e+00,  9.9838e-01, -6.1532e-01,  2.6833e-01, -1.2786e+00],
           [ 1.8762e-01, -6.8857e-01,  6.2736e-01, -4.3622e-01,  3.7992e-01],
           [-7.4648e-01,  2.1041e-01,  9.0266e-01, -1.1191e+00,  9.3862e-02]],

          [[ 3.0181e-01,  4.0373e-02, -8.7220e-01, -5.2011e-01,  4.6089e-01],
           [ 2.8674e-01, -3.0049e-01, -1.7049e+00, -1.4587e+00,  2.8076e-01],
           [-2.0561e-02,  1.4840e-01, -1.1705e-01, -1.5750e+00, -4.9979e-01],
           [-3.4979e-02, -2.2382e+00, -8.0744e-01,  4.8958e-01, -6.9414e-03],
           [ 3.3803e-01,  5.2228e-01,  4.6876e-02, -1.1296e-01, -6.3076e-01]],

          [[-3.5729e-01,  1.2579e+00, -5.5797e-01, -1.2824e+00,  2.8992e-01],
           [ 8.2230e-01, -4.5214e-01,  3.2368e-01,  9.4724e-02,  2.4407e+00],
           [-1.2704e+00,  2.7714e-01,  7.2789e-02,  9.2973e-01, -3.1997e-01],
           [-1.3208e-01, -1.4231e+00, -2.7735e-01,  2.7211e-01, -4.2336e-01],
           [ 7.5354e-01, -2.2061e-01, -1.4393e-01,  6.0575e-01, -1.1262e+00]]],


         [[[ 5.3742e-02,  8.2642e-01, -6.2055e-01,  1.2531e+00, -1.6536e-01],
           [-3.9541e-01, -2.0237e-01, -1.1699e+00,  3.5179e-02, -1.0149e+00],
           [ 3.6397e-01, -9.1534e-02, -7.6868e-01, -1.1445e+00, -6.4953e-01],
           [-4.5394e-01, -3.9051e-01,  7.5944e-01,  9.2531e-01,  1.2107e+00],
           [-4.4119e-01, -1.4383e+00,  3.0592e-01,  1.8999e-01, -7.1820e-01]],

          [[-1.1630e+00, -1.5975e-01, -5.4233e-01,  3.9724e-02, -2.4968e+00],
           [-4.3015e-01,  9.0982e-01,  8.6209e-01,  9.1329e-01,  4.3185e-01],
           [ 1.2126e+00,  7.7428e-01, -7.8957e-01, -2.0635e-01, -8.9979e-01],
           [-5.9508e-01, -8.7973e-01, -2.6213e+00,  1.6183e+00,  6.9877e-01],
           [-7.5514e-01, -2.3081e-01, -1.2845e-01,  1.5114e+00,  9.5189e-01]],

          [[-7.4949e-01,  3.9437e-01, -1.8863e-01, -2.3308e-01,  4.4665e-02],
           [ 7.5541e-01,  1.1475e-01,  8.4598e-01, -1.3091e+00, -5.0016e-01],
           [ 1.6062e+00, -1.5293e+00, -9.4639e-01, -1.1926e-01,  2.0189e-01],
           [-1.6043e-01, -2.3845e-02,  7.0491e-02,  1.1347e+00, -6.8573e-01],
           [ 3.2799e-01, -1.2632e+00,  7.7024e-01,  1.0526e+00, -1.9441e-01]],

          [[ 6.3536e-01,  6.8994e-02,  1.1031e+00,  2.5976e+00, -5.9173e-02],
           [ 2.8109e-01, -9.9978e-01, -1.0297e+00, -6.6330e-01, -2.3533e+00],
           [-3.6825e-01,  1.2059e+00, -6.7255e-01,  9.0993e-03, -1.1551e+00],
           [-8.4553e-01,  2.2455e-01, -1.6792e+00, -1.0454e+00, -6.4601e-01],
           [ 2.3051e+00,  5.4523e-01,  7.4099e-01, -2.1058e-01,  1.2212e+00]],

          [[-2.6009e-01, -4.9484e-01, -2.9092e+00, -5.8243e-02,  4.9593e-01],
           [ 9.6673e-01,  2.1564e-01,  1.3181e+00,  1.6915e+00,  2.0366e+00],
           [ 6.5330e-01, -1.7003e+00,  1.7339e-03,  3.0407e-01,  5.9578e-01],
           [-2.0100e+00,  9.2629e-01,  1.1525e-01,  3.4881e-01,  2.9752e-01],
           [-1.2822e+00,  4.7037e-01, -2.6658e+00,  6.3094e-01, -2.9173e-01]]]]).unsqueeze(0).reverseDiff()
        let revy = dsharp.tensor([[[[ 0.6996, -0.3127, -0.8337],
           [-0.4648, -1.0491,  0.2662],
           [ 0.1741, -0.1338, -1.1185]],

          [[-0.5431, -1.9117, -0.3445],
           [ 0.1090, -0.0702, -3.2707],
           [-0.7626, -0.4830,  0.4755]],

          [[ 1.8418,  1.5318,  0.4656],
           [ 1.1695,  0.5450, -1.1726],
           [-0.0227,  0.6563, -1.5468]]],


         [[[-1.5093,  0.5981,  1.2025],
           [ 0.8754,  1.2323,  0.6609],
           [ 1.8901,  0.3092, -0.8540]],

          [[ 0.4275,  0.3249, -0.4688],
           [-0.6980,  0.1173, -0.1754],
           [-0.0808,  1.3685,  2.0843]],

          [[-1.7598, -1.0454, -0.4667],
           [ 0.3086, -0.5486,  0.4746],
           [ 0.7247,  1.6422, -1.2026]]]]).unsqueeze(0).reverseDiff()
        let revz = revx.conv3d(revy, stride=2, padding=2, dilation=3)
        let revzCorrect = dsharp.tensor([[[[ 2.3117,  3.4901],
           [-0.8979, -4.6094]],

          [[ 1.3561, -1.0276],
           [-1.0418,  3.9914]]]]).unsqueeze(0)
        revz.reverse(dsharp.tensor([[[[ 3.9327e-02,  8.2953e-01],
           [ 1.9582e+00,  9.1915e-01]],

          [[ 2.4345e-01, -9.2549e-04],
           [ 9.4359e-02, -9.3872e-01]]]]).unsqueeze(0))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[-6.5670e-01, -2.9506e-02,  0.0000e+00,  2.9354e-01, -7.8669e-02],
            [ 4.3019e-04, -2.5541e-01,  0.0000e+00,  9.7095e-04,  6.4813e-02],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 4.3634e-01, -9.8994e-02,  0.0000e+00,  9.8483e-01,  2.5121e-02],
            [-1.6113e-04, -3.2578e-02,  0.0000e+00,  1.2385e-04, -2.7229e-01]],
 
           [[-4.9920e-01, -3.7433e+00,  0.0000e+00, -1.7571e+00, -6.7465e-01],
            [ 9.0398e-02, -2.7614e-03,  0.0000e+00, -5.8246e-02, -1.2863e-01],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 1.0016e-01, -1.3749e-01,  0.0000e+00, -6.4538e-02, -6.4045e+00],
            [-6.3262e-01, -1.8994e-02,  0.0000e+00, -4.0065e-01,  1.8698e-02]],
 
           [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],
 
           [[ 5.0982e-01, -1.8038e-01,  0.0000e+00,  1.7945e+00, -3.2510e-02],
            [-1.0085e-04, -1.7094e-02,  0.0000e+00,  6.4983e-05, -7.9623e-01],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [-1.0230e-01, -6.6254e-03,  0.0000e+00,  6.5912e-02, -3.0862e-01],
            [ 7.0580e-04, -1.1758e-01,  0.0000e+00,  4.4700e-04,  1.1575e-01]],
 
           [[ 1.6929e+00,  2.9994e+00,  0.0000e+00,  1.4079e+00,  9.1181e-01],
            [ 9.7014e-01,  2.1432e-02,  0.0000e+00,  4.5208e-01, -4.6115e-02],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 1.0749e+00,  1.0671e+00,  0.0000e+00,  5.0092e-01, -2.2961e+00],
            [-1.8861e-02,  2.5810e-02,  0.0000e+00,  5.4441e-01, -6.0832e-02]]],
 
 
          [[[ 1.4168e+00,  5.6437e-02,  0.0000e+00, -5.6146e-01,  1.1347e-01],
            [-8.1017e-04,  3.0001e-01,  0.0000e+00, -1.1405e-03,  1.6089e-01],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [-8.2175e-01,  1.1628e-01,  0.0000e+00, -1.1568e+00,  6.2359e-02],
            [-1.7493e-03,  7.5263e-02,  0.0000e+00, -2.8612e-04, -2.0791e-01]],
 
           [[ 3.9294e-01,  6.3630e-01,  0.0000e+00,  2.9868e-01, -9.1794e-01],
            [-5.7901e-01,  4.6120e-03,  0.0000e+00,  9.7282e-02, -6.8988e-03],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [-6.4156e-01,  2.2964e-01,  0.0000e+00,  1.0779e-01, -3.4350e-01],
            [-6.7031e-02,  5.3819e-02,  0.0000e+00,  1.1352e+00,  8.1971e-02]],
 
           [[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],
 
           [[-4.0131e-01,  3.0662e-02,  0.0000e+00, -3.0504e-01, -4.4233e-02],
            [ 6.4598e-04,  2.8550e-02,  0.0000e+00, -1.0854e-04, -4.2705e-02],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 6.5521e-01,  1.1066e-02,  0.0000e+00, -1.1009e-01, -1.6552e-02],
            [ 7.4785e-05,  3.3316e-01,  0.0000e+00, -1.2665e-03,  5.0742e-01]],
 
           [[-1.6175e+00, -2.0470e+00,  0.0000e+00, -9.6085e-01, -9.1381e-01],
            [ 2.5603e-01, -2.1575e-02,  0.0000e+00, -4.5508e-01,  1.8663e-02],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 2.8369e-01, -1.0742e+00,  0.0000e+00, -5.0424e-01,  9.2928e-01],
            [ 6.0113e-01,  6.4585e-02,  0.0000e+00,  1.3623e+00, -4.7294e-02]]]]).unsqueeze(0)
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[-1.7854e-01, -1.7375e+00,  6.1192e-02],
            [ 7.7402e-01,  5.7279e-01, -6.8908e-02],
            [ 3.2452e-04, -5.8230e-02, -2.0979e-01]],
 
           [[ 9.2565e-01,  2.4904e+00,  1.3269e+00],
            [ 4.5964e-01, -7.7384e-01,  1.1565e+00],
            [ 9.6399e-02,  3.1238e-02, -8.5824e-02]],
 
           [[-3.2841e-01,  1.2845e+00,  5.6771e-01],
            [ 5.6073e-01, -2.4757e+00, -7.3302e-01],
            [ 6.2509e-01,  4.9381e-01, -4.4290e-02]]],
 
 
          [[[-5.0448e-02, -1.0983e+00, -1.5603e-02],
            [ 4.2649e-01, -9.5475e-01, -1.3283e-01],
            [ 4.0831e-04, -3.5034e-01, -1.7484e-01]],
 
           [[-1.6654e+00, -2.7082e+00, -4.8948e+00],
            [-1.1035e-01,  1.3179e+00,  7.5142e-01],
            [-6.2855e-01,  1.3776e+00,  3.3473e-01]],
 
           [[-2.3906e-01, -1.0225e+00,  9.7111e-01],
            [-1.0455e+00,  3.5461e+00,  6.6268e-01],
            [-1.0636e+00,  5.4188e-01, -1.1473e-02]]]]).unsqueeze(0)

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeConv2Ds23p32d23 () =
        let fwdx = dsharp.tensor([[[[ 0.4011, -1.2780,  1.5871, -1.7213],
          [ 0.2381,  1.1069,  0.0987, -0.9293],
          [ 0.3164,  0.8496,  0.9187, -1.1235],
          [-1.7634, -2.0868, -0.5947,  0.6119]],

         [[-0.3527, -1.2002, -1.5498, -0.6298],
          [-0.2286,  0.9309, -0.1588, -1.9190],
          [ 1.7520, -0.0821, -1.5686, -1.5113],
          [-0.8367,  1.2131,  0.7808, -0.0747]],

         [[ 1.2769, -0.1038, -0.8314,  0.3005],
          [ 0.6932,  0.2336,  0.2408, -0.0935],
          [-1.1946,  0.7827, -0.5326, -1.3461],
          [-0.8028,  0.6509, -0.5550,  1.4831]]],


        [[[ 0.4528, -0.7187, -0.5513,  0.9885],
          [-1.9050,  0.7175, -0.6040, -0.6299],
          [-0.3445, -1.4051, -1.1323,  0.2162],
          [-0.1055,  0.2839, -1.4775,  0.1079]],

         [[ 2.0242, -1.2543,  0.6890, -0.2272],
          [-1.4539, -0.5034, -2.8591, -0.0892],
          [ 2.2453,  0.9355, -0.3067,  0.1946],
          [ 0.7941,  0.3432, -0.2243,  0.4131]],

         [[-0.9877,  1.0967,  0.2313, -1.2953],
          [ 0.2964,  0.1028,  0.9264,  0.6879],
          [ 1.1085,  0.7679, -0.4832,  1.1809],
          [-0.2029,  0.9352,  0.8924,  0.2688]]]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[[[-9.6456e-01, -7.4969e-02, -5.3567e-01, -1.5748e+00],
          [-6.4667e-01, -4.5245e-01, -5.5802e-01,  1.4635e-01],
          [ 8.7809e-01,  5.8505e-01,  5.3637e-01,  4.4096e-01],
          [-2.2662e-01,  8.8711e-02, -1.8153e-01,  9.4813e-01]],

         [[-1.6630e-01,  1.5228e-01,  8.5819e-02,  6.7602e-03],
          [-1.6795e+00,  4.8293e-01, -3.8172e-01,  6.4784e-01],
          [ 5.4453e-01, -6.3874e-01, -9.6772e-02, -4.9632e-01],
          [-3.9779e+00,  2.3978e-02,  3.7684e-01, -4.0631e-01]],

         [[ 1.1614e+00, -9.8773e-01,  7.0932e-01, -6.6573e-01],
          [ 1.3459e+00, -1.0123e+00, -4.8873e-01,  6.5714e-01],
          [ 1.5338e-01,  2.0095e+00,  9.5947e-02,  9.7789e-01],
          [ 4.4044e-01,  1.5384e-01, -6.0818e-01,  1.2463e-01]]],


        [[[ 3.1008e-01, -2.9977e+00, -1.1546e+00,  9.0698e-03],
          [-1.5160e+00,  2.1643e-01, -6.7407e-01, -1.9241e-03],
          [ 2.8470e-01, -2.1449e-01,  9.6742e-01,  3.4073e-01],
          [ 9.2719e-01,  1.4742e+00, -1.6652e+00, -5.3817e-01]],

         [[-1.0078e+00, -8.0932e-01, -1.3082e+00, -2.5629e-01],
          [-8.1680e-01, -1.1634e-01,  7.1552e-01,  9.6431e-01],
          [-2.7368e-01,  1.7727e+00, -9.1372e-01,  3.0905e-01],
          [ 8.5897e-01, -4.2025e-01,  1.6485e+00, -6.5409e-01]],

         [[ 1.2498e+00, -1.4730e+00,  3.6122e-01, -1.2689e+00],
          [-1.2666e+00, -8.3490e-01,  5.7762e-01,  6.2228e-01],
          [ 1.0982e+00, -6.6295e-02, -4.4212e-01,  6.9732e-01],
          [ 7.3720e-01,  9.7936e-01,  5.0589e-01, -6.4268e-01]]]]))

        let fwdy = dsharp.tensor([[[[ 0.4282,  1.2302],
          [ 2.2170,  1.2013]],

         [[ 0.6013,  0.0310],
          [ 1.4277,  1.4859]],

         [[ 0.7032,  1.1422],
          [-0.5346, -0.6356]]],


        [[[-0.4717, -0.6798],
          [ 1.7547, -0.0347]],

         [[-1.7426, -0.7333],
          [-0.5278,  0.5487]],

         [[-1.5099,  0.5905],
          [ 1.5490, -1.1658]]]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[[[ 0.5367, -1.9746],
          [-0.0403,  1.1286]],

         [[ 0.4467, -1.4777],
          [-0.6061, -0.8810]],

         [[-0.6518, -0.0857],
          [-0.3082,  1.0148]]],


        [[[-0.6499, -0.6090],
          [-1.1788, -0.3610]],

         [[ 0.5513,  0.4726],
          [-0.5593, -0.0034]],

         [[-2.1767, -0.2879],
          [-0.7617, -0.8730]]]]))

        let fwdz = fwdx.conv2d(fwdy, strides=[2;3], paddings=[3;2], dilations=[2;3])
        let fwdzCorrect = dsharp.tensor([[[[ 0.0000,  0.0000],
          [ 2.5643,  3.6581],
          [ 0.5393, -2.0446],
          [-1.7860,  0.2936]],

         [[ 0.0000,  0.0000],
          [ 0.2000,  1.8128],
          [-1.3179, -5.7909],
          [ 0.9134, -2.1124]]],


        [[[ 0.0000,  0.0000],
          [ 0.0485,  0.8170],
          [ 1.2410,  0.6963],
          [ 1.4282,  0.9856]],

         [[ 0.0000,  0.0000],
          [-0.4210,  1.6840],
          [-0.9697,  2.1493],
          [ 0.1075, -2.1441]]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[[[ 0.0000,  0.0000],
          [ 1.4837, -0.4532],
          [-7.9980, -0.4608],
          [ 2.5579, -0.8418]],

         [[ 0.0000,  0.0000],
          [ 0.8541, -4.6203],
          [-0.9341,  1.8527],
          [ 1.6697,  0.2923]]],


        [[[ 0.0000,  0.0000],
          [ 1.9754,  1.0046],
          [ 0.1185,  1.1658],
          [ 1.7711,  0.7633]],

         [[ 0.0000,  0.0000],
          [ 0.5549, -1.4946],
          [-3.6029,  3.4802],
          [-0.3956, -3.4729]]]])

        let revx = dsharp.tensor([[[[ 0.4011, -1.2780,  1.5871, -1.7213],
          [ 0.2381,  1.1069,  0.0987, -0.9293],
          [ 0.3164,  0.8496,  0.9187, -1.1235],
          [-1.7634, -2.0868, -0.5947,  0.6119]],

         [[-0.3527, -1.2002, -1.5498, -0.6298],
          [-0.2286,  0.9309, -0.1588, -1.9190],
          [ 1.7520, -0.0821, -1.5686, -1.5113],
          [-0.8367,  1.2131,  0.7808, -0.0747]],

         [[ 1.2769, -0.1038, -0.8314,  0.3005],
          [ 0.6932,  0.2336,  0.2408, -0.0935],
          [-1.1946,  0.7827, -0.5326, -1.3461],
          [-0.8028,  0.6509, -0.5550,  1.4831]]],


        [[[ 0.4528, -0.7187, -0.5513,  0.9885],
          [-1.9050,  0.7175, -0.6040, -0.6299],
          [-0.3445, -1.4051, -1.1323,  0.2162],
          [-0.1055,  0.2839, -1.4775,  0.1079]],

         [[ 2.0242, -1.2543,  0.6890, -0.2272],
          [-1.4539, -0.5034, -2.8591, -0.0892],
          [ 2.2453,  0.9355, -0.3067,  0.1946],
          [ 0.7941,  0.3432, -0.2243,  0.4131]],

         [[-0.9877,  1.0967,  0.2313, -1.2953],
          [ 0.2964,  0.1028,  0.9264,  0.6879],
          [ 1.1085,  0.7679, -0.4832,  1.1809],
          [-0.2029,  0.9352,  0.8924,  0.2688]]]]).reverseDiff()
        let revy = dsharp.tensor([[[[ 0.4282,  1.2302],
          [ 2.2170,  1.2013]],

         [[ 0.6013,  0.0310],
          [ 1.4277,  1.4859]],

         [[ 0.7032,  1.1422],
          [-0.5346, -0.6356]]],


        [[[-0.4717, -0.6798],
          [ 1.7547, -0.0347]],

         [[-1.7426, -0.7333],
          [-0.5278,  0.5487]],

         [[-1.5099,  0.5905],
          [ 1.5490, -1.1658]]]]).reverseDiff()
        let revz = revx.conv2d(revy, strides=[2;3], paddings=[3;2], dilations=[2;3])
        let revzCorrect = dsharp.tensor([[[[ 0.0000,  0.0000],
          [ 2.5643,  3.6581],
          [ 0.5393, -2.0446],
          [-1.7860,  0.2936]],

         [[ 0.0000,  0.0000],
          [ 0.2000,  1.8128],
          [-1.3179, -5.7909],
          [ 0.9134, -2.1124]]],


        [[[ 0.0000,  0.0000],
          [ 0.0485,  0.8170],
          [ 1.2410,  0.6963],
          [ 1.4282,  0.9856]],

         [[ 0.0000,  0.0000],
          [-0.4210,  1.6840],
          [-0.9697,  2.1493],
          [ 0.1075, -2.1441]]]])
        revz.reverse(dsharp.tensor([[[[-0.5596, -0.6619],
          [-0.4751,  0.5980],
          [ 1.3078, -2.0496],
          [ 1.2081,  0.2797]],

         [[-0.2093, -1.4207],
          [ 0.1662,  2.9676],
          [-0.2906,  0.7331],
          [-0.7858,  0.5466]]],


        [[[-0.6618,  0.5721],
          [ 0.8084, -1.1384],
          [ 2.0816, -0.6468],
          [ 0.1776,  0.0817]],

         [[ 1.0989, -0.7164],
          [ 0.3258, -1.1641],
          [-0.6983, -0.2945],
          [ 0.1820,  0.3053]]]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  6.5395,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.2058,  0.0000,  0.0000]],
 
          [[ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000, -3.5835,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000, -1.6999,  0.0000,  0.0000]],
 
          [[ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  3.1594,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  2.0262,  0.0000,  0.0000]]],
 
 
         [[[ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.7093,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  0.5597,  0.0000,  0.0000]],
 
          [[ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  1.0700,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000,  1.3310,  0.0000,  0.0000]],
 
          [[ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.1330,  0.0000,  0.0000],
           [ 0.0000,  0.0000,  0.0000,  0.0000],
           [ 0.0000, -0.7126,  0.0000,  0.0000]]]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[[[-3.2932,  0.4704],
           [ 3.9386, -2.0841]],
 
          [[-1.2150,  1.6960],
           [-1.5787,  1.4518]],
 
          [[-0.2869,  1.4721],
           [-1.9164,  2.7702]]],
 
 
         [[[-0.4538,  0.8688],
           [ 0.8361,  0.8259]],
 
          [[ 1.5986, -0.8098],
           [ 4.1368, -0.6016]],
 
          [[ 0.7823, -0.4810],
           [ 0.7753, -0.7699]]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeMatMulT2T2 () =
        let fwdx = dsharp.tensor([[6.2381; 0.0393; 8.2364; 3.9906; 6.2291];
            [9.8762; 3.2263; 6.2866; 4.7111; 0.0652];
            [3.5832; 7.9801; 1.9854; 4.4965; 4.1712]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[4.6453; 8.4388; 4.6549; 9.5680; 1.5756];
            [3.2066; 4.2429; 2.2028; 9.1037; 3.4022];
            [4.2324; 4.5508; 3.4755; 2.7196; 5.5344]]))
        let fwdy = dsharp.tensor([[4.4220; 3.7293];
            [6.1928; 2.1446];
            [0.0525; 1.2494];
            [7.5281; 1.4816];
            [5.0328; 2.2756]])
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[1.4749; 9.7608];
            [3.6599; 7.9553];
            [3.5503; 1.3757];
            [8.3172; 6.6748];
            [2.2959; 0.6784]]))
        let fwdz = dsharp.matmul(fwdx, fwdy)
        let fwdzCorrect = dsharp.tensor([[ 89.6516; 53.7260];
            [ 99.7751; 58.7331];
            [120.2113; 49.1116]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[239.0819; 162.3930];
            [214.2522; 207.2430];
            [183.9220; 180.5424]])

        let revx = dsharp.tensor([[6.2381; 0.0393; 8.2364; 3.9906; 6.2291];
            [9.8762; 3.2263; 6.2866; 4.7111; 0.0652];
            [3.5832; 7.9801; 1.9854; 4.4965; 4.1712]]).reverseDiff()
        let revy = dsharp.tensor([[4.4220; 3.7293];
            [6.1928; 2.1446];
            [0.0525; 1.2494];
            [7.5281; 1.4816];
            [5.0328; 2.2756]]).reverseDiff()
        let revz = dsharp.matmul(revx, revy)
        let revzCorrect = dsharp.tensor([[ 89.6516; 53.7260];
            [ 99.7751; 58.7331];
            [120.2113; 49.1116]])
        revz.reverse(dsharp.tensor([[7.3984; 0.1849];
            [1.2520; 9.5731];
            [6.8201; 9.5221]]))            
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[33.4050; 46.2136;  0.6191; 55.9696; 37.6556];
            [41.2370; 28.2842; 12.0266; 23.6085; 28.0854];
            [65.6689; 62.6571; 12.2551; 65.4497; 55.9926]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[ 82.9549;129.8180];
            [ 58.7551;106.8801];
            [ 82.3474; 80.6097];
            [ 66.0888; 88.6534];
            [ 74.6154; 41.4950]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    //TODO: add test for MatMulT2T2Const
    //TODO: add test for MatMulT2ConstT2

    [<Test>]
    member this.TestTensorStackTs () =
        let fwdxa = dsharp.tensor([1.; 2.]).forwardDiff(dsharp.tensor([10.; 20.]))
        let fwdxb = dsharp.tensor([3.; 4.]).forwardDiff(dsharp.tensor([30.; 40.]))
        let fwdxc = dsharp.tensor([5.; 6.]).forwardDiff(dsharp.tensor([50.; 60.]))
        let fwdz = dsharp.stack([fwdxa;fwdxb;fwdxc])
        let fwdzCorrect = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]])

        let revxa = dsharp.tensor([1.; 2.]).reverseDiff()
        let revxb = dsharp.tensor([3.; 4.]).reverseDiff()
        let revxc = dsharp.tensor([5.; 6.]).reverseDiff()
        let revz = dsharp.stack([revxa;revxb;revxc])
        let revzCorrect = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]])
        revz.reverse(dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]]))
        let revxda = revxa.derivative
        let revxdaCorrect = dsharp.tensor([10.; 20.])
        let revxdb = revxb.derivative
        let revxdbCorrect = dsharp.tensor([30.; 40.])
        let revxdc = revxc.derivative
        let revxdcCorrect = dsharp.tensor([50.; 60.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdaCorrect, revxda)
        Assert.AreEqual(revxdbCorrect, revxdb)
        Assert.AreEqual(revxdcCorrect, revxdc)

    [<Test>]
    member this.TestDerivativeNeg () =
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = -fwdx
        let fwdzCorrect = dsharp.tensor([-1.; -2.; -3.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([-2.; -3.; -4.])

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revz = -revx
        let revzCorrect = dsharp.tensor([-1.; -2.; -3.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-5.; -5.; -5.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeGather () =
        let fwdx = dsharp.tensor([1,2,3,4,5]).forwardDiff(dsharp.tensor([10,20,30,40,50]))
        let fwdz = dsharp.gather(fwdx, 0, dsharp.tensor([0,2,3], dtype=DType.Int32))
        let fwdzCorrect = dsharp.tensor([1,3,4])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([10,30,40])

        let revx = dsharp.tensor([1,2,3,4,5]).reverseDiff()
        let revz = dsharp.gather(revx, 0, dsharp.tensor([0,2,3], dtype=DType.Int32))
        let revzCorrect = dsharp.tensor([1,3,4])
        revz.reverse(dsharp.tensor([10,30,40]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([10.,  0., 30., 40.,  0.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

        let fwdx = dsharp.tensor([[1,2,3],[4,5,6]]).forwardDiff(dsharp.tensor([[10,20,30],[40,50,60]]))
        let fwdz = dsharp.gather(fwdx, 0, dsharp.tensor([[1,0,1],[0,1,1]], dtype=DType.Int32))
        let fwdzCorrect = dsharp.tensor([[4,2,6],[1,5,6]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[40,20,60],[10,50,60]])

        let revx = dsharp.tensor([[1,2,3],[4,5,6]]).reverseDiff()
        let revz = dsharp.gather(revx, 0, dsharp.tensor([[1,0,1],[0,1,1]], dtype=DType.Int32))
        let revzCorrect = dsharp.tensor([[4,2,6],[1,5,6]])
        revz.reverse(dsharp.tensor([[40,20,60],[10,50,60]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10,20,0],[40,50,120]])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

        let fwdx = dsharp.tensor([[1,2,3],[4,5,6]]).forwardDiff(dsharp.tensor([[10,20,30],[40,50,60]]))
        let fwdz = dsharp.gather(fwdx, 1, dsharp.tensor([[1,0,1],[0,1,1]], dtype=DType.Int32))
        let fwdzCorrect = dsharp.tensor([[2,1,2],[4,5,5]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[20,10,20],[40,50,50]])

        let revx = dsharp.tensor([[1,2,3],[4,5,6]]).reverseDiff()
        let revz = dsharp.gather(revx, 1, dsharp.tensor([[1,0,1],[0,1,1]], dtype=DType.Int32))
        let revzCorrect = dsharp.tensor([[2,1,2],[4,5,5]])
        revz.reverse(dsharp.tensor([[20,10,20],[40,50,50]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10,40,0],[40,100,0]])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeSum () =
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.sum()
        let fwdzCorrect = dsharp.tensor(6.)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor(9.)

        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revz = revx.sum()
        let revzCorrect = dsharp.tensor(6.)
        revz.reverse(dsharp.tensor(5.))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([5.; 5.; 5.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeSumT2Dim0 () =
        let fwdx = dsharp.tensor([[1.; 2.]; [3.; 4.]]).forwardDiff(dsharp.tensor([[2.; 3.]; [4.; 5.]]))
        let fwdz = fwdx.sumT2Dim0()
        let fwdzCorrect = dsharp.tensor([4.; 6.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([6.; 8.])

        let revx = dsharp.tensor([[1.; 2.]; [3.; 4.]]).reverseDiff()
        let revz = revx.sumT2Dim0()
        let revzCorrect = dsharp.tensor([4.; 6.])
        revz.reverse(dsharp.tensor([5.; 6.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[5.; 6.]; [5.; 6.]])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeMean () =
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.mean()
        let fwdzCorrect = dsharp.tensor(2.)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor(3.)

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)

        (* Python:
        input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
        loss = input.mean()
        loss.backward(torch.tensor(3.0))
        input.grad
        --> tensor([1., 1., 1.])
        *)
        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revz = revx.mean()
        let revzCorrect = dsharp.tensor(2.)
        revz.reverse(dsharp.tensor(3.))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([1.; 1.; 1.])

        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeVariance () =
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.variance()
        let fwdzCorrect = dsharp.tensor(1.0)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor(2.0)

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)

        (* Python:
        input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
        loss = input.var()
        loss.backward(torch.tensor(3.0))
        input.grad
        --> tensor([-3.,  0.,  3.])
        *)
        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revz = revx.variance()
        let revzCorrect = dsharp.tensor(1.)
        revz.reverse(dsharp.tensor(3.))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-3.; 0.; 3.])

        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

        // keepDim = true, forward
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.variance(0,keepDim=true)
        let fwdzCorrect = dsharp.tensor([1.0])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([2.0])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)

        // keepDim = true, backward
        (* Python:
        input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
        loss = input.var(0, keepdim=True)
        loss.backward(torch.tensor([3.0]))
        input.grad
        --> tensor([-3.,  0.,  3.])
        *)
        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revz = revx.variance(0,keepDim=true)
        let revzCorrect = dsharp.tensor([1.])
        revz.reverse(dsharp.tensor([3.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-3.; 0.; 3.])

        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeStddev () =
        let fwdx = dsharp.tensor([1.; 2.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.stddev()
        let fwdzCorrect = dsharp.tensor(1.0)
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor(1.0)

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)

        (* Python:
        import torch
        input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
        loss = input.std()
        loss.backward(torch.tensor(3.0))
        input.grad
        --> tensor([-1.5000,  0.0000,  1.5000])
        *)
        let revx = dsharp.tensor([1.; 2.; 3.]).reverseDiff()
        let revz = revx.stddev()
        let revzCorrect = dsharp.tensor(1.)
        revz.reverse(dsharp.tensor(3.))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-1.5; 0.; 1.5])

        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeTransposeT2 () =
        let fwdx = dsharp.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]]).forwardDiff(dsharp.tensor([[2.; 3.; 4.]; [10.; 20.; 30.]]))
        let fwdz = fwdx.transpose()
        let fwdzCorrect = dsharp.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[2.; 10.]; [3.; 20.]; [4.; 30.]])

        let revx = dsharp.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]]).reverseDiff()
        let revz = revx.transpose()
        let revzCorrect = dsharp.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])
        revz.reverse(dsharp.tensor([[5.; 5.]; [2.; 5.]; [3.; 7.]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[5.; 2.; 3.]; [5.; 5.; 7.]])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeSignT () =
        let fwdx = dsharp.tensor([-1.; 0.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.sign()
        let fwdzCorrect = dsharp.tensor([-1.; 0.; 1.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.; 0.; 0.])

        let revx = dsharp.tensor([-1.; 0.; 3.]).reverseDiff()
        let revz = revx.sign()
        let revzCorrect = dsharp.tensor([-1.; 0.; 1.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; 0.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeFloorT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.floor()
        let fwdzCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.floor()
        let revzCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeCeilT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.ceil()
        let fwdzCorrect = dsharp.tensor([1.; 1.; 1.; 1.; 1.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.ceil()
        let revzCorrect = dsharp.tensor([1.; 1.; 1.; 1.; 1.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeRoundT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.round()
        let fwdzCorrect = dsharp.tensor([1.; 0.; 0.; 1.; 1.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.round()
        let revzCorrect = dsharp.tensor([1.; 0.; 0.; 1.; 1.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeAbsT () =
        let fwdx = dsharp.tensor([-1.; 0.; 3.]).forwardDiff(dsharp.tensor([2.; 3.; 4.]))
        let fwdz = fwdx.abs()
        let fwdzCorrect = dsharp.tensor([1.; 0.; 3.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([-2.; 0.; 4.])

        let revx = dsharp.tensor([-1.; 0.; 3.]).reverseDiff()
        let revz = revx.abs()
        let revzCorrect = dsharp.tensor([1.; 0.; 3.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-5.; 0.; 5.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeReluT () =
        let fwdx = dsharp.tensor([-1.; -2.; 0.; 3.; 10.]).forwardDiff(dsharp.tensor([2.; 3.; 4.; 5.; 6.]))
        let fwdz = fwdx.relu()
        let fwdzCorrect = dsharp.tensor([0.; 0.; 0.; 3.; 10.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.; 0.; 0.; 5.; 6.])

        let revx = dsharp.tensor([-1.; -2.; 0.; 3.; 10.]).reverseDiff()
        let revz = revx.relu()
        let revzCorrect = dsharp.tensor([0.; 0.; 0.; 3.; 10.])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; 0.; 5.; -5.])

        Assert.AreEqual(fwdzCorrect, fwdz)
        Assert.AreEqual(fwdzdCorrect, fwdzd)
        Assert.AreEqual(revzCorrect, revz)
        Assert.AreEqual(revxdCorrect, revxd)

    [<Test>]
    member this.TestDerivativeLeakyRelu () =
        let fwdx = dsharp.tensor([-1.; -2.; 0.; 3.; 10.]).forwardDiff(dsharp.tensor([2.; 3.; 4.; 5.; 6.]))
        let fwdz = fwdx.leakyRelu()
        let fwdzCorrect = dsharp.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])
        let fwdzd = fwdz.derivative
        // TODO: behavior of derivative at 0 (where it is undefined) can be reconsidered
        // let fwdzdCorrect = dsharp.tensor([0.0200; 0.0300; 0.0400; 5.; 6.])
        let fwdzdCorrect = dsharp.tensor([0.0200; 0.0300; 2.02; 5.; 6.])

        let revx = dsharp.tensor([-1.; -2.; 0.; 3.; 10.]).reverseDiff()
        let revz = revx.leakyRelu()
        let revzCorrect = dsharp.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        // TODO: behavior of derivative at 0 (where it is undefined) can be reconsidered
        // let revxdCorrect = dsharp.tensor([0.0500; 0.0500; 0.0500; 5.; -5.])
        let revxdCorrect = dsharp.tensor([0.0500; 0.0500; 2.52; 5.; -5.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeSoftplusT () =
        let fwdx = dsharp.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05]).forwardDiff(dsharp.tensor([  765080.1250,  1507281.3750,  -646660.5000,   -90687.9375, 821899.7500,  -180674.6875, -1726284.8750,   212356.4219]))
        let fwdz = fwdx.softplus()
        let fwdzCorrect = dsharp.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([ 344587.4062, 1072155.8750, -232257.6719,  -69829.2578,       0.0000, -180674.6875,      -0.0000,       0.0000])

        let revx = dsharp.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05]).reverseDiff()
        let revz = revx.softplus()
        let revzCorrect = dsharp.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])
        revz.reverse(dsharp.tensor([  765080.1250,  1507281.3750,  -646660.5000,   -90687.9375, 821899.7500,  -180674.6875, -1726284.8750,   212356.4219]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([ 344587.4062, 1072155.8750, -232257.6719,  -69829.2578,       0.0000, -180674.6875,      -0.0000,       0.0000])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeSigmoidT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.sigmoid()
        let fwdzCorrect = dsharp.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.3456; 0.0684; 0.3681; 0.2893; 0.1215])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.sigmoid()
        let revzCorrect = dsharp.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([1.0067;  1.1781;  1.2374;  1.1499; -1.0514])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeExpT () =
        let fwdx = dsharp.tensor([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).forwardDiff(dsharp.tensor([-1.9015; 0.4606; -0.1030; 0.0466; -0.2321]))
        let fwdz = fwdx.exp()
        let fwdzCorrect = dsharp.tensor([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([-2.5300; 0.1606; -0.2845; 0.0710; -0.8331])

        let revx = dsharp.tensor([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).reverseDiff()
        let revz = revx.exp()
        let revzCorrect = dsharp.tensor([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([6.6526; 1.7435; 13.8140; 7.6152; -17.9474])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeLogT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.log()
        let fwdzCorrect = dsharp.tensor([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([1.8118; 0.1951; 7.3820; 2.1624; 0.6847])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.log()
        let revzCorrect = dsharp.tensor([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([5.2780; 3.3576; 24.8177; 8.5945; -5.9248])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeLog10T () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.log10()
        let fwdzCorrect = dsharp.tensor([-0.0235;  0.1729; -0.6957; -0.2352; -0.0737])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.7869; 0.0847; 3.2054; 0.9391; 0.2974])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.log10()
        let revzCorrect = dsharp.tensor([-0.0235;  0.1729; -0.6957; -0.2352; -0.0737])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([2.2923;  1.4582; 10.7765;  3.7323; -2.5731])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        
    [<Test>]
    member this.TestDerivativeSqrtT () =
        let fwdx = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).forwardDiff(dsharp.tensor([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]))
        let fwdz = fwdx.sqrt()
        let fwdzCorrect = dsharp.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.5972; 0.1617; 0.1971; 0.5093; 0.0061])

        let revx = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).reverseDiff()
        let revz = revx.sqrt()
        let revzCorrect = dsharp.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
        revz.reverse(dsharp.tensor([7.0478; 2.0493; 1.8341; 0.0166; 9.4089]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.4761; 0.1219; 0.2286; 0.0010; 0.5166])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.05))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.05))
        Assert.True(revz.allclose(revzCorrect, 0.05))
        Assert.True(revxd.allclose(revxdCorrect, 0.05))

    [<Test>]
    member this.TestDerivativeSinT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.sin()
        let fwdzCorrect = dsharp.tensor([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([1.0022; 0.0237; 1.4571; 1.0510; 0.3840])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.sin()
        let revzCorrect = dsharp.tensor([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([2.9194;  0.4080;  4.8988;  4.1774; -3.3228])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeCosT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.cos()
        let fwdzCorrect = dsharp.tensor([0.5839; 0.0816; 0.9798; 0.8355; 0.6646])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([-1.3934; -0.2895; -0.2976; -0.6913; -0.4318])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.cos()
        let revzCorrect = dsharp.tensor([0.5839; 0.0816; 0.9798; 0.8355; 0.6646])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-4.0592; -4.9833; -1.0007; -2.7476;  3.7362])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeTanT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.tan()
        let fwdzCorrect = dsharp.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([5.0347; 43.6222;  1.5493;  1.8022;  1.3083])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.tan()
        let revzCorrect = dsharp.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([14.6665; 750.8119;   5.2086;   7.1631; -11.3217])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
    
    [<Test>]
    member this.TestDerivativeSinhT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.sinh()
        let fwdzCorrect = dsharp.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([2.5459; 0.6767; 1.5175; 1.4770; 0.7960])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.sinh()
        let revzCorrect = dsharp.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([7.4163; 11.6467;  5.1018;  5.8704; -6.8886])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeCoshT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.cosh()
        let fwdzCorrect = dsharp.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([1.8803; 0.6111; 0.3017; 0.7739; 0.5476])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.cosh()
        let revzCorrect = dsharp.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([5.4774; 10.5188;  1.0143;  3.0759; -4.7385])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeTanhT () =
        let fwdx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.tanh()
        let fwdzCorrect = dsharp.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.7802; 0.0535; 1.4284; 0.9126; 0.3044])

        let revx = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.tanh()
        let revzCorrect = dsharp.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([2.2727;  0.9215;  4.8024;  3.6273; -2.6342])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
    
    [<Test>]
    member this.TestDerivativeAsinT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.asin()
        let fwdzCorrect = dsharp.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([5.3579; 0.3331; 1.5183; 1.5467; 1.0770])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.asin()
        let revzCorrect = dsharp.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([15.6080;  5.7324;  5.1047;  6.1476; -9.3197])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeAcosT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.acos()
        let fwdzCorrect = dsharp.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([-5.3579; -0.3331; -1.5183; -1.5467; -1.0770])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.acos()
        let revzCorrect = dsharp.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([-15.6080;  -5.7324;  -5.1047;  -6.1476;   9.3197])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeAtanT () =
        let fwdx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(dsharp.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
        let fwdz = fwdx.atan()
        let fwdzCorrect = dsharp.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([0.9046; 0.2344; 1.4292; 0.9399; 0.3375])

        let revx = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
        let revz = revx.atan()
        let revzCorrect = dsharp.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])
        revz.reverse(dsharp.tensor([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([2.6352;  4.0348;  4.8049;  3.7355; -2.9203])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeStackTs () =
        let fwdxa = dsharp.tensor([1.; 2.]).forwardDiff(dsharp.tensor([10.;20.]))
        let fwdxb = dsharp.tensor([3.; 4.]).forwardDiff(dsharp.tensor([30.;40.]))
        let fwdxc = dsharp.tensor([5.; 6.]).forwardDiff(dsharp.tensor([50.;60.]))
        let fwdz = dsharp.stack([fwdxa;fwdxb;fwdxc])
        let fwdzCorrect = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]])

        let revxa = dsharp.tensor([1.; 2.]).reverseDiff()
        let revxb = dsharp.tensor([3.; 4.]).reverseDiff()
        let revxc = dsharp.tensor([5.; 6.]).reverseDiff()
        let revz = dsharp.stack([revxa;revxb;revxc])
        let revzCorrect = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]])
        revz.reverse(dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]]))
        let revxda = revxa.derivative
        let revxdb = revxb.derivative
        let revxdc = revxc.derivative
        let revxdaCorrect = dsharp.tensor([10.; 20.])
        let revxdbCorrect = dsharp.tensor([30.; 40.])
        let revxdcCorrect = dsharp.tensor([50.; 60.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxda.allclose(revxdaCorrect, 0.01))
        Assert.True(revxdb.allclose(revxdbCorrect, 0.01))
        Assert.True(revxdc.allclose(revxdcCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeUnstackT () =
        let fwdx = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]]).forwardDiff(dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]]))
        let fwdz = dsharp.unstack(fwdx) |> Seq.toArray
        let fwdza = fwdz.[0]
        let fwdzb = fwdz.[1]
        let fwdzc = fwdz.[2]
        let fwdzda = fwdza.derivative
        let fwdzdb = fwdzb.derivative
        let fwdzdc = fwdzc.derivative
        let fwdzaCorrect = dsharp.tensor([1.; 2.])
        let fwdzbCorrect = dsharp.tensor([3.; 4.])
        let fwdzcCorrect = dsharp.tensor([5.; 6.])
        let fwdzdaCorrect = dsharp.tensor([10.; 20.])
        let fwdzdbCorrect = dsharp.tensor([30.; 40.])
        let fwdzdcCorrect = dsharp.tensor([50.; 60.])

        let revx = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]]).reverseDiff()
        let revz = dsharp.unstack(revx) |> Seq.toArray
        let revza = revz.[0]
        let revzb = revz.[1]
        let revzc = revz.[2]
        let revzaCorrect = dsharp.tensor([1.; 2.])
        let revzbCorrect = dsharp.tensor([3.; 4.])
        let revzcCorrect = dsharp.tensor([5.; 6.])
        revza.reverse(dsharp.tensor([10.; 20.]))
        revzb.reverse(dsharp.tensor([30.; 40.]), zeroDerivatives=false)
        revzc.reverse(dsharp.tensor([50.; 60.]), zeroDerivatives=false)
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]])

        Assert.True(fwdza.allclose(fwdzaCorrect, 0.01))
        Assert.True(fwdzb.allclose(fwdzbCorrect, 0.01))
        Assert.True(fwdzc.allclose(fwdzcCorrect, 0.01))
        Assert.True(fwdzda.allclose(fwdzdaCorrect, 0.01))
        Assert.True(fwdzdb.allclose(fwdzdbCorrect, 0.01))
        Assert.True(fwdzdc.allclose(fwdzdcCorrect, 0.01))
        Assert.True(revza.allclose(revzaCorrect, 0.01))
        Assert.True(revzb.allclose(revzbCorrect, 0.01))
        Assert.True(revzc.allclose(revzcCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeUnstackT_Dim1 () =
        let fwdx = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]]).forwardDiff(dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]]))
        let fwdz = fwdx.unstack(dim=1) |> Seq.toArray
        let fwdza = fwdz.[0]
        let fwdzb = fwdz.[1]
        let fwdzda = fwdza.derivative
        let fwdzdb = fwdzb.derivative
        let fwdzaCorrect = dsharp.tensor([1.; 3.; 5.])
        let fwdzbCorrect = dsharp.tensor([2.; 4.; 6.])
        let fwdzdaCorrect = dsharp.tensor([10.; 30.; 50.])
        let fwdzdbCorrect = dsharp.tensor([20.; 40.; 60.])

        let revx = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]]).reverseDiff()
        let revz = revx.unstack(dim=1) |> Seq.toArray
        let revza = revz.[0]
        let revzb = revz.[1]
        let revzaCorrect = dsharp.tensor([1.; 3.; 5.])
        let revzbCorrect = dsharp.tensor([2.; 4.; 6.])
        revza.reverse(dsharp.tensor([10.; 30.; 50.]))
        revzb.reverse(dsharp.tensor([20.; 40.; 60.]), zeroDerivatives=false)
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10.;20.];[30.;40.];[50.;60.]])

        Assert.True(fwdza.allclose(fwdzaCorrect, 0.01))
        Assert.True(fwdzb.allclose(fwdzbCorrect, 0.01))
        Assert.True(fwdzda.allclose(fwdzdaCorrect, 0.01))
        Assert.True(fwdzdb.allclose(fwdzdbCorrect, 0.01))
        Assert.True(revza.allclose(revzaCorrect, 0.01))
        Assert.True(revzb.allclose(revzbCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeCatTs () =
        let fwdxa = dsharp.tensor([1.; 2.]).forwardDiff(dsharp.tensor([10.;20.]))
        let fwdxb = dsharp.tensor([3.; 4.]).forwardDiff(dsharp.tensor([30.;40.]))
        let fwdxc = dsharp.tensor([5.; 6.]).forwardDiff(dsharp.tensor([50.;60.]))
        let fwdz = Tensor.cat([fwdxa;fwdxb;fwdxc])
        let fwdzCorrect = dsharp.tensor([1.;2.;3.;4.;5.;6.])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([10.;20.;30.;40.;50.;60.])

        let revxa = dsharp.tensor([1.; 2.]).reverseDiff()
        let revxb = dsharp.tensor([3.; 4.]).reverseDiff()
        let revxc = dsharp.tensor([5.; 6.]).reverseDiff()
        let revz = Tensor.cat([revxa;revxb;revxc])
        let revzCorrect = dsharp.tensor([1.;2.;3.;4.;5.;6.])
        revz.reverse(dsharp.tensor([10.;20.;30.;40.;50.;60.]))
        let revxda = revxa.derivative
        let revxdb = revxb.derivative
        let revxdc = revxc.derivative
        let revxdaCorrect = dsharp.tensor([10.; 20.])
        let revxdbCorrect = dsharp.tensor([30.; 40.])
        let revxdcCorrect = dsharp.tensor([50.; 60.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxda.allclose(revxdaCorrect, 0.01))
        Assert.True(revxdb.allclose(revxdbCorrect, 0.01))
        Assert.True(revxdc.allclose(revxdcCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeSplitT () =
        let fwdx = dsharp.tensor([1.;2.;3.;4.;5.;6.]).forwardDiff(dsharp.tensor([10.;20.;30.;40.;50.;60.]))
        let fwdz = fwdx.split([| 1;3;2 |])
        let fwdza = fwdz.[0]
        let fwdzb = fwdz.[1]
        let fwdzc = fwdz.[2]
        let fwdzda = fwdza.derivative
        let fwdzdb = fwdzb.derivative
        let fwdzdc = fwdzc.derivative
        let fwdzaCorrect = dsharp.tensor([1.])
        let fwdzbCorrect = dsharp.tensor([2.; 3.; 4.])
        let fwdzcCorrect = dsharp.tensor([5.; 6.])
        let fwdzdaCorrect = dsharp.tensor([10.])
        let fwdzdbCorrect = dsharp.tensor([20.;30.; 40.])
        let fwdzdcCorrect = dsharp.tensor([50.; 60.])

        let revx = dsharp.tensor([1.;2.;3.;4.;5.;6.]).reverseDiff()
        let revz = revx.split([| 1;3;2 |])
        let revza = revz.[0]
        let revzb = revz.[1]
        let revzc = revz.[2]
        let revzaCorrect = dsharp.tensor([1.])
        let revzbCorrect = dsharp.tensor([2.;3.; 4.])
        let revzcCorrect = dsharp.tensor([5.; 6.])
        revza.reverse(dsharp.tensor([10.]))
        revzb.reverse(dsharp.tensor([20.;30.; 40.]), zeroDerivatives=false)
        revzc.reverse(dsharp.tensor([50.; 60.]), zeroDerivatives=false)
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([10.;20.;30.;40.;50.;60.])

        Assert.True(fwdza.allclose(fwdzaCorrect, 0.01))
        Assert.True(fwdzb.allclose(fwdzbCorrect, 0.01))
        Assert.True(fwdzc.allclose(fwdzcCorrect, 0.01))
        Assert.True(fwdzda.allclose(fwdzdaCorrect, 0.01))
        Assert.True(fwdzdb.allclose(fwdzdbCorrect, 0.01))
        Assert.True(fwdzdc.allclose(fwdzdcCorrect, 0.01))
        Assert.True(revza.allclose(revzaCorrect, 0.01))
        Assert.True(revzb.allclose(revzbCorrect, 0.01))
        Assert.True(revzc.allclose(revzcCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeSliceT () =
        let fwdx = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).forwardDiff(dsharp.tensor([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]))
        let fwdz = fwdx.[2..]
        let fwdzCorrect = dsharp.tensor([16.0868; 74.5486; 82.9318])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([1.5814; 8.7951; 0.1119])

        let revx = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).reverseDiff()
        let revz = revx.[2..]
        let revzCorrect = dsharp.tensor([16.0868; 74.5486; 82.9318])
        revz.reverse(dsharp.tensor([0.9360; 0.8748; 0.4353]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; 0.9360; 0.8748; 0.4353])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeAddTTSlice () =
        let fwdx = dsharp.tensor([[-0.2754;  0.0172;  0.7105];
            [-0.1890;  1.7664;  0.5377];
            [-0.5313; -2.2530; -0.6235];
            [ 0.6776;  1.5844; -0.5686]])
        let fwdx = fwdx.forwardDiff(dsharp.tensor([[-0.0552;  0.6113; -0.2341];
            [ 1.4232; -1.2062;  0.3189];
            [ 0.6859; -0.3385; -0.1263];
            [-0.5159; -1.1882; -1.3437]]))
        let fwdy = dsharp.tensor([[-111.8892;   -7.0328];
            [  18.7557;  -86.2308]])            
        let fwdy = fwdy.forwardDiff(dsharp.tensor([[ 1.3431; 23.0647];
            [71.1838; 39.8339]]))        
        let fwdz = fwdx.addSlice([0;1], fwdy)
        let fwdzCorrect = dsharp.tensor([[  -0.2754; -111.8720;   -6.3222];
            [  -0.1890;   20.5221;  -85.6932];
            [  -0.5313;   -2.2530;   -0.6235];
            [   0.6776;    1.5844;   -0.5686]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[-5.5237e-02;  1.9544e+00;  2.2831e+01];
            [ 1.4232e+00;  6.9978e+01;  4.0153e+01];
            [ 6.8592e-01; -3.3845e-01; -1.2635e-01];
            [-5.1592e-01; -1.1882e+00; -1.3437e+00]])

        let revx = dsharp.tensor([[-0.2754;  0.0172;  0.7105];
            [-0.1890;  1.7664;  0.5377];
            [-0.5313; -2.2530; -0.6235];
            [ 0.6776;  1.5844; -0.5686]]).reverseDiff()
        let revy = dsharp.tensor([[-111.8892;   -7.0328];
            [  18.7557;  -86.2308]]).reverseDiff()
        let revz = revx.addSlice([0;1], revy)
        let revzCorrect = dsharp.tensor([[  -0.2754; -111.8720;   -6.3222];
            [  -0.1890;   20.5221;  -85.6932];
            [  -0.5313;   -2.2530;   -0.6235];
            [   0.6776;    1.5844;   -0.5686]])
        revz.reverse(dsharp.tensor([[ 1.2453;  1.2199; -0.5281];
            [ 1.2203; -0.8378; -0.3876];
            [ 0.3626; -0.1200; -0.1496];
            [-0.6304;  1.0198; -0.4969]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[ 1.2453;  1.2199; -0.5281];
            [ 1.2203; -0.8378; -0.3876];
            [ 0.3626; -0.1200; -0.1496];
            [-0.6304;  1.0198; -0.4969]])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([[1.2199; -0.5281]; [-0.8378; -0.3876]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeSqueezeT () =
        let fwdx = dsharp.tensor([[[1.; 2.]]; [[3.;4.]]]).forwardDiff(dsharp.tensor([[[10.; 20.]]; [[30.;40.]]]))
        let fwdz = fwdx.squeeze()
        let fwdzCorrect =  dsharp.tensor([[1.;2.];[3.;4.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect =  dsharp.tensor([[10.;20.];[30.;40.]])

        let revx = dsharp.tensor([[[1.; 2.]]; [[3.;4.]]]).reverseDiff()
        let revz = revx.squeeze()
        let revzCorrect =  dsharp.tensor([[1.;2.];[3.;4.]])
        revz.reverse(dsharp.tensor([[10.;20.];[30.;40.]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[[10.; 20.]]; [[30.;40.]]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeUnsqueezeT () =
        let fwdx = dsharp.tensor([[1.;2.];[3.;4.]]).forwardDiff(dsharp.tensor([[10.;20.];[30.;40.]]))
        let fwdz = fwdx.unsqueeze(1)
        let fwdzCorrect =  dsharp.tensor([[[1.; 2.]]; [[3.;4.]]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect =  dsharp.tensor([[[10.; 20.]]; [[30.;40.]]])

        let revx = dsharp.tensor([[1.;2.];[3.;4.]]).reverseDiff()
        let revz = revx.unsqueeze(1)
        let revzCorrect =  dsharp.tensor([[[1.; 2.]]; [[3.;4.]]])
        revz.reverse(dsharp.tensor([[[10.; 20.]]; [[30.;40.]]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10.;20.];[30.;40.]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeFlipT () =
        let fwdx = dsharp.tensor([[1.;2.];[3.;4.]]).forwardDiff(dsharp.tensor([[10.;20.];[30.;40.]]))
        let fwdz = fwdx.flip([|0; 1|])
        let fwdzCorrect =  dsharp.tensor([[4.; 3.]; [2.;1.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect =  dsharp.tensor([[40.; 30.]; [20.;10.]])

        let revx = dsharp.tensor([[1.;2.];[3.;4.]]).reverseDiff()
        let revz = revx.flip([|0; 1|])
        let revzCorrect =  dsharp.tensor([[4.; 3.]; [2.;1.]])
        revz.reverse(dsharp.tensor([[40.; 30.]; [20.;10.]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10.;20.];[30.;40.]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeDilateT () =
        let fwdx = dsharp.tensor([[1.;2.];[3.;4.]]).forwardDiff(dsharp.tensor([[10.;20.];[30.;40.]]))
        let fwdz = fwdx.dilate([|2; 2|])
        let fwdzCorrect =  dsharp.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect =  dsharp.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]])

        let revx = dsharp.tensor([[1.;2.];[3.;4.]]).reverseDiff()
        let revz = revx.dilate([|2; 2|])
        let revzCorrect =  dsharp.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]])
        revz.reverse(dsharp.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10.;20.];[30.;40.]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeUndilateT () =
        let fwdx = dsharp.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]]).forwardDiff(dsharp.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]]))
        let fwdz = fwdx.undilate([|2; 2|])
        let fwdzCorrect =  dsharp.tensor([[1.;2.];[3.;4.]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect =  dsharp.tensor([[10.;20.];[30.;40.]])

        let revx = dsharp.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]]).reverseDiff()
        let revz = revx.undilate([|2; 2|])
        let revzCorrect =  dsharp.tensor([[1.;2.];[3.;4.]])
        revz.reverse(dsharp.tensor([[10.;20.];[30.;40.]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeSoftmax () =
        let fwdx = dsharp.tensor([[4.6815; 5.6441; 7.4689];
            [9.1976; 8.1241; 7.4521]]).forwardDiff(dsharp.tensor([[8.0030; 7.0798; 6.8637];
                [9.5760; 7.4524; 2.6404]]))
        let fwdz = fwdx.softmax(dim=1)
        let fwdzCorrect = dsharp.tensor([[0.0504; 0.1319; 0.8178];
            [0.6595; 0.2254; 0.1151]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[0.0530; 0.0172; -0.0702]; [0.8422; -0.1908; -0.6514]])

        let revx = dsharp.tensor([[4.6815; 5.6441; 7.4689];
            [9.1976; 8.1241; 7.4521]]).reverseDiff()
        let revz = revx.softmax(dim=1)
        let revzCorrect = dsharp.tensor([[0.0504; 0.1319; 0.8178];
            [0.6595; 0.2254; 0.1151]])
        revz.reverse(dsharp.tensor([[6.0933; 9.6456; 7.0996];
            [0.2617; 1.7002; 4.9711]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[-0.0649; 0.2988; -0.2329]; [-0.5713; 0.1291; 0.4426]])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeMaxBinary () =
        let fwdx = dsharp.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).forwardDiff(dsharp.tensor([1.9788; 0.2861; 4.2025; 0.5602; 7.9510]))
        let fwdy = dsharp.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).forwardDiff(dsharp.tensor([9.6600; 6.9111; 9.7303; 0.1491; 7.7003]))
        let fwdz = dsharp.max(fwdx, fwdy)
        let fwdzCorrect = dsharp.tensor([19.3520;  8.9730;  4.2405; -3.5977; 12.2048])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([1.9788; 0.2861; 9.7303; 0.5602; 7.7003])

        let revx = dsharp.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).reverseDiff()
        let revy = dsharp.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).reverseDiff()
        let revz = dsharp.max(revx, revy)
        let revzCorrect = dsharp.tensor([19.3520;  8.9730;  4.2405; -3.5977; 12.2048])
        revz.reverse(dsharp.tensor([  9.7293; -10.2704; -13.7527;  -3.9050;  -1.6439]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([9.7293; -10.2704; 0.; -3.9050; 0.])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([0.; 0.; -13.7527; 0.; -1.6439])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member this.TestDerivativeMinBinary () =
        let fwdx = dsharp.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).forwardDiff(dsharp.tensor([1.9788; 0.2861; 4.2025; 0.5602; 7.9510]))
        let fwdy = dsharp.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).forwardDiff(dsharp.tensor([9.6600; 6.9111; 9.7303; 0.1491; 7.7003]))
        let fwdz = dsharp.min(fwdx, fwdy)
        let fwdzCorrect = dsharp.tensor([-17.1885;  -4.0684; -23.6274; -21.7158; -20.3245])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([9.6600; 6.9111; 4.2025; 0.1491; 7.9510])

        let revx = dsharp.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).reverseDiff()
        let revy = dsharp.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).reverseDiff()
        let revz = dsharp.min(revx, revy)
        let revzCorrect = dsharp.tensor([-17.1885;  -4.0684; -23.6274; -21.7158; -20.3245])
        revz.reverse(dsharp.tensor([  9.7293; -10.2704; -13.7527;  -3.9050;  -1.6439]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([0.; 0.; -13.7527; 0.; -1.6439])
        let revyd = revy.derivative
        let revydCorrect = dsharp.tensor([9.7293; -10.2704; 0.; -3.9050; 0.])

        Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
        Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
        Assert.True(revz.allclose(revzCorrect, 0.01))
        Assert.True(revxd.allclose(revxdCorrect, 0.01))
        Assert.True(revyd.allclose(revydCorrect, 0.01))