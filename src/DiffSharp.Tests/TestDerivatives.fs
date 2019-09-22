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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx + fwdy
        let fwdzCorrect = Tensor.Create([6.; 8.; 10.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([4.; 5.; 7.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([[1.; 2.]; [3.; 4.]]).GetForward(Tensor.Create([[2.; 3.]; [4.; 5.]]), 1u)
        let fwdy = Tensor.Create([5.; 6.]).GetForward(Tensor.Create([2.; 3.]), 1u)
        let fwdz = fwdx + fwdy
        let fwdzCorrect = Tensor.Create([[6.; 8.]; [8.; 10.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[4.; 6.]; [6.; 8.]])

        let revx = Tensor.Create([[1.; 2.]; [3.; 4.]]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx - fwdy
        let fwdzCorrect = Tensor.Create([-4.; -4.; -4.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 1.; 1.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx * fwdy
        let fwdzCorrect = Tensor.Create([5.; 12.; 21.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([12.; 22.; 37.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx / fwdy
        let fwdzCorrect = Tensor.Create([0.2; 0.333333; 0.428571])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.32; 0.388889; 0.387755])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx ** fwdy
        let fwdzCorrect = Tensor.Create([1.; 64.; 2187.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([10.; 664.723; 27620.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
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
    member this.TestDerivativeMatMulT2T2 () =
        let fwdx = Tensor.Create([[6.2381; 0.0393; 8.2364; 3.9906; 6.2291];
            [9.8762; 3.2263; 6.2866; 4.7111; 0.0652];
            [3.5832; 7.9801; 1.9854; 4.4965; 4.1712]])
        let fwdx = fwdx.GetForward(Tensor.Create([[4.6453; 8.4388; 4.6549; 9.5680; 1.5756];
            [3.2066; 4.2429; 2.2028; 9.1037; 3.4022];
            [4.2324; 4.5508; 3.4755; 2.7196; 5.5344]]), 1u)
        let fwdy = Tensor.Create([[4.4220; 3.7293];
            [6.1928; 2.1446];
            [0.0525; 1.2494];
            [7.5281; 1.4816];
            [5.0328; 2.2756]])
        let fwdy = fwdy.GetForward(Tensor.Create([[1.4749; 9.7608];
            [3.6599; 7.9553];
            [3.5503; 1.3757];
            [8.3172; 6.6748];
            [2.2959; 0.6784]]), 1u)
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
            [3.5832; 7.9801; 1.9854; 4.4965; 4.1712]]).GetReverse(1u)
        let revy = Tensor.Create([[4.4220; 3.7293];
            [6.1928; 2.1446];
            [0.0525; 1.2494];
            [7.5281; 1.4816];
            [5.0328; 2.2756]]).GetReverse(1u)
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
        let fwdxa = Tensor.Create([1.; 2.]).GetForward(Tensor.Create([10.; 20.]), 1u)
        let fwdxb = Tensor.Create([3.; 4.]).GetForward(Tensor.Create([30.; 40.]), 1u)
        let fwdxc = Tensor.Create([5.; 6.]).GetForward(Tensor.Create([50.; 60.]), 1u)
        let fwdz = Tensor.Stack([fwdxa;fwdxb;fwdxc])
        let fwdzCorrect = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[10.;20.];[30.;40.];[50.;60.]])

        let revxa = Tensor.Create([1.; 2.]).GetReverse(1u)
        let revxb = Tensor.Create([3.; 4.]).GetReverse(1u)
        let revxc = Tensor.Create([5.; 6.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdz = -fwdx
        let fwdzCorrect = Tensor.Create([-1.; -2.; -3.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.; -3.; -4.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdz = fwdx.Sum()
        let fwdzCorrect = Tensor.Create(6.)
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create(9.)

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
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
        let fwdx = Tensor.Create([[1.; 2.]; [3.; 4.]]).GetForward(Tensor.Create([[2.; 3.]; [4.; 5.]]), 1u)
        let fwdz = fwdx.SumT2Dim0()
        let fwdzCorrect = Tensor.Create([4.; 6.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([6.; 8.])

        let revx = Tensor.Create([[1.; 2.]; [3.; 4.]]).GetReverse(1u)
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
        let fwdx = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]]).GetForward(Tensor.Create([[2.; 3.; 4.]; [10.; 20.; 30.]]), 1u)
        let fwdz = fwdx.Transpose()
        let fwdzCorrect = Tensor.Create([[1.; 4.]; [2.; 5.]; [3.; 6.]])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([[2.; 10.]; [3.; 20.]; [4.; 30.]])

        let revx = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]]).GetReverse(1u)
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
        let fwdx = Tensor.Create([-1.; 0.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdz = fwdx.Sign()
        let fwdzCorrect = Tensor.Create([-1.; 0.; 1.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.])

        let revx = Tensor.Create([-1.; 0.; 3.]).GetReverse(1u)
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
    member this.TestDerivativeAbsT () =
        let fwdx = Tensor.Create([-1.; 0.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdz = fwdx.Abs()
        let fwdzCorrect = Tensor.Create([1.; 0.; 3.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.; 0.; 4.])

        let revx = Tensor.Create([-1.; 0.; 3.]).GetReverse(1u)
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
    member this.TestDerivativeReLUT () =
        let fwdx = Tensor.Create([-1.; -2.; 0.; 3.; 10.]).GetForward(Tensor.Create([2.; 3.; 4.; 5.; 6.]), 1u)
        let fwdz = fwdx.ReLU()
        let fwdzCorrect = Tensor.Create([0.; 0.; 0.; 3.; 10.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 0.; 0.; 5.; 6.])

        let revx = Tensor.Create([-1.; -2.; 0.; 3.; 10.]).GetReverse(1u)
        let revz = revx.ReLU()
        let revzCorrect = Tensor.Create([0.; 0.; 0.; 3.; 10.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.; 5.; -5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.; 0.; 0.; 5.; -5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeExpT () =
        let fwdx = Tensor.Create([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).GetForward(Tensor.Create([-1.9015; 0.4606; -0.1030; 0.0466; -0.2321]), 1u)
        let fwdz = fwdx.Exp()
        let fwdzCorrect = Tensor.Create([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.5300; 0.1606; -0.2845; 0.0710; -0.8331])

        let revx = Tensor.Create([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).GetReverse(1u)
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
        let fwdx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).GetForward(Tensor.Create([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]), 1u)
        let fwdz = fwdx.Log()
        let fwdzCorrect = Tensor.Create([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([1.8118; 0.1951; 7.3820; 2.1624; 0.6847])

        let revx = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).GetReverse(1u)
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
    member this.TestDerivativeSqrtT () =
        let fwdx = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).GetForward(Tensor.Create([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]), 1u)
        let fwdz = fwdx.Sqrt()
        let fwdzCorrect = Tensor.Create([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.5972; 0.1617; 0.1971; 0.5093; 0.0061])

        let revx = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).GetReverse(1u)
        let revz = revx.Sqrt()
        let revzCorrect = Tensor.Create([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
        revz.Reverse(Tensor.Create([7.0478; 2.0493; 1.8341; 0.0166; 9.4089]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([0.4761; 0.1219; 0.2286; 0.0010; 0.5166])

        Assert.True(fwdz.ApproximatelyEqual(fwdzCorrect))
        Assert.True(fwdzd.ApproximatelyEqual(fwdzdCorrect))
        Assert.True(revz.ApproximatelyEqual(revzCorrect))
        Assert.True(revxd.ApproximatelyEqual(revxdCorrect))
          