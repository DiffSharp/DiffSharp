namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestTensor () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestTensorCreate () =
        let t0 = Tensor.Create(1.)
        let t0Shape = t0.Shape
        let t0Dim = t0.Dim
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Shape = t1.Shape
        let t1Dim = t1.Dim
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        let t2 = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]])
        let t2Shape = t2.Shape
        let t2Dim = t2.Dim
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2

        let t3 = Tensor.Create([[[1.; 2.; 3.]; [4.; 5.; 6.]]])
        let t3Shape = t3.Shape
        let t3Dim = t3.Dim
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3

        let t4 = Tensor.Create([[[[1.; 2.]]]])
        let t4Shape = t4.Shape
        let t4Dim = t4.Dim
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4

        Assert.AreEqual(t0Shape, t0ShapeCorrect)
        Assert.AreEqual(t1Shape, t1ShapeCorrect)
        Assert.AreEqual(t2Shape, t2ShapeCorrect)
        Assert.AreEqual(t3Shape, t3ShapeCorrect)
        Assert.AreEqual(t4Shape, t4ShapeCorrect)
        Assert.AreEqual(t0Dim, t0DimCorrect)
        Assert.AreEqual(t1Dim, t1DimCorrect)
        Assert.AreEqual(t2Dim, t2DimCorrect)
        Assert.AreEqual(t3Dim, t3DimCorrect)
        Assert.AreEqual(t4Dim, t4DimCorrect)

    [<Test>]
    member this.TestTensorToArray () =
        let a = array2D [[1.; 2.]; [3.; 4.]]
        let t = Tensor.Create(a)
        let v = t.ToArray()
        Assert.AreEqual(a, v)

    [<Test>]
    member this.TestTensorToString () =
        let t0 = Tensor.Create(2.)
        let t1 = Tensor.Create([[2.]; [2.]])
        let t2 = Tensor.Create([[[2.; 2.]]])
        let t3 = Tensor.Create([[1.;2.]; [3.;4.]])
        let t4 = Tensor.Create([[[[1.]]]])
        let t0String = t0.ToString()
        let t1String = t1.ToString()
        let t2String = t2.ToString()
        let t3String = t3.ToString()
        let t4String = t4.ToString()
        let t0StringCorrect = "Tensor 2.0f"
        let t1StringCorrect = "Tensor [[2.0f]; [2.0f]]"
        let t2StringCorrect = "Tensor [[[2.0f; 2.0f]]]"
        let t3StringCorrect = "Tensor [[1.0f; 2.0f]; [3.0f; 4.0f]]"
        let t4StringCorrect = "Tensor [[[[1.0f]]]]"
        Assert.AreEqual(t0String, t0StringCorrect)
        Assert.AreEqual(t1String, t1StringCorrect)
        Assert.AreEqual(t2String, t2StringCorrect)
        Assert.AreEqual(t3String, t3StringCorrect)
        Assert.AreEqual(t4String, t4StringCorrect)

    [<Test>]
    member this.TestTensorCompare () =
        let t1 = Tensor.Create(-1.)
        let t2 = Tensor.Create(1.)
        let t3 = Tensor.Create(1.)
        let t1t2Less = t1 < t2
        let t1t2LessCorrect = true
        let t1t2Equal = t1 = t2
        let t1t2EqualCorrect = false
        let t2t3Equal = t2 = t3
        let t2t3EqualCorrect = true

        Assert.AreEqual(t1t2Less, t1t2LessCorrect)
        Assert.AreEqual(t1t2Equal, t1t2EqualCorrect)
        Assert.AreEqual(t2t3Equal, t2t3EqualCorrect)

    [<Test>]
    member this.TestTensorLtTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Lt = t1.Lt(t2)
        let t1t2LtCorrect = Tensor.Create([0.; 1.; 1.; 0.])

        Assert.AreEqual(t1t2Lt, t1t2LtCorrect)

    [<Test>]
    member this.TestTensorLeTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Le = t1.Le(t2)
        let t1t2LeCorrect = Tensor.Create([1.; 1.; 1.; 0.])

        Assert.AreEqual(t1t2Le, t1t2LeCorrect)

    [<Test>]
    member this.TestTensorGtTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Gt = t1.Gt(t2)
        let t1t2GtCorrect = Tensor.Create([0.; 0.; 0.; 1.])

        Assert.AreEqual(t1t2Gt, t1t2GtCorrect)

    [<Test>]
    member this.TestTensorGeTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Ge = t1.Ge(t2)
        let t1t2GeCorrect = Tensor.Create([1.; 0.; 0.; 1.])

        Assert.AreEqual(t1t2Ge, t1t2GeCorrect)

    [<Test>]
    member this.TestTensorAddTT () =
        let t1 = Tensor.Create([1.; 2.]) + Tensor.Create([3.; 4.])
        let t1Correct = Tensor.Create([4.; 6.])

        let t2 = Tensor.Create([1.; 2.]) + Tensor.Create(5.)
        let t2Correct = Tensor.Create([6.; 7.])

        let t3 = Tensor.Create([1.; 2.]) + 5.f
        let t3Correct = Tensor.Create([6.; 7.])

        let t4 = Tensor.Create([1.; 2.]) + 5.
        let t4Correct = Tensor.Create([6.; 7.])

        let t5 = Tensor.Create([1.; 2.]) + 5
        let t5Correct = Tensor.Create([6.; 7.])

        Assert.AreEqual(t1, t1Correct)
        Assert.AreEqual(t2, t2Correct)
        Assert.AreEqual(t3, t3Correct)
        Assert.AreEqual(t4, t4Correct)
        Assert.AreEqual(t5, t5Correct)

    [<Test>]
    member this.TestTensorStackTs () =
        let t0a = Tensor.Create(1.)
        let t0b = Tensor.Create(3.)
        let t0c = Tensor.Create(5.)
        let t0 = Tensor.Stack([t0a;t0b;t0c])
        let t0Correct = Tensor.Create([1.;3.;5.])

        let t1a = Tensor.Create([1.; 2.])
        let t1b = Tensor.Create([3.; 4.])
        let t1c = Tensor.Create([5.; 6.])
        let t1 = Tensor.Stack([t1a;t1b;t1c])
        let t1Correct = Tensor.Create([[1.;2.];[3.;4.];[5.;6.]])

        Assert.AreEqual(t0, t0Correct)
        Assert.AreEqual(t1, t1Correct)

    [<Test>]
    member this.TestTensorUnstackT () =
        let t0a = Tensor.Create(1.)
        let t0b = Tensor.Create(3.)
        let t0c = Tensor.Create(5.)
        let t0Correct = [t0a;t0b;t0c]
        let t0 = Tensor.Stack(t0Correct).Unstack()

        let t1a = Tensor.Create([1.; 2.])
        let t1b = Tensor.Create([3.; 4.])
        let t1c = Tensor.Create([5.; 6.])
        let t1Correct = [t1a;t1b;t1c]
        let t1 = Tensor.Stack(t1Correct).Unstack()

        Assert.AreEqual(t0, t0Correct)
        Assert.AreEqual(t1, t1Correct)

    [<Test>]
    member this.TestTensorAddT2T1 () =
        let t1 = Tensor.Create([[1.; 2.]; [3.; 4.]]) + Tensor.Create([5.; 6.])
        let t1Correct = Tensor.Create([[6.; 8.]; [8.; 10.]])

        Assert.AreEqual(t1, t1Correct)

    [<Test>]
    member this.TestTensorSubTT () =
        let t1 = Tensor.Create([1.; 2.]) - Tensor.Create([3.; 4.])
        let t1Correct = Tensor.Create([-2.; -2.])

        let t2 = Tensor.Create([1.; 2.]) - Tensor.Create(5.)
        let t2Correct = Tensor.Create([-4.; -3.])

        let t3 = Tensor.Create([1.; 2.]) - 5.f
        let t3Correct = Tensor.Create([-4.; -3.])

        let t4 = 5. - Tensor.Create([1.; 2.])
        let t4Correct = Tensor.Create([4.; 3.])

        Assert.AreEqual(t1, t1Correct)
        Assert.AreEqual(t2, t2Correct)
        Assert.AreEqual(t3, t3Correct)
        Assert.AreEqual(t4, t4Correct)

    [<Test>]
    member this.TestTensorMulTT () =
        let t1 = Tensor.Create([1.; 2.]) * Tensor.Create([3.; 4.])
        let t1Correct = Tensor.Create([3.; 8.])

        let t2 = Tensor.Create([1.; 2.]) * Tensor.Create(5.)
        let t2Correct = Tensor.Create([5.; 10.])

        let t3 = Tensor.Create([1.; 2.]) * 5.f
        let t3Correct = Tensor.Create([5.; 10.])

        let t4 = 5. * Tensor.Create([1.; 2.])
        let t4Correct = Tensor.Create([5.; 10.])

        Assert.AreEqual(t1, t1Correct)
        Assert.AreEqual(t2, t2Correct)
        Assert.AreEqual(t3, t3Correct)
        Assert.AreEqual(t4, t4Correct)

    [<Test>]
    member this.TestTensorDivTT () =
        let t1 = Tensor.Create([1.; 2.]) / Tensor.Create([3.; 4.])
        let t1Correct = Tensor.Create([0.333333; 0.5])

        let t2 = Tensor.Create([1.; 2.]) / Tensor.Create(5.)
        let t2Correct = Tensor.Create([0.2; 0.4])

        let t3 = Tensor.Create([1.; 2.]) / 5.
        let t3Correct = Tensor.Create([0.2; 0.4])

        let t4 = 5. / Tensor.Create([1.; 2.])
        let t4Correct = Tensor.Create([5.; 2.5])

        Assert.True(t1.ApproximatelyEqual(t1Correct))
        Assert.True(t2.ApproximatelyEqual(t2Correct))
        Assert.True(t3.ApproximatelyEqual(t3Correct))
        Assert.True(t4.ApproximatelyEqual(t4Correct))

    [<Test>]
    member this.TestTensorPowTT () =
        let t1 = Tensor.Create([1.; 2.]) ** Tensor.Create([3.; 4.])
        let t1Correct = Tensor.Create([1.; 16.])

        let t2 = Tensor.Create([1.; 2.]) ** Tensor.Create(5.)
        let t2Correct = Tensor.Create([1.; 32.])

        let t3 = Tensor.Create(5.) ** Tensor.Create([1.; 2.])
        let t3Correct = Tensor.Create([5.; 25.])

        Assert.AreEqual(t1, t1Correct)
        Assert.AreEqual(t2, t2Correct)
        Assert.AreEqual(t3, t3Correct)

    [<Test>]
    member this.TestTensorMatMulT2T2 () =
        let t1 = Tensor.Create([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
        let t2 = Tensor.Create([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])

        let t3 = Tensor.MatMul(t1, t2)
        let t3Correct = Tensor.Create([[118.0367; 56.6266];
                                        [190.5926; 90.8155];
                                        [134.3925; 64.1030]])

        Assert.True(t3.ApproximatelyEqual(t3Correct))

    [<Test>]
    member this.TestTensorConv1D () =
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

        let t3 = Tensor.Conv1D(t1, t2)
        let t3Correct = Tensor.Create([[[2.8516; 2.0732; 2.6420];
                                         [2.3239; 1.7078; 2.7450]];

                                        [[3.0127; 2.9651; 2.5219];
                                         [3.0899; 3.1496; 2.4110]];

                                        [[3.4749; 2.9038; 2.7131];
                                         [2.7692; 2.9444; 3.2554]]])

        let t3p1 = Tensor.Conv1D(t1, t2, padding=1)
        let t3p1Correct = Tensor.Create([[[1.4392; 2.8516; 2.0732; 2.6420; 2.1177];
                                         [1.4345; 2.3239; 1.7078; 2.7450; 2.1474]];

                                        [[2.4208; 3.0127; 2.9651; 2.5219; 1.2960];
                                         [1.5544; 3.0899; 3.1496; 2.4110; 1.8567]];

                                        [[1.2965; 3.4749; 2.9038; 2.7131; 1.7408];
                                         [1.3549; 2.7692; 2.9444; 3.2554; 1.2120]]])

        let t3p2 = Tensor.Conv1D(t1, t2, padding=2)
        let t3p2Correct = Tensor.Create([[[0.6333; 1.4392; 2.8516; 2.0732; 2.6420; 2.1177; 1.0258];
                                         [0.6539; 1.4345; 2.3239; 1.7078; 2.7450; 2.1474; 1.2585]];

                                        [[0.5982; 2.4208; 3.0127; 2.9651; 2.5219; 1.2960; 1.0620];
                                         [0.5157; 1.5544; 3.0899; 3.1496; 2.4110; 1.8567; 1.3182]];

                                        [[0.3165; 1.2965; 3.4749; 2.9038; 2.7131; 1.7408; 0.5275];
                                         [0.3861; 1.3549; 2.7692; 2.9444; 3.2554; 1.2120; 0.7428]]])

        let t3s2 = Tensor.Conv1D(t1, t2, stride=2)
        let t3s2Correct = Tensor.Create([[[2.8516; 2.6420];
                                         [2.3239; 2.7450]];

                                        [[3.0127; 2.5219];
                                         [3.0899; 2.4110]];

                                        [[3.4749; 2.7131];
                                         [2.7692; 3.2554]]])

        let t3s3 = Tensor.Conv1D(t1, t2, stride=3)
        let t3s3Correct = Tensor.Create([[[2.8516];
                                         [2.3239]];

                                        [[3.0127];
                                         [3.0899]];

                                        [[3.4749];
                                         [2.7692]]])

        let t3p1s2 = Tensor.Conv1D(t1, t2, padding=1, stride=2)
        let t3p1s2Correct = Tensor.Create([[[1.4392; 2.0732; 2.1177];
                                             [1.4345; 1.7078; 2.1474]];

                                            [[2.4208; 2.9651; 1.2960];
                                             [1.5544; 3.1496; 1.8567]];

                                            [[1.2965; 2.9038; 1.7408];
                                             [1.3549; 2.9444; 1.2120]]])

        let t3p2s3 = Tensor.Conv1D(t1, t2, padding=2, stride=3)
        let t3p2s3Correct = Tensor.Create([[[0.6333; 2.0732; 1.0258];
                                             [0.6539; 1.7078; 1.2585]];

                                            [[0.5982; 2.9651; 1.0620];
                                             [0.5157; 3.1496; 1.3182]];

                                            [[0.3165; 2.9038; 0.5275];
                                             [0.3861; 2.9444; 0.7428]]])

        Assert.True(t3.ApproximatelyEqual(t3Correct))
        Assert.True(t3p1.ApproximatelyEqual(t3p1Correct))
        Assert.True(t3p2.ApproximatelyEqual(t3p2Correct))
        Assert.True(t3s2.ApproximatelyEqual(t3s2Correct))
        Assert.True(t3s3.ApproximatelyEqual(t3s3Correct))
        Assert.True(t3p1s2.ApproximatelyEqual(t3p1s2Correct))
        Assert.True(t3p2s3.ApproximatelyEqual(t3p2s3Correct))

    [<Test>]
    member this.TestTensorNegT () =
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Neg = -t1
        let t1NegCorrect = Tensor.Create([-1.; -2.; -3.])

        Assert.AreEqual(t1Neg, t1NegCorrect)

    [<Test>]
    member this.TestTensorSumT () =
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Sum = t1.Sum()
        let t1SumCorrect = Tensor.Create(6.)

        let t2 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t2Sum = t2.Sum()
        let t2SumCorrect = Tensor.Create(10.)

        Assert.AreEqual(t1Sum, t1SumCorrect)
        Assert.AreEqual(t2Sum, t2SumCorrect)

    [<Test>]
    member this.TestTensorSumT2Dim0 () =
        let t1 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t1Sum = t1.SumT2Dim0()
        let t1SumCorrect = Tensor.Create([4.; 6.])

        Assert.AreEqual(t1Sum, t1SumCorrect)
    
    [<Test>]
    member this.TestTensorSumDim () =
        let t = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tSum0 = t.Sum(0)
        let tSum0Correct = Tensor.Create([[14.0f; 16.0f; 18.0f; 20.0f]; [22.0f; 24.0f; 26.0f; 28.0f]; [30.0f; 32.0f; 34.0f; 36.0f]])
        let tSum1 = t.Sum(1)
        let tSum1Correct = Tensor.Create([[15.0f; 18.0f; 21.0f; 24.0f]; [51.0f; 54.0f; 57.0f; 60.0f]])
        let tSum2 = t.Sum(2)
        let tSum2Correct = Tensor.Create([[10.0f; 26.0f; 42.0f]; [58.0f; 74.0f; 90.0f]])

        Assert.AreEqual(tSum0, tSum0Correct)
        Assert.AreEqual(tSum1, tSum1Correct)
        Assert.AreEqual(tSum2, tSum2Correct)
    
    [<Test>]
    member this.TestTensorSumDimKeepDim () =
        let t = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tSum0 = t.Sum(0, keepDim=true)
        let tSum0Correct = Tensor.Create([[[14.0f; 16.0f; 18.0f; 20.0f]; [22.0f; 24.0f; 26.0f; 28.0f]; [30.0f; 32.0f; 34.0f; 36.0f]]])
        let tSum1 = t.Sum(1, keepDim=true)
        let tSum1Correct = Tensor.Create([[[15.0f; 18.0f; 21.0f; 24.0f]]; [[51.0f; 54.0f; 57.0f; 60.0f]]])
        let tSum2 = t.Sum(2, keepDim=true)
        let tSum2Correct = Tensor.Create([[[10.0f]; [26.0f]; [42.0f]]; [[58.0f]; [74.0f]; [90.0f]]])

        Assert.AreEqual(tSum0, tSum0Correct)
        Assert.AreEqual(tSum1, tSum1Correct)
        Assert.AreEqual(tSum2, tSum2Correct)

    [<Test>]
    member this.TestTensorMean () =
        let t = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tMean = t.Mean()
        let tMeanCorrect = Tensor.Create(12.5)

        Assert.AreEqual(tMean, tMeanCorrect)

    [<Test>]
    member this.TestTensorMeanDim () =
        let t = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tMean0 = t.Mean(0)
        let tMean0Correct = Tensor.Create([[7.0f; 8.0f; 9.0f; 10.0f]; [11.0f; 12.0f; 13.0f; 14.0f]; [15.0f; 16.0f; 17.0f; 18.0f]])
        let tMean1 = t.Mean(1)
        let tMean1Correct = Tensor.Create([[5.0f; 6.0f; 7.0f; 8.0f]; [17.0f; 18.0f; 19.0f; 20.0f]])
        let tMean2 = t.Mean(2)
        let tMean2Correct = Tensor.Create([[2.5f; 6.5f; 10.5f]; [14.5f; 18.5f; 22.5f]])

        Assert.AreEqual(tMean0, tMean0Correct)
        Assert.AreEqual(tMean1, tMean1Correct)
        Assert.AreEqual(tMean2, tMean2Correct)


    [<Test>]
    member this.TestTensorStddev () =
        let t = Tensor.Create([[[0.3787;0.7515;0.2252;0.3416];
          [0.6078;0.4742;0.7844;0.0967];
          [0.1416;0.1559;0.6452;0.1417]];
 
         [[0.0848;0.4156;0.5542;0.4166];
          [0.5187;0.0520;0.4763;0.1509];
          [0.4767;0.8096;0.1729;0.6671]]])
        let tStddev = t.Stddev()
        let tStddevCorrect = Tensor.Create(0.2398)

        Assert.True(tStddev.ApproximatelyEqual(tStddevCorrect))

    [<Test>]
    member this.TestTensorStddevDim () =
        let t = Tensor.Create([[[0.3787;0.7515;0.2252;0.3416];
          [0.6078;0.4742;0.7844;0.0967];
          [0.1416;0.1559;0.6452;0.1417]];
 
         [[0.0848;0.4156;0.5542;0.4166];
          [0.5187;0.0520;0.4763;0.1509];
          [0.4767;0.8096;0.1729;0.6671]]])
        let tStddev0 = t.Stddev(0)
        let tStddev0Correct = Tensor.Create([[0.2078; 0.2375; 0.2326; 0.0530];
         [0.0630; 0.2985; 0.2179; 0.0383];
         [0.2370; 0.4623; 0.3339; 0.3715]])
        let tStddev1 = t.Stddev(1)
        let tStddev1Correct = Tensor.Create([[0.2331; 0.2981; 0.2911; 0.1304];
         [0.2393; 0.3789; 0.2014; 0.2581]])
        let tStddev2 = t.Stddev(2)
        let tStddev2Correct = Tensor.Create([[0.2277; 0.2918; 0.2495];
         [0.1996; 0.2328; 0.2753]])

        Assert.True(tStddev0.ApproximatelyEqual(tStddev0Correct))
        Assert.True(tStddev1.ApproximatelyEqual(tStddev1Correct))
        Assert.True(tStddev2.ApproximatelyEqual(tStddev2Correct))

    [<Test>]
    member this.TestTensorTransposeT2 () =
        let t1 = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]])
        let t1Transpose = t1.Transpose()
        let t1TransposeCorrect = Tensor.Create([[1.; 4.]; [2.; 5.]; [3.; 6.]])

        let t2 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t2TransposeTranspose = t2.Transpose().Transpose()
        let t2TransposeTransposeCorrect = t2

        Assert.AreEqual(t1Transpose, t1TransposeCorrect)
        Assert.AreEqual(t2TransposeTranspose, t2TransposeTransposeCorrect)

    [<Test>]
    member this.TestTensorSignT () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.])
        let t1Sign = t1.Sign()
        let t1SignCorrect = Tensor.Create([-1.; -1.; 0.; 1.])

        Assert.AreEqual(t1Sign, t1SignCorrect)

    [<Test>]
    member this.TestTensorFloorT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Floor = t1.Floor()
        let t1FloorCorrect = Tensor.Create([0.; 0.; 0.; 0.; 0.])

        Assert.True(t1Floor.ApproximatelyEqual(t1FloorCorrect))

    [<Test>]
    member this.TestTensorCeilT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Ceil = t1.Ceil()
        let t1CeilCorrect = Tensor.Create([1.; 1.; 1.; 1.; 1.])

        Assert.True(t1Ceil.ApproximatelyEqual(t1CeilCorrect))

    [<Test>]
    member this.TestTensorRoundT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Round = t1.Round()
        let t1RoundCorrect = Tensor.Create([1.; 0.; 0.; 1.; 1.])

        Assert.True(t1Round.ApproximatelyEqual(t1RoundCorrect))

    [<Test>]
    member this.TestTensorAbsT () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.])
        let t1Abs = t1.Abs()
        let t1AbsCorrect = Tensor.Create([1.; 2.; 0.; 3.])

        Assert.AreEqual(t1Abs, t1AbsCorrect)

    [<Test>]
    member this.TestTensorReluT () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.; 10.])
        let t1Relu = t1.Relu()
        let t1ReluCorrect = Tensor.Create([0.; 0.; 0.; 3.; 10.])

        Assert.AreEqual(t1Relu, t1ReluCorrect)

    [<Test>]
    member this.TestTensorLeakyRelu () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.; 10.])
        let t1LeakyRelu = t1.LeakyRelu()
        let t1LeakyReluCorrect = Tensor.Create([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])

        Assert.AreEqual(t1LeakyRelu, t1LeakyReluCorrect)

    [<Test>]
    member this.TestTensorSigmoidT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Sigmoid = t1.Sigmoid()
        let t1SigmoidCorrect = Tensor.Create([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])

        Assert.True(t1Sigmoid.ApproximatelyEqual(t1SigmoidCorrect))

    [<Test>]
    member this.TestTensorExpT () =
        let t1 = Tensor.Create([0.9139; -0.5907;  1.9422; -0.7763; -0.3274])
        let t1Exp = t1.Exp()
        let t1ExpCorrect = Tensor.Create([2.4940; 0.5539; 6.9742; 0.4601; 0.7208])

        Assert.True(t1Exp.ApproximatelyEqual(t1ExpCorrect))

    [<Test>]
    member this.TestTensorLogT () =
        let t1 = Tensor.Create([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
        let t1Log = t1.Log()
        let t1LogCorrect = Tensor.Create([-2.0516; -0.5426; -0.4301; -0.9727; -0.9100])

        Assert.True(t1Log.ApproximatelyEqual(t1LogCorrect))

    [<Test>]
    member this.TestTensorLog10T () =
        let t1 = Tensor.Create([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
        let t1Log10 = t1.Log10()
        let t1Log10Correct = Tensor.Create([-0.8911; -0.2357; -0.1868; -0.4224; -0.3952])

        Assert.True(t1Log10.ApproximatelyEqual(t1Log10Correct))

    [<Test>]
    member this.TestTensorSqrtT () =
        let t1 = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
        let t1Sqrt = t1.Sqrt()
        let t1SqrtCorrect = Tensor.Create([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])

        Assert.True(t1Sqrt.ApproximatelyEqual(t1SqrtCorrect))

    [<Test>]
    member this.TestTensorSinT () =
        let t1 = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
        let t1Sin = t1.Sin()
        let t1SinCorrect = Tensor.Create([-0.9828;  0.9991; -0.3698; -0.7510;  0.9491])

        Assert.True(t1Sin.ApproximatelyEqual(t1SinCorrect))

    [<Test>]
    member this.TestTensorCosT () =
        let t1 = Tensor.Create([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
        let t1Cos = t1.Cos()
        let t1CosCorrect = Tensor.Create([-0.1849;  0.0418; -0.9291;  0.6603;  0.3150])

        Assert.True(t1Cos.ApproximatelyEqual(t1CosCorrect))

    [<Test>]
    member this.TestTensorTanT () =
        let t1 = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Tan = t1.Tan()
        let t1TanCorrect = Tensor.Create([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])

        Assert.True(t1Tan.ApproximatelyEqual(t1TanCorrect))

    [<Test>]
    member this.TestTensorSinhT () =
        let t1 = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Sinh = t1.Sinh()
        let t1SinhCorrect = Tensor.Create([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])

        Assert.True(t1Sinh.ApproximatelyEqual(t1SinhCorrect))

    [<Test>]
    member this.TestTensorCoshT () =
        let t1 = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Cosh = t1.Cosh()
        let t1CoshCorrect = Tensor.Create([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])

        Assert.True(t1Cosh.ApproximatelyEqual(t1CoshCorrect))

    [<Test>]
    member this.TestTensorTanhT () =
        let t1 = Tensor.Create([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Tanh = t1.Tanh()
        let t1TanhCorrect = Tensor.Create([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])

        Assert.True(t1Tanh.ApproximatelyEqual(t1TanhCorrect))

    [<Test>]
    member this.TestTensorAsinT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Asin = t1.Asin()
        let t1AsinCorrect = Tensor.Create([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])

        Assert.True(t1Asin.ApproximatelyEqual(t1AsinCorrect))

    [<Test>]
    member this.TestTensorAcosT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Acos = t1.Acos()
        let t1AcosCorrect = Tensor.Create([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])

        Assert.True(t1Acos.ApproximatelyEqual(t1AcosCorrect))

    [<Test>]
    member this.TestTensorAtanT () =
        let t1 = Tensor.Create([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Atan = t1.Atan()
        let t1AtanCorrect = Tensor.Create([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])

        Assert.True(t1Atan.ApproximatelyEqual(t1AtanCorrect))

    [<Test>]
    member this.TestTensorSlice () =
        let t1 = Tensor.Create([1.;2.])
        let t1s1 = t1.[0]
        let t1s2 = t1.[*]
        let t1s1Correct = Tensor.Create(1.)
        let t1s2Correct = Tensor.Create([1.;2.])

        let t2 = Tensor.Create([[1.;2.];[3.;4.]])
        let t2s1 = t2.[0]
        let t2s2 = t2.[*]
        let t2s3 = t2.[0,0]
        let t2s4 = t2.[0,*]
        let t2s5 = t2.[*,0]
        let t2s6 = t2.[*,*]
        let t2s1Correct = Tensor.Create([1.;2.])
        let t2s2Correct = Tensor.Create([[1.;2.];[3.;4.]])
        let t2s3Correct = Tensor.Create(1.)
        let t2s4Correct = Tensor.Create([1.;2.])
        let t2s5Correct = Tensor.Create([1.;3.])
        let t2s6Correct = Tensor.Create([[1.;2.];[3.;4.]])

        let t2b = Tensor.Create([[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]])
        let t2bs1 = t2b.[1..,2..]
        let t2bs1Correct = Tensor.Create([[7.;8.];[11.;12.]])
        let t2bs2 = t2b.[1..2,2..3]
        let t2bs2Correct = Tensor.Create([[7.;8.];[11.;12.]])

        let t3 = Tensor.Create([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
        let t3s1  = t3.[0]
        let t3s2  = t3.[*]
        let t3s3  = t3.[0,0]
        let t3s4  = t3.[0,*]
        let t3s5  = t3.[*,0]
        let t3s6  = t3.[*,*]
        let t3s7  = t3.[0,0,0]
        let t3s8  = t3.[0,0,*]
        let t3s9  = t3.[0,*,0]
        let t3s10 = t3.[0,*,*]
        let t3s11 = t3.[*,0,0]
        let t3s12 = t3.[*,0,*]
        let t3s13 = t3.[*,*,0]
        let t3s14 = t3.[*,*,*]
        let t3s1Correct  = Tensor.Create([[1.;2.];[3.;4.]])
        let t3s2Correct  = Tensor.Create([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
        let t3s3Correct  = Tensor.Create([1.;2.])
        let t3s4Correct  = Tensor.Create([[1.;2.];[3.;4.]])
        let t3s5Correct  = Tensor.Create([[1.;2.];[5.;6.]])
        let t3s6Correct  = Tensor.Create([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
        let t3s7Correct  = Tensor.Create(1.)
        let t3s8Correct  = Tensor.Create([1.;2.])
        let t3s9Correct  = Tensor.Create([1.;3.])
        let t3s10Correct = Tensor.Create([[1.;2.];[3.;4.]])
        let t3s11Correct = Tensor.Create([1.;5.])
        let t3s12Correct = Tensor.Create([[1.;2.];[5.;6.]])
        let t3s13Correct = Tensor.Create([[1.;3.];[5.;7.]])
        let t3s14Correct = Tensor.Create([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])

        Assert.AreEqual(t1s1, t1s1Correct)
        Assert.AreEqual(t1s2, t1s2Correct)

        Assert.AreEqual(t2s1, t2s1Correct)
        Assert.AreEqual(t2s2, t2s2Correct)
        Assert.AreEqual(t2s3, t2s3Correct)
        Assert.AreEqual(t2s4, t2s4Correct)
        Assert.AreEqual(t2s5, t2s5Correct)
        Assert.AreEqual(t2s6, t2s6Correct)

        Assert.AreEqual(t2bs1, t2bs1Correct)
        Assert.AreEqual(t2bs2, t2bs2Correct)

        Assert.AreEqual(t3s1, t3s1Correct)
        Assert.AreEqual(t3s2, t3s2Correct)
        Assert.AreEqual(t3s3, t3s3Correct)
        Assert.AreEqual(t3s4, t3s4Correct)
        Assert.AreEqual(t3s5, t3s5Correct)
        Assert.AreEqual(t3s6, t3s6Correct)
        Assert.AreEqual(t3s7, t3s7Correct)
        Assert.AreEqual(t3s8, t3s8Correct)
        Assert.AreEqual(t3s9, t3s9Correct)
        Assert.AreEqual(t3s10, t3s10Correct)
        Assert.AreEqual(t3s11, t3s11Correct)
        Assert.AreEqual(t3s12, t3s12Correct)
        Assert.AreEqual(t3s13, t3s13Correct)
        Assert.AreEqual(t3s14, t3s14Correct)

    [<Test>]
    member this.TestTensorAddTTSlice () =
        let t1 = Tensor.Create([[-0.2754;  0.0172;  0.7105];
            [-0.1890;  1.7664;  0.5377];
            [-0.5313; -2.2530; -0.6235];
            [ 0.6776;  1.5844; -0.5686]])
        let t2 = Tensor.Create([[-111.8892;   -7.0328];
            [  18.7557;  -86.2308]])
        let t3 = Tensor.AddSlice(t1, [0;1], t2)
        let t3Correct = Tensor.Create([[  -0.2754; -111.8720;   -6.3222];
            [  -0.1890;   20.5221;  -85.6932];
            [  -0.5313;   -2.2530;   -0.6235];
            [   0.6776;    1.5844;   -0.5686]])

        Assert.True(t3.ApproximatelyEqual(t3Correct))

    [<Test>]
    member this.TestTensorSqueezeT () =
        let t1 = Tensor.Create([[[1.; 2.]]; [[3.;4.]]])
        let t1Squeeze = t1.Squeeze()
        let t1SqueezeCorrect = Tensor.Create([[1.;2.];[3.;4.]])

        Assert.True(t1Squeeze.ApproximatelyEqual(t1SqueezeCorrect))

    [<Test>]
    member this.TestTensorUnsqueezeT () =
        let t1 = Tensor.Create([[1.;2.];[3.;4.]])
        let t1Unsqueeze = t1.Unsqueeze(1)
        let t1UnsqueezeCorrect = Tensor.Create([[[1.;2.]]; [[3.;4.]]])

        Assert.True(t1Unsqueeze.ApproximatelyEqual(t1UnsqueezeCorrect))

    [<Test>]
    member this.TestTensorFlipT () =
        let t1 = Tensor.Create([[1.;2.];[3.;4.]])
        let t2 = t1.Flip([|0|])
        let t2Correct = Tensor.Create([[3.;4.]; [1.;2.]])
        let t3 = t1.Flip([|1|])
        let t3Correct = Tensor.Create([[2.;1.]; [4.;3.]])
        let t4 = t1.Flip([|0; 1|])
        let t4Correct = Tensor.Create([[4.;3.]; [2.;1.]])
        let t5 = t1.Flip([|0; 1|]).Flip([|0; 1|])
        let t5Correct = Tensor.Create([[1.;2.]; [3.;4.]])

        Assert.AreEqual(t2, t2Correct)
        Assert.AreEqual(t3, t3Correct)
        Assert.AreEqual(t4, t4Correct)
        Assert.AreEqual(t5, t5Correct)

    [<Test>]
    member this.TestTensorView () =
        let t = Tensor.Random([10;10])
        let t1Shape = t.View(-1).Shape
        let t1ShapeCorrect = [|100|]
        let t2Shape = t.View([-1;50]).Shape
        let t2ShapeCorrect = [|2;50|]
        let t3Shape = t.View([2;-1;50]).Shape
        let t3ShapeCorrect = [|2;1;50|]
        let t4Shape = t.View([2;-1;10]).Shape
        let t4ShapeCorrect = [|2;5;10|]
        
        Assert.AreEqual(t1Shape, t1ShapeCorrect)
        Assert.AreEqual(t2Shape, t2ShapeCorrect)
        Assert.AreEqual(t3Shape, t3ShapeCorrect)
        Assert.AreEqual(t4Shape, t4ShapeCorrect)


    [<Test>]
    member this.TestTensorMax () =
        let t1 = Tensor.Create([4.;1.;20.;3.])
        let t1Max = t1.Max()
        let t1MaxCorrect = Tensor.Create(20.)

        let t2 = Tensor.Create([[1.;4.];[2.;3.]])
        let t2Max = t2.Max()
        let t2MaxCorrect = Tensor.Create(4.)

        let t3 = Tensor.Create([[[ 7.6884; 65.9125;  4.0114];
             [46.7944; 61.5331; 40.1627];
             [48.3240;  4.9910; 50.1571]];

            [[13.4777; 65.7656; 36.8161];
             [47.8268; 42.2229;  5.6115];
             [43.4779; 77.8675; 95.7660]];

            [[59.8422; 47.1146; 36.7614];
             [71.6328; 18.5912; 27.7328];
             [49.9120; 60.3023; 53.0838]]])
        let t3Max = t3.Max()
        let t3MaxCorrect = Tensor.Create(95.7660)
        
        let t4 = Tensor.Create([[[[8.8978; 8.0936];
              [4.8087; 1.0921];
              [8.5664; 3.7814]];

             [[2.3581; 3.7361];
              [1.0436; 6.0353];
              [7.7843; 8.7153]];

             [[3.9188; 6.7906];
              [9.1242; 4.8711];
              [1.7870; 9.7456]];

             [[5.7825; 8.0450];
              [2.7801; 1.0877];
              [3.4042; 5.1911]]];

            [[[0.5370; 7.1115];
              [5.4971; 2.3567];
              [0.9318; 8.6992]];

             [[3.3796; 8.7833];
              [5.8722; 5.9881];
              [0.7646; 7.3685]];

             [[7.5344; 9.6162];
              [2.6404; 4.3938];
              [3.1335; 7.6783]];

             [[5.0444; 0.5447];
              [6.2945; 5.9047];
              [8.0867; 3.1606]]]])
        let t4Max = t4.Max()
        let t4MaxCorrect = Tensor.Create(9.7456)

        Assert.AreEqual(t1Max, t1MaxCorrect)
        Assert.AreEqual(t2Max, t2MaxCorrect)
        Assert.AreEqual(t3Max, t3MaxCorrect)
        Assert.AreEqual(t4Max, t4MaxCorrect)


    [<Test>]
    member this.TestTensorMin () =
        let t1 = Tensor.Create([4.;1.;20.;3.])
        let t1Min = t1.Min()
        let t1MinCorrect = Tensor.Create(1.)

        let t2 = Tensor.Create([[1.;4.];[2.;3.]])
        let t2Min = t2.Min()
        let t2MinCorrect = Tensor.Create(1.)

        let t3 = Tensor.Create([[[ 7.6884; 65.9125;  4.0114];
             [46.7944; 61.5331; 40.1627];
             [48.3240;  4.9910; 50.1571]];

            [[13.4777; 65.7656; 36.8161];
             [47.8268; 42.2229;  5.6115];
             [43.4779; 77.8675; 95.7660]];

            [[59.8422; 47.1146; 36.7614];
             [71.6328; 18.5912; 27.7328];
             [49.9120; 60.3023; 53.0838]]])
        let t3Min = t3.Min()
        let t3MinCorrect = Tensor.Create(4.0114)
       
        let t4 = Tensor.Create([[[[8.8978; 8.0936];
              [4.8087; 1.0921];
              [8.5664; 3.7814]];

             [[2.3581; 3.7361];
              [1.0436; 6.0353];
              [7.7843; 8.7153]];

             [[3.9188; 6.7906];
              [9.1242; 4.8711];
              [1.7870; 9.7456]];

             [[5.7825; 8.0450];
              [2.7801; 1.0877];
              [3.4042; 5.1911]]];

            [[[0.5370; 7.1115];
              [5.4971; 2.3567];
              [0.9318; 8.6992]];

             [[3.3796; 8.7833];
              [5.8722; 5.9881];
              [0.7646; 7.3685]];

             [[7.5344; 9.6162];
              [2.6404; 4.3938];
              [3.1335; 7.6783]];

             [[5.0444; 0.5447];
              [6.2945; 5.9047];
              [8.0867; 3.1606]]]])
        let t4Min = t4.Min()
        let t4MinCorrect = Tensor.Create(0.5370)

        Assert.AreEqual(t1Min, t1MinCorrect)
        Assert.AreEqual(t2Min, t2MinCorrect)
        Assert.AreEqual(t3Min, t3MinCorrect)
        Assert.AreEqual(t4Min, t4MinCorrect)

    [<Test>]
    member this.TestTensorMaxBinary () =
        let t1 = Tensor.Create([[-4.9385; 12.6206; 10.1783];
            [-2.9624; 17.6992;  2.2506];
            [-2.3536;  8.0772; 13.5639]])
        let t2 = Tensor.Create([[  0.7027;  22.3251; -11.4533];
            [  3.6887;   4.3355;   3.3767];
            [  0.1203;  -5.4088;   1.5658]])
        let t3 = Tensor.Max(t1, t2)
        let t3Correct = Tensor.Create([[ 0.7027; 22.3251; 10.1783];
            [ 3.6887; 17.6992;  3.3767];
            [ 0.1203;  8.0772; 13.5639]])

        Assert.True(t3.ApproximatelyEqual(t3Correct))

    [<Test>]
    member this.TestTensorMinBinary () =
        let t1 = Tensor.Create([[-4.9385; 12.6206; 10.1783];
            [-2.9624; 17.6992;  2.2506];
            [-2.3536;  8.0772; 13.5639]])
        let t2 = Tensor.Create([[  0.7027;  22.3251; -11.4533];
            [  3.6887;   4.3355;   3.3767];
            [  0.1203;  -5.4088;   1.5658]])
        let t3 = Tensor.Min(t1, t2)
        let t3Correct = Tensor.Create([[ -4.9385;  12.6206; -11.4533];
            [ -2.9624;   4.3355;   2.2506];
            [ -2.3536;  -5.4088;   1.5658]])

        Assert.True(t3.ApproximatelyEqual(t3Correct))

    [<Test>]
    member this.TestTensorSoftmax () =
        let t1 = Tensor.Create([2.7291; 0.0607; 0.8290])
        let t1Softmax0 = t1.Softmax(0)
        let t1Softmax0Correct = Tensor.Create([0.8204; 0.0569; 0.1227])

        let t2 = Tensor.Create([[1.3335; 1.6616; 2.4874; 6.1722];
            [3.3478; 9.3019; 1.0844; 8.9874];
            [8.6300; 1.8842; 9.1387; 9.1321]])
        let t2Softmax0 = t2.Softmax(0)
        let t2Softmax0Correct = Tensor.Create([[6.7403e-04; 4.8014e-04; 1.2904e-03; 2.7033e-02];
            [5.0519e-03; 9.9892e-01; 3.1723e-04; 4.5134e-01];
            [9.9427e-01; 5.9987e-04; 9.9839e-01; 5.2163e-01]])
        let t2Softmax1 = t2.Softmax(1)
        let t2Softmax1Correct = Tensor.Create([[7.5836e-03; 1.0528e-02; 2.4044e-02; 9.5784e-01];
            [1.4974e-03; 5.7703e-01; 1.5573e-04; 4.2131e-01];
            [2.3167e-01; 2.7240e-04; 3.8528e-01; 3.8277e-01]])

        let t3 = Tensor.Create([[[3.0897; 2.0902];
             [2.4055; 1.2437];
             [2.1253; 8.7802];
             [4.3856; 3.4456]];

            [[8.6233; 6.9789];
             [4.9583; 9.9497];
             [2.6964; 1.6048];
             [2.1182; 2.1071]];

            [[8.1097; 6.9804];
             [8.1223; 6.3030];
             [0.1873; 8.7840];
             [9.3609; 0.6493]]])
             
        let t3Softmax0 = t3.Softmax(0)
        let t3Softmax0Correct = Tensor.Create([[[2.4662e-03; 3.7486e-03];
             [3.1467e-03; 1.6136e-04];
             [3.4316e-01; 4.9885e-01];
             [6.8542e-03; 7.5571e-01]];

            [[6.2411e-01; 4.9776e-01];
             [4.0415e-02; 9.7443e-01];
             [6.0743e-01; 3.8170e-04];
             [7.0995e-04; 1.9817e-01]];

            [[3.7342e-01; 4.9849e-01];
             [9.5644e-01; 2.5410e-02];
             [4.9412e-02; 5.0077e-01];
             [9.9244e-01; 4.6122e-02]]])
        let t3Softmax1 = t3.Softmax(1)
        let t3Softmax1Correct = Tensor.Create([[[1.8050e-01; 1.2351e-03];
             [9.1058e-02; 5.2978e-04];
             [6.8813e-02; 9.9344e-01];
             [6.5963e-01; 4.7904e-03]];

            [[9.7109e-01; 4.8732e-02];
             [2.4864e-02; 9.5067e-01];
             [2.5896e-03; 2.2587e-04];
             [1.4526e-03; 3.7327e-04]];

            [[1.8156e-01; 1.3190e-01];
             [1.8387e-01; 6.6997e-02];
             [6.5824e-05; 8.0087e-01];
             [6.3451e-01; 2.3479e-04]]])
        let t3Softmax2 = t3.Softmax(2)
        let t3Softmax2Correct = Tensor.Create([[[7.3096e-01; 2.6904e-01];
             [7.6165e-01; 2.3835e-01];
             [1.2861e-03; 9.9871e-01];
             [7.1910e-01; 2.8090e-01]];

            [[8.3814e-01; 1.6186e-01];
             [6.7502e-03; 9.9325e-01];
             [7.4868e-01; 2.5132e-01];
             [5.0278e-01; 4.9722e-01]];

            [[7.5571e-01; 2.4429e-01];
             [8.6049e-01; 1.3951e-01];
             [1.8468e-04; 9.9982e-01];
             [9.9984e-01; 1.6463e-04]]])

        Assert.True(t1Softmax0.ApproximatelyEqual(t1Softmax0Correct, 0.001))
        Assert.True(t2Softmax0.ApproximatelyEqual(t2Softmax0Correct, 0.001))
        Assert.True(t2Softmax1.ApproximatelyEqual(t2Softmax1Correct, 0.001))
        Assert.True(t3Softmax0.ApproximatelyEqual(t3Softmax0Correct, 0.001))
        Assert.True(t3Softmax1.ApproximatelyEqual(t3Softmax1Correct, 0.001))
        Assert.True(t3Softmax2.ApproximatelyEqual(t3Softmax2Correct, 0.001))

    [<Test>]
    member this.TestTensorDepth () =
        let t0 = Tensor.Create([1.;2.])
        let t0Depth = t0.Depth
        let t0DepthCorrect = 0
        let t1 = Tensor.Create([1.;2.]).ReverseDiff()
        let t1Depth = t1.Depth
        let t1DepthCorrect = 1
        let t2 = Tensor.Create([1.;2.]).ReverseDiff().ReverseDiff()
        let t2Depth = t2.Depth
        let t2DepthCorrect = 2
        let t3 = Tensor.Create([1.;2.]).ReverseDiff().ReverseDiff().ForwardDiff(Tensor.Create([1.; 1.]))
        let t3Depth = t3.Depth
        let t3DepthCorrect = 3

        Assert.AreEqual(t0Depth, t0DepthCorrect)
        Assert.AreEqual(t1Depth, t1DepthCorrect)
        Assert.AreEqual(t2Depth, t2DepthCorrect)
        Assert.AreEqual(t3Depth, t3DepthCorrect)
