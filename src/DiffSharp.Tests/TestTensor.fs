namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Backend

[<TestFixture>]
type TestTensor () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestTensorCreate0 () =
        let t0 = Tensor.Create(1.)
        let t0Shape = t0.Shape
        let t0Dim = t0.Dim
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.AreEqual(t0DimCorrect, t0Dim)
        Assert.AreEqual(t0ShapeCorrect, t0Shape)

    [<Test>]
    member this.TestTensorCreate1 () =
        // create from double list
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        Assert.AreEqual(t1ShapeCorrect, t1.Shape)
        Assert.AreEqual(t1DimCorrect, t1.Dim)

        // create from double[]
        let t1Array = Tensor.Create([| 1.; 2.; 3. |])

        Assert.AreEqual(t1ShapeCorrect, t1Array.Shape)
        Assert.AreEqual(t1DimCorrect, t1Array.Dim)

        // create from seq<double>
        let t1Seq = Tensor.Create(seq { 1.; 2.; 3. })

        Assert.AreEqual(t1ShapeCorrect, t1Seq.Shape)
        Assert.AreEqual(t1DimCorrect, t1Seq.Dim)

    [<Test>]
    member this.TestTensorCreate2 () =
        let t2Values = [[1.; 2.; 3.]; [4.; 5.; 6.]]
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2
        // let t2DTypeCorrect = DType.Float32
        let t2ValuesCorrect = array2D (List.map (List.map float32) t2Values)

        // create from double list list
        let t2 = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]])
        Assert.AreEqual(t2ShapeCorrect, t2.Shape)
        Assert.AreEqual(t2DimCorrect, t2.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2.ToArray())

        // create from double array list
        let t2ArrayList = Tensor.Create([[|1.; 2.; 3.|]; [|4.; 5.; 6.|]])
        Assert.AreEqual(t2ShapeCorrect, t2ArrayList.Shape)
        Assert.AreEqual(t2DimCorrect, t2ArrayList.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArrayList.ToArray())

        // create from double list array
        let t2ListArray = Tensor.Create([| [1.; 2.; 3.]; [4.; 5.; 6.] |])
        Assert.AreEqual(t2ShapeCorrect, t2ListArray.Shape)
        Assert.AreEqual(t2DimCorrect, t2ListArray.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2ListArray.ToArray())

        // create from double[][]
        let t2ArrayArray = Tensor.Create([| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.AreEqual(t2ShapeCorrect, t2ArrayArray.Shape)
        Assert.AreEqual(t2DimCorrect, t2ArrayArray.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArrayArray.ToArray())

        // create from double[,]
        let t2Array2D = Tensor.Create(array2D [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.AreEqual(t2ShapeCorrect, t2Array2D.Shape)
        Assert.AreEqual(t2DimCorrect, t2Array2D.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2Array2D.ToArray())

        // create from seq<double[]>
        let t2ArraySeq = Tensor.Create(seq { yield [| 1.; 2.; 3. |]; yield [| 4.; 5.; 6.|] })
        Assert.AreEqual(t2ShapeCorrect, t2ArraySeq.Shape)
        Assert.AreEqual(t2DimCorrect, t2ArraySeq.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArraySeq.ToArray())

        // create from seq<seq<double>>
        let t2SeqSeq = Tensor.Create(seq { seq { 1.; 2.; 3. }; seq { 4.; 5.; 6.} })
        Assert.AreEqual(t2ShapeCorrect, t2SeqSeq.Shape)
        Assert.AreEqual(t2DimCorrect, t2SeqSeq.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2SeqSeq.ToArray())

        // create from (double * double * double) list list
        let t2TupleListList = Tensor.Create([ [ 1., 2., 3. ]; [ 4., 5., 6. ] ])
        Assert.AreEqual(t2ShapeCorrect, t2TupleListList.Shape)
        Assert.AreEqual(t2DimCorrect, t2TupleListList.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleListList.ToArray())

        // create from ((double * double * double) list * (double * double * double) list) list
        let t2TupleListTupleList = Tensor.Create([ [ 1., 2., 3. ], [ 4., 5., 6. ] ])
        Assert.AreEqual(t2ShapeCorrect, t2TupleListTupleList.Shape)
        Assert.AreEqual(t2DimCorrect, t2TupleListTupleList.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleListTupleList.ToArray())

        // create from (double * double * double)[]
        let t2TupleArray = Tensor.Create([| [ 1., 2., 3. ]; [ 4., 5., 6. ] |])
        Assert.AreEqual(t2ShapeCorrect, t2TupleArray.Shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArray.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArray.ToArray())

        // create from ((double * double * double) [] * (double * double * double) []) []
        let t2TupleArrayTupleArray = Tensor.Create([| [| 1., 2., 3. |], [| 4., 5., 6. |] |])
        Assert.AreEqual(t2ShapeCorrect, t2TupleArrayTupleArray.Shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArrayTupleArray.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArrayTupleArray.ToArray())
        Assert.AreEqual(t2ValuesCorrect, t2TupleArrayTupleArray.ToArray())

        // create from (double * double * double)seq
        let t2TupleArray = Tensor.Create(seq { [ 1., 2., 3. ]; [ 4., 5., 6. ] })
        Assert.AreEqual(t2ShapeCorrect, t2TupleArray.Shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArray.Dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArray.ToArray())

        let t2TupleOfList = Tensor.Create [[2.], [3.], [4.]]
        Assert.AreEqual([| 3; 1 |], t2TupleOfList.Shape)
        Assert.AreEqual(array2D [ [2]; [3]; [4] ], t2TupleOfList.ToArray())

    [<Test>]
    member this.TestTensorCreate3 () =
        let t3Values = [[[1.; 2.; 3.]; [4.; 5.; 6.]]]
        let t3 = Tensor.Create(t3Values)
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3
        let t3ValuesCorrect = Util.array3D (List.map (List.map (List.map float32)) t3Values)

        Assert.AreEqual(t3ShapeCorrect, t3.Shape)
        Assert.AreEqual(t3DimCorrect, t3.Dim)
        Assert.AreEqual(t3ValuesCorrect, t3.ToArray())

    [<Test>]
    member this.TestTensorCreate4 () =
        let t4Values = [[[[1.; 2.]]]]
        let t4 = Tensor.Create(t4Values)
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4
        let t4ValuesCorrect = Util.array4D (List.map (List.map (List.map (List.map float32))) t4Values)

        Assert.AreEqual(t4ShapeCorrect, t4.Shape)
        Assert.AreEqual(t4DimCorrect, t4.Dim)
        Assert.AreEqual(t4ValuesCorrect, t4.ToArray())

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
        let t0StringCorrect = "Tensor 2.000000"
        let t1StringCorrect = "Tensor [[2.000000], \n [2.000000]]"
        let t2StringCorrect = "Tensor [[[2.000000, 2.000000]]]"
        let t3StringCorrect = "Tensor [[1.000000, 2.000000], \n [3.000000, 4.000000]]"
        let t4StringCorrect = "Tensor [[[[1.000000]]]]"
        Assert.AreEqual(t0StringCorrect, t0String)
        Assert.AreEqual(t1StringCorrect, t1String)
        Assert.AreEqual(t2StringCorrect, t2String)
        Assert.AreEqual(t3StringCorrect, t3String)
        Assert.AreEqual(t4StringCorrect, t4String)

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

        Assert.AreEqual(t1t2LessCorrect, t1t2Less)
        Assert.AreEqual(t1t2EqualCorrect, t1t2Equal)
        Assert.AreEqual(t2t3EqualCorrect, t2t3Equal)

    [<Test>]
    member this.TestTensorLtTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Lt = t1.Lt(t2)
        let t1t2LtCorrect = Tensor.Create([0.; 1.; 1.; 0.])

        Assert.AreEqual(t1t2LtCorrect, t1t2Lt)

    [<Test>]
    member this.TestTensorLeTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Le = t1.Le(t2)
        let t1t2LeCorrect = Tensor.Create([1.; 1.; 1.; 0.])

        Assert.AreEqual(t1t2LeCorrect, t1t2Le)

    [<Test>]
    member this.TestTensorGtTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Gt = t1.Gt(t2)
        let t1t2GtCorrect = Tensor.Create([0.; 0.; 0.; 1.])

        Assert.AreEqual(t1t2GtCorrect, t1t2Gt)

    [<Test>]
    member this.TestTensorGeTT () =
        let t1 = Tensor.Create([1.; 2.; 3.; 5.])
        let t2 = Tensor.Create([1.; 3.; 5.; 4.])
        let t1t2Ge = t1.Ge(t2)
        let t1t2GeCorrect = Tensor.Create([1.; 0.; 0.; 1.])

        Assert.AreEqual(t1t2GeCorrect, t1t2Ge)

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

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)
        Assert.AreEqual(t5Correct, t5)

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

        Assert.AreEqual(t0Correct, t0)
        Assert.AreEqual(t1Correct, t1)

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

        Assert.AreEqual(t0Correct, t0)
        Assert.AreEqual(t1Correct, t1)

    [<Test>]
    member this.TestTensorAddT2T1 () =
        let t1 = Tensor.Create([[1.; 2.]; [3.; 4.]]) + Tensor.Create([5.; 6.])
        let t1Correct = Tensor.Create([[6.; 8.]; [8.; 10.]])

        Assert.AreEqual(t1Correct, t1)

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

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)

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

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)

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

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)

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

        let t3s2p1 = Tensor.Conv1D(t1, t2, stride=2, padding=1)
        let t3s2p1Correct = Tensor.Create([[[1.4392; 2.0732; 2.1177];
                                             [1.4345; 1.7078; 2.1474]];

                                            [[2.4208; 2.9651; 1.2960];
                                             [1.5544; 3.1496; 1.8567]];

                                            [[1.2965; 2.9038; 1.7408];
                                             [1.3549; 2.9444; 1.2120]]])

        let t3s3p2 = Tensor.Conv1D(t1, t2, stride=3, padding=2)
        let t3s3p2Correct = Tensor.Create([[[0.6333; 2.0732; 1.0258];
                                             [0.6539; 1.7078; 1.2585]];

                                            [[0.5982; 2.9651; 1.0620];
                                             [0.5157; 3.1496; 1.3182]];

                                            [[0.3165; 2.9038; 0.5275];
                                             [0.3861; 2.9444; 0.7428]]])
        
        let t3d2 = Tensor.Conv1D(t1, t2, dilation=2)
        let t3d2Correct = Tensor.Create([[[2.8030];
                                         [2.4735]];

                                        [[2.9226];
                                         [3.1868]];

                                        [[2.8469];
                                         [2.4790]]])

        let t3p2d3 = Tensor.Conv1D(t1, t2, padding=2, dilation=3)
        let t3p2d3Correct = Tensor.Create([[[2.1121; 0.8484; 2.2709];
                                             [1.6692; 0.5406; 1.8381]];

                                            [[2.5078; 1.2137; 0.9173];
                                             [2.2395; 1.1805; 1.1954]];

                                            [[1.5215; 1.3946; 2.1327];
                                             [1.0732; 1.3014; 2.0696]]])

        let t3s3p6d3 = Tensor.Conv1D(t1, t2, stride=3, padding=6, dilation=3)
        let t3s3p6d3Correct = Tensor.Create([[[0.6333; 1.5018; 2.2709; 1.0580];
                                             [0.6539; 1.5130; 1.8381; 1.0479]];

                                            [[0.5982; 1.7459; 0.9173; 0.2709];
                                             [0.5157; 0.8537; 1.1954; 0.7027]];

                                            [[0.3165; 1.4118; 2.1327; 1.1949];
                                             [0.3861; 1.5697; 2.0696; 0.8520]]])

        let t3b1 = Tensor.Conv1D(t1.[0].Unsqueeze(0) , t2)
        let t3b1Correct = t3Correct.[0].Unsqueeze(0)
        let t3b1s2 = Tensor.Conv1D(t1.[0].Unsqueeze(0) , t2, stride = 2)
        let t3b1s2Correct = t3s2Correct.[0].Unsqueeze(0)

        Assert.True(t3.ApproximatelyEqual(t3Correct))
        Assert.True(t3p1.ApproximatelyEqual(t3p1Correct))
        Assert.True(t3p2.ApproximatelyEqual(t3p2Correct))
        Assert.True(t3s2.ApproximatelyEqual(t3s2Correct))
        Assert.True(t3s3.ApproximatelyEqual(t3s3Correct))
        Assert.True(t3s2p1.ApproximatelyEqual(t3s2p1Correct))
        Assert.True(t3s3p2.ApproximatelyEqual(t3s3p2Correct))
        Assert.True(t3d2.ApproximatelyEqual(t3d2Correct))
        Assert.True(t3p2d3.ApproximatelyEqual(t3p2d3Correct))
        Assert.True(t3s3p6d3.ApproximatelyEqual(t3s3p6d3Correct))
        Assert.True(t3b1.ApproximatelyEqual(t3b1Correct))
        Assert.True(t3b1s2.ApproximatelyEqual(t3b1s2Correct))

    [<Test>]
    member this.TestTensorConv2D () =
        let t1 = Tensor.Create([[[[ 10.7072,  -5.0993,   3.6884,   2.0982],
                                  [ -6.4356,   0.6351,  -2.3156,  -1.3384],
                                  [ -5.1846,   0.6805, -14.1961,   0.8657],
                                  [ -8.8655,  -7.1694,  -3.4903,  -2.9479]],

                                 [[  2.5630,  -2.2935,  -0.8665,   6.7999],
                                  [  1.8098,   3.2082,   2.3160,  -4.7734],
                                  [ 14.7205,   0.9631,   8.1039,   6.7437],
                                  [  3.7847,  -5.9792,  -2.7371,  -7.8548]]],


                                [[[  3.5499,   0.9546,  -7.5715,   2.8211],
                                  [ -1.2659,   5.2366,  -7.2322,  -5.8877],
                                  [ -2.8041,   2.1746,   2.2397,   0.1242],
                                  [  1.8172,  -0.3989,  -0.2394,   7.1078]],

                                 [[ -3.7765,   2.1584,   6.8627,  -4.1471],
                                  [  4.6748,   7.9756,  -6.0065,   2.0826],
                                  [  5.1038,  -5.5801,  -4.4420,  -2.9498],
                                  [  0.1037,   4.6578,   3.0760,  -4.9566]]]])
        let t2 = Tensor.Create([[[[-5.6745, -1.9422,  4.1369],
                                  [ 4.4623,  4.8385,  0.8345],
                                  [ 1.3015,  0.0708,  3.8132]],

                                 [[ 0.9448, -1.9028, -8.0195],
                                  [-5.3200,  0.4264, -1.2142],
                                  [ 1.4442, -7.3623, 14.5340]]],


                                [[[-3.3486, -3.2725, -3.4595],
                                  [-5.0818, -0.5769, -3.5363],
                                  [ 3.1498,  0.6293, -1.2527]],

                                 [[ 3.2029,  3.9409, 12.6924],
                                  [ 4.1056, -3.2890,  2.4071],
                                  [ 4.2373, -1.8852,  4.4640]]],


                                [[[ 4.0582, -4.6075,  6.2574],
                                  [-0.9867,  3.4303, -1.9686],
                                  [-5.0618,  5.0045, -2.0878]],

                                 [[ 1.0605, -3.2697, -1.9856],
                                  [-6.5763, -6.3535,  7.2228],
                                  [15.1009,  4.9045,  5.1197]]]])

        let t3 = Tensor.Conv2D(t1, t2)
        let t3Correct = Tensor.Create([[[[  10.6089;   -1.4459];
                                          [-132.3437; -165.9882]];

                                         [[  97.8425;   81.2322];
                                          [ 215.2763; -112.2244]];

                                         [[ 427.2891; -101.3674];
                                          [ -35.6012; -168.9572]]];


                                        [[[-127.6157;  -35.6266];
                                          [  -7.7668;  -47.1349]];

                                         [[ 104.2333;   28.7020];
                                          [  27.1404;    8.1246]];

                                         [[-106.0468;  -94.3428];
                                          [ -78.6259;  136.6283]]]])

        let t3p1 = Tensor.Conv2D(t1, t2, padding=1)
        let t3p1Correct = Tensor.Create([[[[  86.6988;    8.1164;  -85.8172;   69.5001];
                                          [-154.2592;   10.6089;   -1.4459; -126.2889];
                                          [-176.1860; -132.3437; -165.9882;  -23.2585];
                                          [ -62.8550; -180.0650;  -52.4599;   55.0733]];

                                         [[   3.9697;  -53.5450;   16.3075;  -35.2008];
                                          [ -60.7372;   97.8425;   81.2322;   20.0075];
                                          [  -9.2216;  215.2763; -112.2244;   73.8351];
                                          [  88.4748;  308.1942;  176.2158;  131.2712]];

                                         [[   5.6857;   51.6497;  106.6138;  -17.3603];
                                          [ -46.9604;  427.2891; -101.3674;  226.5788];
                                          [-125.8047;  -35.6012; -168.9572; -141.2721];
                                          [-105.4274; -132.2796;   35.6026;  -13.8173]]];


                                        [[[ 115.1200; -141.3008;   36.3188;  -92.2498];
                                          [-133.0979; -127.6157;  -35.6266;   42.1693];
                                          [  14.0058;   -7.7668;  -47.1349;  116.9311];
                                          [  52.3284;   75.6948;   -3.7964;    3.3106]];

                                         [[  31.6266;  -11.5726;   39.5819;   22.8020];
                                          [ -55.3912;  104.2333;   28.7020;   24.2710];
                                          [  91.6285;   27.1404;    8.1246;   38.5616];
                                          [ -37.8251;  -83.1444; -113.7539;   -7.7113]];

                                         [[  96.3737;  202.0389;  -68.9841;  -74.9820];
                                          [ -11.1773; -106.0468;  -94.3428; -101.9384];
                                          [ -44.8701;  -78.6259;  136.6283;   89.6921];
                                          [  60.9218;   14.3467;  -86.6495;   49.3313]]]])

        let t3p12 = Tensor.Conv2D(t1, t2, padding=[|1; 2|])
        let t3p12Correct = Tensor.Create([[[[   7.5867;   86.6988;    8.1164;  -85.8172;   69.5001;  -35.4485];
                                          [ 210.3501; -154.2592;   10.6089;   -1.4459; -126.2889;   24.8066];
                                          [ -42.1367; -176.1860; -132.3437; -165.9882;  -23.2585;  -44.1093];
                                          [-151.4929;  -62.8550; -180.0650;  -52.4599;   55.0733;   30.0922]];

                                         [[ -15.5535;    3.9697;  -53.5450;   16.3075;  -35.2008;   -7.1871];
                                          [  94.8112;  -60.7372;   97.8425;   81.2322;   20.0075;   33.2591];
                                          [ 127.0036;   -9.2216;  215.2763; -112.2244;   73.8351;  -30.0885];
                                          [ 245.2360;   88.4748;  308.1942;  176.2158;  131.2712;    1.4327]];

                                         [[  20.1355;    5.6857;   51.6497;  106.6138;  -17.3603; -112.0973];
                                          [ 173.8400;  -46.9604;  427.2891; -101.3674;  226.5788;  145.8927];
                                          [ 110.5519; -125.8047;  -35.6012; -168.9572; -141.2721; -159.3897];
                                          [ -16.8828; -105.4274; -132.2796;   35.6026;  -13.8173;   65.2295]]];


                                        [[[  70.6642;  115.1200; -141.3008;   36.3188;  -92.2498;   29.9960];
                                          [ 101.7243; -133.0979; -127.6157;  -35.6266;   42.1693;  -61.3766];
                                          [ -42.8275;   14.0058;   -7.7668;  -47.1349;  116.9311;   53.7170];
                                          [ -51.1392;   52.3284;   75.6948;   -3.7964;    3.3106;   54.5939]];

                                         [[   0.8100;   31.6266;  -11.5726;   39.5819;   22.8020;  -41.0836];
                                          [ -18.1888;  -55.3912;  104.2333;   28.7020;   24.2710;    3.6328];
                                          [  84.1016;   91.6285;   27.1404;    8.1246;   38.5616;   15.0304];
                                          [  68.3032;  -37.8251;  -83.1444; -113.7539;   -7.7113;  -66.3344]];

                                         [[  -7.6892;   96.3737;  202.0389;  -68.9841;  -74.9820;   85.7395];
                                          [  97.9534;  -11.1773; -106.0468;  -94.3428; -101.9384;  -46.0084];
                                          [  21.9169;  -44.8701;  -78.6259;  136.6283;   89.6921; -113.2355];
                                          [ -30.5091;   60.9218;   14.3467;  -86.6495;   49.3313;   22.9582]]]])

        let t3s2 = Tensor.Conv2D(t1, t2, stride=2)
        let t3s2Correct = Tensor.Create([[[[  10.6089]];

                                         [[  97.8425]];

                                         [[ 427.2891]]];


                                        [[[-127.6157]];

                                         [[ 104.2333]];

                                         [[-106.0468]]]])

        let t3s13 = Tensor.Conv2D(t1, t2, stride=[|1; 3|])
        let t3s13Correct = Tensor.Create([[[[  10.6089];
                                          [-132.3437]];

                                         [[  97.8425];
                                          [ 215.2763]];

                                         [[ 427.2891];
                                          [ -35.6012]]];


                                        [[[-127.6157];
                                          [  -7.7668]];

                                         [[ 104.2333];
                                          [  27.1404]];

                                         [[-106.0468];
                                          [ -78.6259]]]])

        let t3s2p1 = Tensor.Conv2D(t1, t2, stride=2, padding=1)
        let t3s2p1Correct = Tensor.Create([[[[  86.6988;  -85.8172];
                                              [-176.1860; -165.9882]];

                                             [[   3.9697;   16.3075];
                                              [  -9.2216; -112.2244]];

                                             [[   5.6857;  106.6138];
                                              [-125.8047; -168.9572]]];


                                            [[[ 115.1200;   36.3188];
                                              [  14.0058;  -47.1349]];

                                             [[  31.6266;   39.5819];
                                              [  91.6285;    8.1246]];

                                             [[  96.3737;  -68.9841];
                                              [ -44.8701;  136.6283]]]])

        let t3s23p32 = Tensor.Conv2D(t1, t2, stride=[2; 3], padding=[3; 2])
        let t3s23p32Correct = Tensor.Create([[[[   0.0000,    0.0000],
                                                  [   7.5866,  -85.8172],
                                                  [ -42.1364, -165.9885],
                                                  [ -67.0271,   97.8170]],

                                                 [[   0.0000,    0.0000],
                                                  [ -15.5537,   16.3071],
                                                  [ 127.0034, -112.2239],
                                                  [  78.7071,  -84.0060]],

                                                 [[   0.0000,    0.0000],
                                                  [  20.1357,  106.6139],
                                                  [ 110.5519, -168.9587],
                                                  [ -62.9899,  -13.2544]]],


                                                [[[   0.0000,    0.0000],
                                                  [  70.6642,   36.3191],
                                                  [ -42.8270,  -47.1361],
                                                  [   6.6860,   70.4299]],

                                                 [[   0.0000,    0.0000],
                                                  [   0.8102,   39.5820],
                                                  [  84.1018,    8.1256],
                                                  [  -4.9704,  -58.3407]],

                                                 [[   0.0000,    0.0000],
                                                  [  -7.6887,  -68.9838],
                                                  [  21.9173,  136.6280],
                                                  [  11.1650,   48.6844]]]])
        
        let t3p1d2 = Tensor.Conv2D(t1, t2, padding=1, dilation=2)
        let t3p1d2Correct = Tensor.Create([[[[ -72.7697,  -34.7305],
                                              [ -35.3463, -230.5320]],

                                             [[ -42.2859,   24.9292],
                                              [  96.3085,   25.1894]],

                                             [[-149.3111,   42.9268],
                                              [  73.8409, -159.8669]]],


                                            [[[ -57.9600,  -88.2215],
                                              [  50.7950,  -52.7872]],

                                             [[ -43.4812,   49.7672],
                                              [ -47.4554,   76.3617]],

                                             [[ -25.4452,   -9.8843],
                                              [  35.7940,   27.9557]]]])

        let t3p22d23 = Tensor.Conv2D(t1, t2, padding=[2;2], dilation=[2;3])
        let t3p22d23Correct = Tensor.Create([[[[-3.2693e+01, -4.3192e+01],
                                                  [ 4.7954e+01,  9.6877e+00],
                                                  [ 1.7971e+01, -7.0747e+01],
                                                  [-4.4577e+01, -1.7964e+01]],

                                                 [[ 9.0977e+00, -2.3489e+01],
                                                  [-4.1579e+00, -3.3179e+00],
                                                  [ 4.0888e+00, -3.3949e+01],
                                                  [ 3.4366e+01,  2.7721e+01]],

                                                 [[ 5.2087e+00, -1.3141e+01],
                                                  [-8.3409e+01, -5.3549e+01],
                                                  [ 2.7209e+01, -1.1435e+02],
                                                  [-2.0424e-02,  8.5139e+00]]],


                                                [[[ 4.6776e+01, -8.4654e-01],
                                                  [-5.5823e+00, -6.0218e+01],
                                                  [ 2.1814e+00,  1.0590e+01],
                                                  [-2.5290e+01,  2.5629e+01]],

                                                 [[ 4.2384e+00, -8.4199e+00],
                                                  [-3.8285e+01,  1.7978e+01],
                                                  [ 2.2481e+01,  6.5141e+01],
                                                  [-7.9511e-01, -9.9825e+00]],

                                                 [[-2.6924e+01, -8.0152e+01],
                                                  [-1.1862e+01,  2.7242e+01],
                                                  [ 3.1457e+01,  4.8352e+01],
                                                  [-8.1167e+01,  3.2597e+01]]]])

        let t3s3p6d3 = Tensor.Conv2D(t1, t2, stride=3, padding=6, dilation=3)
        let t3s3p6d3Correct = Tensor.Create([[[[  78.0793,   88.7191,  -32.2774,   12.5512],
                                                  [  27.0241, -107.5002,   98.7433,  -41.9933],
                                                  [  11.7470, -105.7288, -152.6583,   23.1514],
                                                  [ -67.0271,   60.8134,   74.5546,    9.3066]],

                                                 [[  -1.9717,   29.6326,   33.0870,   35.4221],
                                                  [  -3.6938,  -49.7435,  -66.3994,  -25.3134],
                                                  [  35.9503,   38.2935,   80.4125,   -2.5147],
                                                  [  78.7071,  -45.5705,   20.5010,  -15.2868]],

                                                 [[  -9.2327,   96.5872,   28.3565,   92.0639],
                                                  [  35.3198,    5.5638,  -14.6744, -150.4814],
                                                  [ 106.6989, -163.4741,   37.9205,   70.2904],
                                                  [ -62.9899,   25.6233,    7.3010,  -20.2932]]],


                                                [[[ -41.3512,  -21.4615,   29.8981,   -2.3176],
                                                  [  15.9843,  -22.6151,   87.3233,   36.7436],
                                                  [  46.3618,   66.0061,   18.5348,   38.1597],
                                                  [   6.6860,   65.4270,  -14.5871,  -45.0162]],

                                                 [[ -21.3053,  -12.6932,    4.7727,   -8.6866],
                                                  [ -23.4574,  -39.6679,   -1.5520,  -29.9771],
                                                  [ -66.3903, -127.3519,  -46.1654,  -79.1997],
                                                  [  -4.9704,  -93.0387,  -48.5467,  -39.6767]],

                                                 [[ -26.7460,  -27.8782,  -81.2187,  -76.9048],
                                                  [ -37.5283,  -29.9493,   60.9875,  -86.3384],
                                                  [  26.8834,  -22.3392,   64.3614,   32.6334],
                                                  [  11.1650,   45.6064,   -9.0581,   23.5884]]]])

        let t3b1 = Tensor.Conv2D(t1.[0].Unsqueeze(0) , t2)
        let t3b1Correct = t3Correct.[0].Unsqueeze(0)
        let t3b1s2 = Tensor.Conv2D(t1.[0].Unsqueeze(0), t2, stride = 2)
        let t3b1s2Correct = t3s2Correct.[0].Unsqueeze(0)

        // Assert.True(false)
        Assert.True(t3.ApproximatelyEqual(t3Correct))
        Assert.True(t3p1.ApproximatelyEqual(t3p1Correct))
        Assert.True(t3p12.ApproximatelyEqual(t3p12Correct))
        Assert.True(t3s2.ApproximatelyEqual(t3s2Correct))
        Assert.True(t3s13.ApproximatelyEqual(t3s13Correct))
        Assert.True(t3s2p1.ApproximatelyEqual(t3s2p1Correct))
        Assert.True(t3s23p32.ApproximatelyEqual(t3s23p32Correct))
        Assert.True(t3p1d2.ApproximatelyEqual(t3p1d2Correct))
        Assert.True(t3p22d23.ApproximatelyEqual(t3p22d23Correct))
        Assert.True(t3s3p6d3.ApproximatelyEqual(t3s3p6d3Correct))
        Assert.True(t3b1.ApproximatelyEqual(t3b1Correct))
        Assert.True(t3b1s2.ApproximatelyEqual(t3b1s2Correct))

    [<Test>]
    member this.TestTensorNegT () =
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Neg = -t1
        let t1NegCorrect = Tensor.Create([-1.; -2.; -3.])

        Assert.AreEqual(t1NegCorrect, t1Neg)

    [<Test>]
    member this.TestTensorSumT () =
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Sum = t1.Sum()
        let t1SumCorrect = Tensor.Create(6.)

        let t2 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t2Sum = t2.Sum()
        let t2SumCorrect = Tensor.Create(10.)

        Assert.AreEqual(t1SumCorrect, t1Sum)
        Assert.AreEqual(t2SumCorrect, t2Sum)

    [<Test>]
    member this.TestTensorSumT2Dim0 () =
        let t1 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t1Sum = t1.SumT2Dim0()
        let t1SumCorrect = Tensor.Create([4.; 6.])

        Assert.AreEqual(t1SumCorrect, t1Sum)
    
    [<Test>]
    member this.TestTensorSumDim () =
        (* Python:
        import numpy as np
        input = np.array([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
        input.sum(1)
        # --> array([[15., 18., 21., 24.],[51., 54., 57., 60.]])
        input.sum(2)
        # --> array([[10., 26., 42.],[58., 74., 90.]])
        *)
        let t = Tensor.Create([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
        let tSum0 = t.Sum(0)
        let tSum0Correct = Tensor.Create([[14., 16., 18., 20.], [22., 24., 26., 28.], [30., 32., 34., 36.]])
        let tSum1 = t.Sum(1)
        let tSum1Correct = Tensor.Create([[15., 18., 21., 24.], [51., 54., 57., 60.]])
        let tSum2 = t.Sum(2)
        let tSum2Correct = Tensor.Create([[10., 26., 42.], [58., 74., 90.]])

        Assert.AreEqual(tSum0Correct, tSum0)
        Assert.AreEqual(tSum1Correct, tSum1)
        Assert.AreEqual(tSum2Correct, tSum2)
    
    [<Test>]
    member this.TestTensorSumDimKeepDim () =
        (* Python:
        import torch
        input = torch.tensor([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
        input.sum(0,keepdim=True)
        # --> tensor([[[14., 16., 18., 20.],[22., 24., 26., 28.],[30., 32., 34., 36.]]])
        input.sum(1,keepdim=True)
        # --> tensor([[[15., 18., 21., 24.]],[[51., 54., 57., 60.]]])
        input.sum(2,keepdim=True)
        # --> tensor([[[10.],[26.],[42.]],[[58.],[74.],[90.]]])
        *)
        let t = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tSum0 = t.Sum(0, keepDim=true)
        let tSum0Correct = Tensor.Create([[[14.; 16.; 18.; 20.]; [22.; 24.; 26.; 28.]; [30.; 32.; 34.; 36.]]])
        let tSum1 = t.Sum(1, keepDim=true)
        let tSum1Correct = Tensor.Create([[[15.; 18.; 21.; 24.]]; [[51.; 54.; 57.; 60.]]])
        let tSum2 = t.Sum(2, keepDim=true)
        let tSum2Correct = Tensor.Create([[[10.]; [26.]; [42.]]; [[58.]; [74.]; [90.]]])

        Assert.AreEqual(tSum0Correct, tSum0)
        Assert.AreEqual(tSum1Correct, tSum1)
        Assert.AreEqual(tSum2Correct, tSum2)

    [<Test>]
    member this.TestTensorMean () =
        let t = Tensor.Create([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tMean = t.Mean()
        let tMeanCorrect = Tensor.Create(12.5)

        Assert.AreEqual(tMeanCorrect, tMean)

        // mean, dim={0,1,2}
        (* Python:
        import pytorch as torch
        input = np.[[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]]
        input.mean(1)
        --> array([[15., 18., 21., 24.],[51., 54., 57., 60.]])
        input.sum(2)
        --> array([[10., 26., 42.],[58., 74., 90.]])
        *)
        let tMean0 = t.Mean(0)
        let tMean0Correct = Tensor.Create([[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]])
        let tMean1 = t.Mean(1)
        let tMean1Correct = Tensor.Create([[5.; 6.; 7.; 8.]; [17.; 18.; 19.; 20.]])
        let tMean2 = t.Mean(2)
        let tMean2Correct = Tensor.Create([[2.5; 6.5; 10.5]; [14.5; 18.5; 22.5]])

        Assert.AreEqual(tMean0Correct, tMean0)
        Assert.AreEqual(tMean1Correct, tMean1)
        Assert.AreEqual(tMean2Correct, tMean2)

        // mean, dim={0,1,2}, keepDim=true
        (* Python:
        import torch
        input = torch.tensor([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
        input.mean(0,keepdim=True)
        # --> tensor([[[ 7.,  8.,  9., 10.],[11., 12., 13., 14.],[15., 16., 17., 18.]]])
        input.mean(1,keepdim=True)
        # --> tensor([[[ 5.,  6.,  7.,  8.]],[[17., 18., 19., 20.]]])
        input.mean(2,keepdim=True)
        # --> tensor([[[ 2.5000],[ 6.5000],[10.5000]],[[14.5000],[18.5000],[22.5000]]])
        *)
        let tMeanKeepDim0 = t.Mean(0, keepDim=true)
        let tMeanKeepDim0Correct = Tensor.Create([[[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]]])
        let tMeanKeepDim1 = t.Mean(1, keepDim=true)
        let tMeanKeepDim1Correct = Tensor.Create([[[5.; 6.; 7.; 8.]]; [[17.; 18.; 19.; 20.]]])
        let tMeanKeepDim2 = t.Mean(2, keepDim=true)
        let tMeanKeepDim2Correct = Tensor.Create([[[2.5]; [6.5]; [10.5]]; [[14.5]; [18.5]; [22.5]]])

        Assert.AreEqual(tMeanKeepDim0, tMeanKeepDim0Correct)
        Assert.AreEqual(tMeanKeepDim1, tMeanKeepDim1Correct)
        Assert.AreEqual(tMeanKeepDim2, tMeanKeepDim2Correct)

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

        // stddev, dim={0,1,2,3}, keepDim=true
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

        // stddev, dim={0,1,2,3}, keepDim=true
        (* Python:
        import torch
        input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
        input.std(0,keepdim=True)
        # --> tensor([[[0.2078, 0.2375, 0.2326, 0.0530],[0.0630, 0.2985, 0.2179, 0.0383],[0.2370, 0.4622, 0.3340, 0.3715]]])
        input.std(1,keepdim=True)
        # --> tensor([[[0.2331, 0.2980, 0.2911, 0.1304]],[[0.2393, 0.3789, 0.2015, 0.2581]]])
        input.std(2,keepdim=True)
        # --> tensor([[[0.2278],[0.2918],[0.2495]],[[0.1996],[0.2328],[0.2753]]]) 
        *)
        let tStddev0 = t.Stddev(0, keepDim=true)
        let tStddev0Correct = Tensor.Create([[[0.2078; 0.2375; 0.2326; 0.0530];[0.0630; 0.2985; 0.2179; 0.0383];[0.2370; 0.4623; 0.3339; 0.3715]]])
        let tStddev1 = t.Stddev(1, keepDim=true)
        let tStddev1Correct = Tensor.Create([[[0.2331; 0.2981; 0.2911; 0.1304]];[[0.2393; 0.3789; 0.2014; 0.2581]]])
        let tStddev2 = t.Stddev(2, keepDim=true)
        let tStddev2Correct = Tensor.Create([[[0.2277]; [0.2918]; [0.2495]];[[0.1996]; [0.2328]; [0.2753]]])

        Assert.True(tStddev0.ApproximatelyEqual(tStddev0Correct))
        Assert.True(tStddev1.ApproximatelyEqual(tStddev1Correct))
        Assert.True(tStddev2.ApproximatelyEqual(tStddev2Correct))

    [<Test>]
    member this.TestTensorVariance () =
        (* Python:
        import torch
        input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
        input.var()
        *)
        let t = Tensor.Create([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
        let tVariance = t.Variance()
        let tVarianceCorrect = Tensor.Create(0.0575)

        Assert.True(tVariance.ApproximatelyEqual(tVarianceCorrect))

        // Variance, dim={0,1,2,3}
        (* Python:
        input.var(0)
        # --> tensor([[0.0432, 0.0564, 0.0541, 0.0028],[0.0040, 0.0891, 0.0475, 0.0015],[0.0561, 0.2137, 0.1115, 0.1380]])
        input.var(1)
        # --> tensor([[0.0543, 0.0888, 0.0847, 0.0170],[0.0573, 0.1436, 0.0406, 0.0666]])
        input.var(2)
        # --> tensor([[0.0519, 0.0852, 0.0622],[0.0398, 0.0542, 0.0758]])
        *)
        let tVariance0 = t.Variance(0)
        let tVariance0Correct = Tensor.Create([[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]])
        let tVariance1 = t.Variance(1)
        let tVariance1Correct = Tensor.Create([[0.0543; 0.0888; 0.0847; 0.0170];[0.0573; 0.1436; 0.0406; 0.0666]])
        let tVariance2 = t.Variance(2)
        let tVariance2Correct = Tensor.Create([[0.0519; 0.0852; 0.0622];[0.0398; 0.0542; 0.0758]])

        Assert.True(tVariance0.ApproximatelyEqual(tVariance0Correct))
        Assert.True(tVariance1.ApproximatelyEqual(tVariance1Correct))
        Assert.True(tVariance2.ApproximatelyEqual(tVariance2Correct))

        // Variance, dim={0,1,2,3}, keepDim=true
        (* Python:
        import torch
        input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
        input.var(0,keepdim=True)
        # --> tensor([[[0.0432, 0.0564, 0.0541, 0.0028],[0.0040, 0.0891, 0.0475, 0.0015],[0.0561, 0.2137, 0.1115, 0.1380]]])
        input.var(1,keepdim=True)
        # --> tensor([[[0.0543, 0.0888, 0.0847, 0.0170]],[[0.0573, 0.1436, 0.0406, 0.0666]]])
        input.var(2,keepdim=True)
        # --> tensor([[[0.0519],[0.0852],[0.0622]],[[0.0398],[0.0542],[0.0758]]])
        *)
        let tVariance0 = t.Variance(0, keepDim=true)
        let tVariance0Correct = Tensor.Create([[[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]]])
        let tVariance1 = t.Variance(1, keepDim=true)
        let tVariance1Correct = Tensor.Create([[[0.0543; 0.0888; 0.0847; 0.0170]];[[0.0573; 0.1436; 0.0406; 0.0666]]])
        let tVariance2 = t.Variance(2, keepDim=true)
        let tVariance2Correct = Tensor.Create([[[0.0519];[0.0852];[0.0622]];[[0.0398];[0.0542];[0.0758]]])

        Assert.True(tVariance0.ApproximatelyEqual(tVariance0Correct))
        Assert.True(tVariance1.ApproximatelyEqual(tVariance1Correct))
        Assert.True(tVariance2.ApproximatelyEqual(tVariance2Correct))

    [<Test>]
    member this.TestTensorTransposeT2 () =
        let t1 = Tensor.Create([[1.; 2.; 3.]; [4.; 5.; 6.]])
        let t1Transpose = t1.Transpose()
        let t1TransposeCorrect = Tensor.Create([[1.; 4.]; [2.; 5.]; [3.; 6.]])

        let t2 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t2TransposeTranspose = t2.Transpose().Transpose()
        let t2TransposeTransposeCorrect = t2

        Assert.AreEqual(t1TransposeCorrect, t1Transpose)
        Assert.AreEqual(t2TransposeTransposeCorrect, t2TransposeTranspose)

    [<Test>]
    member this.TestTensorSignT () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.])
        let t1Sign = t1.Sign()
        let t1SignCorrect = Tensor.Create([-1.; -1.; 0.; 1.])

        Assert.AreEqual(t1SignCorrect, t1Sign)

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

        Assert.AreEqual(t1AbsCorrect, t1Abs)

    [<Test>]
    member this.TestTensorReluT () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.; 10.])
        let t1Relu = t1.Relu()
        let t1ReluCorrect = Tensor.Create([0.; 0.; 0.; 3.; 10.])

        Assert.AreEqual(t1ReluCorrect, t1Relu)

    [<Test>]
    member this.TestTensorLeakyRelu () =
        let t1 = Tensor.Create([-1.; -2.; 0.; 3.; 10.])
        let t1LeakyRelu = t1.LeakyRelu()
        let t1LeakyReluCorrect = Tensor.Create([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])

        Assert.AreEqual(t1LeakyReluCorrect, t1LeakyRelu)

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

        let t4 = Tensor.Create([[[[1.]]; 
                                 [[2.]]; 
                                 [[3.]]]; 
                                [[[4.]]; 
                                 [[5.]]; 
                                 [[6.]]]])
        let t4s1 = t4.[0]
        let t4s2 = t4.[0,*,*,*]
        let t4s1Correct = Tensor.Create([[[1]];
                                         [[2]];
                                         [[3]]])
        let t4s2Correct = t4s1Correct

        Assert.AreEqual(t1s1Correct, t1s1)
        Assert.AreEqual(t1s2Correct, t1s2)

        Assert.AreEqual(t2s1Correct, t2s1)
        Assert.AreEqual(t2s2Correct, t2s2)
        Assert.AreEqual(t2s3Correct, t2s3)
        Assert.AreEqual(t2s4Correct, t2s4)
        Assert.AreEqual(t2s5Correct, t2s5)
        Assert.AreEqual(t2s6Correct, t2s6)

        Assert.AreEqual(t2bs1Correct, t2bs1)
        Assert.AreEqual(t2bs2Correct, t2bs2)

        Assert.AreEqual(t3s1Correct, t3s1)
        Assert.AreEqual(t3s2Correct, t3s2)
        Assert.AreEqual(t3s3Correct, t3s3)
        Assert.AreEqual(t3s4Correct, t3s4)
        Assert.AreEqual(t3s5Correct, t3s5)
        Assert.AreEqual(t3s6Correct, t3s6)
        Assert.AreEqual(t3s7Correct, t3s7)
        Assert.AreEqual(t3s8Correct, t3s8)
        Assert.AreEqual(t3s9Correct, t3s9)
        Assert.AreEqual(t3s10Correct, t3s10)
        Assert.AreEqual(t3s11Correct, t3s11)
        Assert.AreEqual(t3s12Correct, t3s12)
        Assert.AreEqual(t3s13Correct, t3s13)
        Assert.AreEqual(t3s14Correct, t3s14)

        Assert.AreEqual(t4s1Correct, t4s1)
        Assert.AreEqual(t4s2Correct, t4s2)

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

        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)
        Assert.AreEqual(t5Correct, t5)

    [<Test>]
    member this.TestTensorDilateT () =
        let t1 = Tensor.Create([[1.;2.]; [3.;4.]])
        let t2 = t1.Dilate([|1; 2|])
        let t2Correct = Tensor.Create([[1.;0.;2.];[3.;0.;4.]])
        let t3 = t1.Dilate([|2; 2|])
        let t3Correct = Tensor.Create([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])
        let t4 = Tensor.Create([1.;2.;3.;4.])
        let t5 = t4.Dilate([|3|])
        let t5Correct = Tensor.Create([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])

        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t5Correct, t5)

    [<Test>]
    member this.TestTensorUndilateT () =
        let t1 = Tensor.Create([[1.;0.;2.];[3.;0.;4.]])
        let t2 = t1.Undilate([|1; 2|])
        let t2Correct = Tensor.Create([[1.;2.]; [3.;4.]])
        let t3 = Tensor.Create([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])
        let t4 = t3.Undilate([|2; 2|])
        let t4Correct = Tensor.Create([[1.;2.]; [3.;4.]])
        let t5 = Tensor.Create([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])
        let t6 = t5.Undilate([|3|])
        let t6Correct = Tensor.Create([1.;2.;3.;4.])

        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t4Correct, t4)
        Assert.AreEqual(t6Correct, t6)

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
        
        Assert.AreEqual(t1ShapeCorrect, t1Shape)
        Assert.AreEqual(t2ShapeCorrect, t2Shape)
        Assert.AreEqual(t3ShapeCorrect, t3Shape)
        Assert.AreEqual(t4ShapeCorrect, t4Shape)


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

        Assert.AreEqual(t1MaxCorrect, t1Max)
        Assert.AreEqual(t2MaxCorrect, t2Max)
        Assert.AreEqual(t3MaxCorrect, t3Max)
        Assert.AreEqual(t4MaxCorrect, t4Max)


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

        Assert.AreEqual(t1MinCorrect, t1Min)
        Assert.AreEqual(t2MinCorrect, t2Min)
        Assert.AreEqual(t3MinCorrect, t3Min)
        Assert.AreEqual(t4MinCorrect, t4Min)

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

        Assert.AreEqual(t0DepthCorrect, t0Depth)
        Assert.AreEqual(t1DepthCorrect, t1Depth)
        Assert.AreEqual(t2DepthCorrect, t2Depth)
        Assert.AreEqual(t3DepthCorrect, t3Depth)
