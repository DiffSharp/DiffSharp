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
        let t0 = dsharp.tensor(1.)
        let t0Shape = t0.shape
        let t0Dim = t0.dim
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.AreEqual(t0DimCorrect, t0Dim)
        Assert.AreEqual(t0ShapeCorrect, t0Shape)

    [<Test>]
    member this.TestTensorCreate1 () =
        // create from double list
        let t1 = dsharp.tensor([1.; 2.; 3.])
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        Assert.AreEqual(t1ShapeCorrect, t1.shape)
        Assert.AreEqual(t1DimCorrect, t1.dim)

        // create from double[]
        let t1Array = dsharp.tensor([| 1.; 2.; 3. |])

        Assert.AreEqual(t1ShapeCorrect, t1Array.shape)
        Assert.AreEqual(t1DimCorrect, t1Array.dim)

        // create from seq<double>
        let t1Seq = dsharp.tensor(seq { 1.; 2.; 3. })

        Assert.AreEqual(t1ShapeCorrect, t1Seq.shape)
        Assert.AreEqual(t1DimCorrect, t1Seq.dim)

    [<Test>]
    member this.TestTensorCreate2 () =
        let t2Values = [[1.; 2.; 3.]; [4.; 5.; 6.]]
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2
        // let t2DTypeCorrect = DType.Float32
        let t2ValuesCorrect = array2D (List.map (List.map float32) t2Values)

        // create from double list list
        let t2 = dsharp.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]])
        Assert.AreEqual(t2ShapeCorrect, t2.shape)
        Assert.AreEqual(t2DimCorrect, t2.dim)
        Assert.AreEqual(t2ValuesCorrect, t2.toArray())

        // create from double array list
        let t2ArrayList = dsharp.tensor([[|1.; 2.; 3.|]; [|4.; 5.; 6.|]])
        Assert.AreEqual(t2ShapeCorrect, t2ArrayList.shape)
        Assert.AreEqual(t2DimCorrect, t2ArrayList.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArrayList.toArray())

        // create from double list array
        let t2ListArray = dsharp.tensor([| [1.; 2.; 3.]; [4.; 5.; 6.] |])
        Assert.AreEqual(t2ShapeCorrect, t2ListArray.shape)
        Assert.AreEqual(t2DimCorrect, t2ListArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ListArray.toArray())

        // create from double[][]
        let t2ArrayArray = dsharp.tensor([| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.AreEqual(t2ShapeCorrect, t2ArrayArray.shape)
        Assert.AreEqual(t2DimCorrect, t2ArrayArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArrayArray.toArray())

        // create from double[,]
        let t2Array2D = dsharp.tensor(array2D [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.AreEqual(t2ShapeCorrect, t2Array2D.shape)
        Assert.AreEqual(t2DimCorrect, t2Array2D.dim)
        Assert.AreEqual(t2ValuesCorrect, t2Array2D.toArray())

        // create from seq<double[]>
        let t2ArraySeq = dsharp.tensor(seq { yield [| 1.; 2.; 3. |]; yield [| 4.; 5.; 6.|] })
        Assert.AreEqual(t2ShapeCorrect, t2ArraySeq.shape)
        Assert.AreEqual(t2DimCorrect, t2ArraySeq.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArraySeq.toArray())

        // create from seq<seq<double>>
        let t2SeqSeq = dsharp.tensor(seq { seq { 1.; 2.; 3. }; seq { 4.; 5.; 6.} })
        Assert.AreEqual(t2ShapeCorrect, t2SeqSeq.shape)
        Assert.AreEqual(t2DimCorrect, t2SeqSeq.dim)
        Assert.AreEqual(t2ValuesCorrect, t2SeqSeq.toArray())

        // create from (double * double * double) list list
        let t2TupleListList = dsharp.tensor([ [ 1., 2., 3. ]; [ 4., 5., 6. ] ])
        Assert.AreEqual(t2ShapeCorrect, t2TupleListList.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleListList.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleListList.toArray())

        // create from ((double * double * double) list * (double * double * double) list) list
        let t2TupleListTupleList = dsharp.tensor([ [ 1., 2., 3. ], [ 4., 5., 6. ] ])
        Assert.AreEqual(t2ShapeCorrect, t2TupleListTupleList.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleListTupleList.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleListTupleList.toArray())

        // create from (double * double * double)[]
        let t2TupleArray = dsharp.tensor([| [ 1., 2., 3. ]; [ 4., 5., 6. ] |])
        Assert.AreEqual(t2ShapeCorrect, t2TupleArray.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArray.toArray())

        // create from ((double * double * double) [] * (double * double * double) []) []
        let t2TupleArrayTupleArray = dsharp.tensor([| [| 1., 2., 3. |], [| 4., 5., 6. |] |])
        Assert.AreEqual(t2ShapeCorrect, t2TupleArrayTupleArray.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArrayTupleArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArrayTupleArray.toArray())
        Assert.AreEqual(t2ValuesCorrect, t2TupleArrayTupleArray.toArray())

        // create from (double * double * double)seq
        let t2TupleArray = dsharp.tensor(seq { [ 1., 2., 3. ]; [ 4., 5., 6. ] })
        Assert.AreEqual(t2ShapeCorrect, t2TupleArray.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArray.toArray())

        let t2TupleOfList = dsharp.tensor [[2.], [3.], [4.]]
        Assert.AreEqual([| 3; 1 |], t2TupleOfList.shape)
        Assert.AreEqual(array2D [ [2]; [3]; [4] ], t2TupleOfList.toArray())

    [<Test>]
    member this.TestTensorCreate3 () =
        let t3Values = [[[1.; 2.; 3.]; [4.; 5.; 6.]]]
        let t3 = dsharp.tensor(t3Values)
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3
        let t3ValuesCorrect = Util.array3D (List.map (List.map (List.map float32)) t3Values)

        Assert.AreEqual(t3ShapeCorrect, t3.shape)
        Assert.AreEqual(t3DimCorrect, t3.dim)
        Assert.AreEqual(t3ValuesCorrect, t3.toArray())

    [<Test>]
    member this.TestTensorCreate4 () =
        let t4Values = [[[[1.; 2.]]]]
        let t4 = dsharp.tensor(t4Values)
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4
        let t4ValuesCorrect = Util.array4D (List.map (List.map (List.map (List.map float32))) t4Values)

        Assert.AreEqual(t4ShapeCorrect, t4.shape)
        Assert.AreEqual(t4DimCorrect, t4.dim)
        Assert.AreEqual(t4ValuesCorrect, t4.toArray())

    [<Test>]
    member this.TestTensorToArray () =
        let a = array2D [[1.; 2.]; [3.; 4.]]
        let t = dsharp.tensor(a)
        let v = t.toArray()
        Assert.AreEqual(a, v)

    [<Test>]
    member this.TestTensorSaveLoad () =
        let a = dsharp.tensor([[1,2],[3,4]])
        let fileName = System.IO.Path.GetTempFileName()
        a.save(fileName)
        let b = Tensor.load(fileName)
        Assert.AreEqual(a, b)

    [<Test>]
    member this.TestTensorClone () =
        let a = dsharp.randn([2;3])
        let b = a.clone()
        Assert.AreEqual(a, b)

    [<Test>]
    member this.TestTensorFull () =
        let t1a = dsharp.full([2;3], 2.5)
        let t1b = dsharp.ones([2;3]) * 2.5
        let t2a = dsharp.full([], 2.5)
        let t2b = dsharp.ones([]) * 2.5
        let t3a = dsharp.full([5], dsharp.tensor(2.5))
        let t3b = dsharp.ones([5]) * 2.5
        Assert.AreEqual(t1a, t1b)
        Assert.AreEqual(t2a, t2b)
        Assert.AreEqual(t3a, t3b)

    [<Test>]
    member this.TestTensorIsTensor () =
        let a = 2.
        let b = dsharp.tensor(2.)
        Assert.True(not (dsharp.isTensor(a)))
        Assert.True(dsharp.isTensor(b))    

    [<Test>]
    member this.TestTensorOnehot () =
        let t0 = dsharp.onehot(3, 0)
        let t1 = dsharp.onehot(3, 1)
        let t2 = dsharp.onehot(3, 2)
        let t0Correct = dsharp.tensor([1,0,0])
        let t1Correct = dsharp.tensor([0,1,0])
        let t2Correct = dsharp.tensor([0,0,1])
        Assert.AreEqual(t0Correct, t0)
        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)

    [<Test>]
    member this.TestTensorToString () =
        let t0 = dsharp.tensor(2.)
        let t1 = dsharp.tensor([[2.]; [2.]])
        let t2 = dsharp.tensor([[[2.; 2.]]])
        let t3 = dsharp.tensor([[1.;2.]; [3.;4.]])
        let t4 = dsharp.tensor([[[[1.]]]])
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
        let t1A = dsharp.tensor(-1.)
        let t1B = dsharp.tensor(1.)
        let t1C = dsharp.tensor(1.)
        let t1At1BLess = t1A < t1B
        let t1At1BLessCorrect = true
        let t1At1BEqual = t1A = t1B
        let t1At1BEqualCorrect = false
        let t1Bt1CEqual = t1B = t1C
        let t1Bt1CEqualCorrect = true

        Assert.AreEqual(t1At1BLessCorrect, t1At1BLess)
        Assert.AreEqual(t1At1BEqualCorrect, t1At1BEqual)
        Assert.AreEqual(t1Bt1CEqualCorrect, t1Bt1CEqual)

        // Systematic testing. The tensors below are listed in expected order of comparison
        let t2S =
            [ dsharp.tensor( 0. )
              dsharp.tensor( 1. )
              dsharp.tensor([ 1.] )
              dsharp.tensor([ 2.] )
              dsharp.tensor([ 1.; 1.] )
              dsharp.tensor([ 1.; 2. ] )
              dsharp.tensor([ 2.; 1. ] ) 
              dsharp.tensor([ [ 1.; 1.] ]) ]

        // Check the F# generic '=' gives expected results
        let equalsResults = [| for a in t2S -> [| for b in t2S -> a = b |] |]
        let equalsCorrect = [| for i in 0..t2S.Length-1 -> [| for j in 0..t2S.Length-1 -> (i=j) |] |]

        Assert.AreEqual(equalsResults, equalsCorrect)

        // Check the F# generic hashes are the same for identical tensors, and different for this small sample of tensors
        let hashSameResults = [| for a in t2S -> [| for b in t2S -> hash a = hash b |] |]
        let hashSameCorrect = [| for i in 0..t2S.Length-1 -> [| for j in 0..t2S.Length-1 -> (i=j) |] |]

        Assert.AreEqual(hashSameResults, hashSameCorrect)

        // Check reallocating an identical tensor doesn't change the hash
        let t2a = dsharp.tensor([ 1.] )
        let t2b = dsharp.tensor([ 1.] )
        Assert.AreEqual(t2a.GetHashCode(), t2b.GetHashCode())

        // Check adding `ForwardDiff` doesn't change the hash or equality
        Assert.AreEqual(t2a.forwardDiff(dsharp.tensor([1.])).GetHashCode(), t2a.GetHashCode())
        Assert.AreEqual(true, (t2a.forwardDiff(dsharp.tensor([1.]))) = t2a)

        // Check adding `ReverseDiff` doesn't change the hash or equality
        Assert.AreEqual(t2a.reverseDiff().GetHashCode(), t2a.GetHashCode())
        Assert.AreEqual(true, (t2a.reverseDiff()) = t2a)

    [<Test>]
    member this.TestTensorLtTT () =
        let t1 = dsharp.tensor([1.; 2.; 3.; 5.])
        let t2 = dsharp.tensor([1.; 3.; 5.; 4.])
        let t1t2Lt = t1.lt(t2)
        let t1t2LtCorrect = dsharp.tensor([0.; 1.; 1.; 0.])

        Assert.AreEqual(t1t2LtCorrect, t1t2Lt)

    [<Test>]
    member this.TestTensorLeTT () =
        let t1 = dsharp.tensor([1.; 2.; 3.; 5.])
        let t2 = dsharp.tensor([1.; 3.; 5.; 4.])
        let t1t2Le = t1.le(t2)
        let t1t2LeCorrect = dsharp.tensor([1.; 1.; 1.; 0.])

        Assert.AreEqual(t1t2LeCorrect, t1t2Le)

    [<Test>]
    member this.TestTensorGtTT () =
        let t1 = dsharp.tensor([1.; 2.; 3.; 5.])
        let t2 = dsharp.tensor([1.; 3.; 5.; 4.])
        let t1t2Gt = t1.gt(t2)
        let t1t2GtCorrect = dsharp.tensor([0.; 0.; 0.; 1.])

        Assert.AreEqual(t1t2GtCorrect, t1t2Gt)

    [<Test>]
    member this.TestTensorGeTT () =
        let t1 = dsharp.tensor([1.; 2.; 3.; 5.])
        let t2 = dsharp.tensor([1.; 3.; 5.; 4.])
        let t1t2Ge = t1.ge(t2)
        let t1t2GeCorrect = dsharp.tensor([1.; 0.; 0.; 1.])

        Assert.AreEqual(t1t2GeCorrect, t1t2Ge)

    [<Test>]
    member this.TestTensorIsinf () =
        let t = dsharp.tensor([1.; infinity; 3.; -infinity])
        let i = dsharp.isinf(t)
        let iCorrect = dsharp.tensor([0.; 1.; 0.; 1.])
        Assert.AreEqual(iCorrect, i)

    [<Test>]
    member this.TestTensorIsnan () =
        let t = dsharp.tensor([1.; nan; 3.; nan])
        let i = dsharp.isnan(t)
        let iCorrect = dsharp.tensor([0.; 1.; 0.; 1.])
        Assert.AreEqual(iCorrect, i)

    [<Test>]
    member this.TestTensorHasinf () =
        let t1 = dsharp.tensor([1.; infinity; 3.; -infinity])
        let t1i = dsharp.hasinf(t1)
        let t1iCorrect = true
        let t2 = dsharp.tensor([1.; 2.; 3.; 4.])
        let t2i = dsharp.hasinf(t2)
        let t2iCorrect = false
        Assert.AreEqual(t1iCorrect, t1i)
        Assert.AreEqual(t2iCorrect, t2i)

    [<Test>]
    member this.TestTensorHasnan () =
        let t1 = dsharp.tensor([1.; nan; 3.; nan])
        let t1i = dsharp.hasnan(t1)
        let t1iCorrect = true
        let t2 = dsharp.tensor([1.; 2.; 3.; 4.])
        let t2i = dsharp.hasnan(t2)
        let t2iCorrect = false
        Assert.AreEqual(t1iCorrect, t1i)
        Assert.AreEqual(t2iCorrect, t2i)

    [<Test>]
    member this.TestTensorAddTT () =
        let t1 = dsharp.tensor([1.; 2.]) + dsharp.tensor([3.; 4.])
        let t1Correct = dsharp.tensor([4.; 6.])

        let t2 = dsharp.tensor([1.; 2.]) + dsharp.tensor(5.)
        let t2Correct = dsharp.tensor([6.; 7.])

        let t3 = dsharp.tensor([1.; 2.]) + 5.f
        let t3Correct = dsharp.tensor([6.; 7.])

        let t4 = dsharp.tensor([1.; 2.]) + 5.
        let t4Correct = dsharp.tensor([6.; 7.])

        let t5 = dsharp.tensor([1.; 2.]) + 5
        let t5Correct = dsharp.tensor([6.; 7.])

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)
        Assert.AreEqual(t5Correct, t5)

        // Check all broadcasts into 2x2
        // 2x2 * 1  (broadcast --> 2x2)
        // 2x2 * 2  (broadcast --> 2x2)
        // 2x2 * 2x1  (broadcast --> 2x2)
        // 2x2 * 1x2  (broadcast --> 2x2)
        let t6a = dsharp.tensor([ [1.; 2.]; [3.; 4.] ])
        for t6b in [ dsharp.tensor([ 5.0 ])
                     dsharp.tensor([ 5.0; 5.0 ])
                     dsharp.tensor([ [5.0]; [5.0] ])
                     dsharp.tensor([ [5.0; 5.0] ]) ] do
            let t6 = t6a + t6b
            let t6Commute = t6b + t6a
            let t6Correct = dsharp.tensor([ [6.; 7.]; [8.; 9.] ])

            Assert.AreEqual(t6Correct, t6)
            Assert.AreEqual(t6Correct, t6Commute)

        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 + 1  (broadcast --> 2x3x4)
        // 2x3x4 + 4  (broadcast --> 2x3x4)
        // 2x3x4 + 1x1  (broadcast --> 2x3x4)
        // 2x3x4 + 3x1  (broadcast --> 2x3x4)
        // 2x3x4 + 1x4  (broadcast --> 2x3x4)
        // etc.
        let t7a = dsharp.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
                                  [ [13.; 14.; 15.; 16.]; [17.; 18.; 19.; 20.]; [21.; 22.; 23.; 24.] ]  ])
        let t7Shapes = 
            [ for i1 in [0;1;2] do
                for i2 in [0;1;3] do
                  for i3 in [0;1;4] do 
                    if i1 <> 2 || i2 <> 3 || i3 <> 4 then
                        [| if i1 <> 0 && i2 <> 0 && i3 <> 0 then yield i1
                           if i2 <> 0 && i3 <> 0 then yield i2
                           if i3 <> 0 then yield i3 |] ]
            |> List.distinct

        let t7Results, t7CommuteResults = 
            [| for shape in t7Shapes do 
                  let t7b = dsharp.tensor( Util.arrayND shape (fun is -> double (Array.sum is) + 2.0))
                  let t7 = t7a + t7b
                  let t7Commute = t7b + t7a
                  yield (t7b, t7), (t7b, t7Commute) |]
            |> Array.unzip

        let t7Expected =
            [|(dsharp.tensor 2.,                                                       dsharp.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (dsharp.tensor [2.],                                                     dsharp.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (dsharp.tensor [2., 3., 4., 5.],                                         dsharp.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (dsharp.tensor [[2.]],                                                   dsharp.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (dsharp.tensor [[2., 3., 4., 5.]],                                       dsharp.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (dsharp.tensor [[2.], [3.], [4.]],                                       dsharp.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[15., 16., 17., 18.], [20., 21., 22., 23.], [25., 26., 27., 28.]]]);
              (dsharp.tensor [[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]],   dsharp.tensor [[[3., 5., 7., 9.], [8., 10., 12., 14.], [13., 15., 17., 19.]], [[15., 17., 19., 21.], [20., 22., 24., 26.], [25., 27., 29., 31.]]]);
              (dsharp.tensor [[[2.]]],                                                 dsharp.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (dsharp.tensor [[[2., 3., 4., 5.]]],                                     dsharp.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (dsharp.tensor [[[2.], [3.], [4.]]],                                     dsharp.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[15., 16., 17., 18.], [20., 21., 22., 23.], [25., 26., 27., 28.]]]);
              (dsharp.tensor [[[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]], dsharp.tensor [[[3., 5., 7., 9.], [8., 10., 12., 14.], [13., 15., 17., 19.]], [[15., 17., 19., 21.], [20., 22., 24., 26.], [25., 27., 29., 31.]]]);
              (dsharp.tensor [[[2.]], [[3.]]],                                         dsharp.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[16., 17., 18., 19.], [20., 21., 22., 23.], [24., 25., 26., 27.]]]);
              (dsharp.tensor [[[2., 3., 4., 5.]], [[3., 4., 5., 6.]]],                 dsharp.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[16., 18., 20., 22.], [20., 22., 24., 26.], [24., 26., 28., 30.]]]);
              (dsharp.tensor [[[2.], [3.], [4.]], [[3.], [4.], [5.]]],                 dsharp.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[16., 17., 18., 19.], [21., 22., 23., 24.], [26., 27., 28., 29.]]])|]


        Assert.AreEqual(t7Expected, t7Results)
        Assert.AreEqual(t7Expected, t7CommuteResults)

    [<Test>]
    member this.TestTensorStackTs () =
        let t0a = dsharp.tensor(1.)
        let t0b = dsharp.tensor(3.)
        let t0c = dsharp.tensor(5.)
        let t0 = Tensor.stack([t0a;t0b;t0c])
        let t0_dim0 = Tensor.stack([t0a;t0b;t0c], dim=0)
        let t0Correct = dsharp.tensor([1.;3.;5.])

        let t1a = dsharp.tensor([1.; 2.])
        let t1b = dsharp.tensor([3.; 4.])
        let t1c = dsharp.tensor([5.; 6.])
        let t1 = Tensor.stack([t1a;t1b;t1c])
        let t1_dim1 = Tensor.stack([t1a;t1b;t1c], dim=1)
        let t1Correct = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]])
        let t1Correct_dim1 = dsharp.tensor([[1.;3.;5.];[2.;4.;6.]])

        let t2a = dsharp.tensor([ [1.; 2.] ])
        let t2b = dsharp.tensor([ [3.; 4.] ])
        let t2c = dsharp.tensor([ [5.; 6.] ])
        let t2 = Tensor.stack([t2a;t2b;t2c])
        let t2_dim0 = Tensor.stack([t2a;t2b;t2c], dim=0)
        let t2_dim1 = Tensor.stack([t2a;t2b;t2c], dim=1)
        let t2_dim2 = Tensor.stack([t2a;t2b;t2c], dim=2)
        let t2Correct_dim0 = dsharp.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t2Correct_dim1 = dsharp.tensor([[[1.;2.];[3.;4.];[5.;6.]]])
        let t2Correct_dim2 = dsharp.tensor([[[1.;3.;5.];[2.;4.;6.]]])

        Assert.AreEqual(t0Correct, t0)
        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct_dim0, t2)
        Assert.AreEqual(t0Correct, t0_dim0)
        Assert.AreEqual(t1Correct_dim1, t1_dim1)
        Assert.AreEqual(t2Correct_dim0, t2_dim0)
        Assert.AreEqual(t2Correct_dim1, t2_dim1)
        Assert.AreEqual(t2Correct_dim2, t2_dim2)

    [<Test>]
    member this.TestTensorUnstackT () =
        let t0a = dsharp.tensor(1.)
        let t0b = dsharp.tensor(3.)
        let t0c = dsharp.tensor(5.)
        let t0Correct = [t0a;t0b;t0c]
        let t0 = Tensor.stack(t0Correct).unstack()

        let t1a = dsharp.tensor([1.; 2.])
        let t1b = dsharp.tensor([3.; 4.])
        let t1c = dsharp.tensor([5.; 6.])
        let t1Correct = [t1a;t1b;t1c]
        let t1Correct_dim1 = [dsharp.tensor [1.;3.;5.]; dsharp.tensor [2.;4.;6.]]
        let t1 = Tensor.stack(t1Correct).unstack()
        let t1_dim1 = Tensor.stack(t1Correct).unstack(dim=1)

        // 3x1x2
        let t2a = dsharp.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t2 = t2a.unstack()
        let t2_dim1 = t2a.unstack(dim=1)
        let t2_dim2 = t2a.unstack(dim=2)
        // 3 of 1x2
        let t2Correct = [dsharp.tensor [[1.;2.]]; dsharp.tensor [[3.;4.]]; dsharp.tensor [[5.;6.]]]
        // 1 of 3x2
        let t2Correct_dim1 = [dsharp.tensor [[1.;2.];[3.;4.];[5.;6.]]]
        // 2 of 3x1
        let t2Correct_dim2 = [dsharp.tensor [[1.];[3.];[5.]]; dsharp.tensor [[2.];[4.];[6.]]]

        Assert.AreEqual(t0Correct, t0)
        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t1Correct_dim1, t1_dim1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t2Correct_dim1, t2_dim1)
        Assert.AreEqual(t2Correct_dim2, t2_dim2)

    [<Test>]
    member this.TestTensorCatTs () =

        let t0a = dsharp.tensor([1.; 2.])
        let t0 = Tensor.cat([t0a])
        let t0Correct = dsharp.tensor([1.;2.])

        Assert.AreEqual(t0Correct, t0)

        let t1a = dsharp.tensor([1.; 2.]) // 2
        let t1b = dsharp.tensor([3.; 4.]) // 2
        let t1c = dsharp.tensor([5.; 6.]) // 2
        let t1 = Tensor.cat([t1a;t1b;t1c]) // 6
        let t1_dim0 = Tensor.cat([t1a;t1b;t1c],dim=0) // 6
        let t1Correct = dsharp.tensor([1.;2.;3.;4.;5.;6.])

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t1Correct, t1_dim0)

        let t2a = dsharp.tensor([ [1.; 2.] ]) // 1x2
        let t2b = dsharp.tensor([ [3.; 4.] ]) // 1x2
        let t2c = dsharp.tensor([ [5.; 6.] ]) // 1x2
        let t2 = Tensor.cat([t2a;t2b;t2c]) // 3x2
        let t2_dim0 = Tensor.cat([t2a;t2b;t2c], dim=0) // 3x2
        let t2_dim1 = Tensor.cat([t2a;t2b;t2c], dim=1) // 1x6
        let t2Correct_dim0 = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.]]) // 3x2
        let t2Correct_dim1 = dsharp.tensor([[1.;2.;3.;4.;5.;6.]]) // 1x6

        Assert.AreEqual(t2Correct_dim0, t2)
        Assert.AreEqual(t2Correct_dim0, t2_dim0)
        Assert.AreEqual(t2Correct_dim1, t2_dim1)

        // irregular sizes dim0
        let t3a = dsharp.tensor([ [1.; 2.] ]) // 1x2
        let t3b = dsharp.tensor([ [3.; 4.];[5.; 6.] ]) // 2x2
        let t3c = dsharp.tensor([ [7.; 8.] ]) // 1x2
        let t3 = Tensor.cat([t3a;t3b;t3c]) // 4x2
        let t3Correct = dsharp.tensor([[1.;2.];[3.;4.];[5.;6.];[7.;8.]]) // 4x2

        Assert.AreEqual(t3Correct, t3)

        // irregular sizes dim1
        let t4a = dsharp.tensor([ [1.]; [2.] ]) // 2x1
        let t4b = dsharp.tensor([ [3.; 4.];[5.; 6.] ]) // 2x2
        let t4c = dsharp.tensor([ [7.]; [8.] ]) // 2x1
        let t4_dim1 = Tensor.cat([t4a;t4b;t4c],dim=1) // 2x4
        let t4Correct_dim1 = dsharp.tensor([[1.;3.;4.;7.];[2.;5.;6.;8.]]) // 2x4

        Assert.AreEqual(t4Correct_dim1, t4_dim1)

    [<Test>]
    member this.TestTensorSplitT () =
        
        //6 --> 2;2;2
        let t1in = dsharp.tensor([1.;2.;3.;4.;5.;6.]) // 6
        let t1 = t1in.split([2;2;2]) |> Seq.toList // 3 of 2
        let t1Correct = [dsharp.tensor([1.; 2.]);dsharp.tensor([3.; 4.]);dsharp.tensor([5.; 6.])]

        Assert.AreEqual(t1Correct, t1)

        // 3x1x2
        let t2in = dsharp.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t2 = t2in.split(sizes=[1;1;1], dim=0)  |> Seq.toList // 3 of 1x1x2
        let t2Correct = [dsharp.tensor [[[1.;2.]]]; dsharp.tensor [[[3.;4.]]]; dsharp.tensor [[[5.;6.]]]]

        Assert.AreEqual(t2Correct, t2)

        let t3in = dsharp.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t3 = t3in.split(sizes=[1;2], dim=0)  |> Seq.toList // 2 of 1x1x2 and 2x1x2
        let t3Correct = [dsharp.tensor [[[1.;2.]]]; dsharp.tensor [[[3.;4.]];[[5.;6.]]]]

        Assert.AreEqual(t3Correct, t3)

        let t4in = dsharp.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t4 = t4in.split(sizes=[1], dim=1)  |> Seq.toList // 1 of 3x1x2
        let t4Correct = [dsharp.tensor [[[1.;2.]];[[3.;4.]];[[5.;6.]]]] // 1 of 3x1x2

        Assert.AreEqual(t4Correct, t4)

        let t5in = dsharp.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t5 = t5in.split(sizes=[1;1], dim=2)  |> Seq.toList // 2 of 3x1x1
        let t5Correct = [dsharp.tensor [[[1.]];[[3.]];[[5.]]]; dsharp.tensor [[[2.]];[[4.]];[[6.]]]] // 2 of 3x1x1

        Assert.AreEqual(t5Correct, t5)

        //systematic split of 6 
        let t6vs = [1..6]
        let t6in = dsharp.tensor(t6vs) // 6
        for p1 in 0..6 do
          for p2 in 0..6 do
            for p3 in 0..6 do
               if p1+p2+p3 = 6 then 
                  let t6 = 
                      t6in.split([if p1 > 0 then p1 
                                  if p2 > 0 then p2
                                  if p3 > 0 then p3])
                      |> Seq.toList 
                  let t6Correct = 
                      [if p1 > 0 then dsharp.tensor(t6vs.[0..p1-1]);
                       if p2 > 0 then dsharp.tensor(t6vs.[p1..p1+p2-1]);
                       if p3 > 0 then dsharp.tensor(t6vs.[p1+p2..])]

                  Assert.AreEqual(t6Correct, t6)


        //systematic split of 2x6 along dim1
        let t7vs1 = [1..6]
        let t7vs2 = [7..12]
        let t7in = dsharp.tensor([ t7vs1; t7vs2] ) // 2x6
        for p1 in 0..6 do
          for p2 in 0..6 do
            for p3 in 0..6 do
               if p1+p2+p3 = 6 then 
                  let sizes =
                      [if p1 > 0 then p1 
                       if p2 > 0 then p2
                       if p3 > 0 then p3]
                  let t7 = t7in.split(sizes,dim=1) |> Seq.toList 
                  let t7Correct = 
                      [if p1 > 0 then dsharp.tensor([ t7vs1.[0..p1-1];     t7vs2.[0..p1-1] ]);
                       if p2 > 0 then dsharp.tensor([ t7vs1.[p1..p1+p2-1]; t7vs2.[p1..p1+p2-1] ]);
                       if p3 > 0 then dsharp.tensor([ t7vs1.[p1+p2..];     t7vs2.[p1+p2..] ])]

                  Assert.AreEqual(t7Correct, t7)


    [<Test>]
    member this.TestTensorAddT2T1 () =
        let t1 = dsharp.tensor([[1.; 2.]; [3.; 4.]]) + dsharp.tensor([5.; 6.])
        let t1Correct = dsharp.tensor([[6.; 8.]; [8.; 10.]])

        Assert.AreEqual(t1Correct, t1)

    [<Test>]
    member this.TestTensorSubTT () =
        let t1 = dsharp.tensor([1.; 2.]) - dsharp.tensor([3.; 4.])
        let t1Correct = dsharp.tensor([-2.; -2.])

        let t2 = dsharp.tensor([1.; 2.]) - dsharp.tensor(5.)
        let t2Correct = dsharp.tensor([-4.; -3.])

        let t3 = dsharp.tensor([1.; 2.]) - 5.f
        let t3Correct = dsharp.tensor([-4.; -3.])

        let t4 = 5. - dsharp.tensor([1.; 2.])
        let t4Correct = dsharp.tensor([4.; 3.])

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)

    [<Test>]
    member this.TestTensorMulTT () =
        let t1 = dsharp.tensor([1.; 2.]) * dsharp.tensor([3.; 4.])
        let t1Correct = dsharp.tensor([3.; 8.])

        Assert.AreEqual(t1Correct, t1)

        let t2 = dsharp.tensor([1.; 2.]) * dsharp.tensor(5.)
        let t2Correct = dsharp.tensor([5.; 10.])

        Assert.AreEqual(t2Correct, t2)

        let t3 = dsharp.tensor([1.; 2.]) * 5.f
        let t3Correct = dsharp.tensor([5.; 10.])

        Assert.AreEqual(t3Correct, t3)

        let t4 = 5. * dsharp.tensor([1.; 2.])
        let t4Correct = dsharp.tensor([5.; 10.])

        Assert.AreEqual(t4Correct, t4)

        // 2x2 * 1  (broadcast --> 2x2)
        // 2x2 * 2  (broadcast --> 2x2)
        // 2x2 * 2x1  (broadcast --> 2x2)
        // 2x2 * 1x2  (broadcast --> 2x2)
        let t5a = dsharp.tensor([ [1.; 2.]; [3.; 4.] ])
        for t5b in [ dsharp.tensor([ 5.0 ])
                     dsharp.tensor([ 5.0; 5.0 ])
                     dsharp.tensor([ [5.0]; [5.0] ])
                     dsharp.tensor([ [5.0; 5.0] ]) ] do
            let t5 = t5a * t5b
            let t5Commute = t5b * t5a
            let t5Correct = dsharp.tensor([ [5.; 10.]; [15.; 20.] ])

            Assert.AreEqual(t5Correct, t5)
            Assert.AreEqual(t5Correct, t5Commute)

        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 * 1  (broadcast --> 2x3x4)
        // 2x3x4 * 4  (broadcast --> 2x3x4)
        // 2x3x4 * 1x1  (broadcast --> 2x3x4)
        // 2x3x4 * 3x1  (broadcast --> 2x3x4)
        // 2x3x4 * 1x4  (broadcast --> 2x3x4)
        // etc.
        let t6a = dsharp.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
                                    [ [13.; 14.; 15.; 16.]; [17.; 18.; 19.; 20.]; [21.; 22.; 23.; 24.] ]  ])

        // These are all the interesting shapes that broadcast into t6a
        let t6Shapes = 
            [ for i1 in [0;1;2] do
                for i2 in [0;1;3] do
                  for i3 in [0;1;4] do 
                    if i1 <> 2 || i2 <> 3 || i3 <> 4 then
                        [| if i1 <> 0 && i2 <> 0 && i3 <> 0 then yield i1
                           if i2 <> 0 && i3 <> 0 then yield i2
                           if i3 <> 0 then yield i3 |] ]
            |> List.distinct

        let t6Results, t6CommuteResults = 
            [| for shape in t6Shapes do 
                  let t6b = dsharp.tensor( Util.arrayND shape (fun is -> double (Array.sum is) + 2.0))
                  let t6 = t6a * t6b
                  let t6Commute = t6b * t6a
                  yield (t6b, t6 ), (t6b, t6Commute ) |]
            |> Array.unzip

        let t6Expected =
            [|(dsharp.tensor 2.,                                                      dsharp.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (dsharp.tensor [2.],                                                    dsharp.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (dsharp.tensor [2., 3., 4., 5.],                                        dsharp.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (dsharp.tensor [[2.]],                                                  dsharp.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (dsharp.tensor [[2., 3., 4., 5.]],                                      dsharp.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (dsharp.tensor [[2.], [3.], [4.]],                                      dsharp.tensor [[[2., 4., 6., 8.], [15., 18., 21., 24.], [36., 40., 44., 48.]], [[26., 28., 30., 32.], [51., 54., 57., 60.], [84., 88., 92., 96.]]]);
              (dsharp.tensor [[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]],  dsharp.tensor [[[2., 6., 12., 20.], [15., 24., 35., 48.], [36., 50., 66., 84.]], [[26., 42., 60., 80.], [51., 72., 95., 120.], [84., 110., 138., 168.]]]);
              (dsharp.tensor [[[2.]]],                                                dsharp.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (dsharp.tensor [[[2., 3., 4., 5.]]],                                    dsharp.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (dsharp.tensor [[[2.], [3.], [4.]]],                                    dsharp.tensor [[[2., 4., 6., 8.], [15., 18., 21., 24.], [36., 40., 44., 48.]], [[26., 28., 30., 32.], [51., 54., 57., 60.], [84., 88., 92., 96.]]]);
              (dsharp.tensor [[[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]],dsharp.tensor [[[2., 6., 12., 20.], [15., 24., 35., 48.], [36., 50., 66., 84.]], [[26., 42., 60., 80.], [51., 72., 95., 120.], [84., 110., 138., 168.]]]);
              (dsharp.tensor [[[2.]], [[3.]]],                                        dsharp.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[39., 42., 45., 48.], [51., 54., 57., 60.], [63., 66., 69., 72.]]]);
              (dsharp.tensor [[[2., 3., 4., 5.]], [[3., 4., 5., 6.]]],                dsharp.tensor [[[2., 6., 12., 20.],  [10., 18., 28., 40.], [18., 30., 44., 60.]], [[39., 56., 75., 96.], [51., 72., 95., 120.], [63., 88., 115., 144.]]]);
              (dsharp.tensor [[[2.], [3.], [4.]], [[3.], [4.], [5.]]],                dsharp.tensor [[[2., 4., 6., 8.],  [15., 18., 21., 24.], [36., 40., 44., 48.]], [[39., 42., 45., 48.], [68., 72., 76., 80.], [105., 110., 115., 120.]]]); |]

        Assert.AreEqual(t6Expected, t6Results)
        Assert.AreEqual(t6Expected, t6CommuteResults)

    [<Test>]
    member this.TestTensorDivTT () =
        let t1 = dsharp.tensor([1.; 2.]) / dsharp.tensor([3.; 4.])
        let t1Correct = dsharp.tensor([0.333333; 0.5])

        let t2 = dsharp.tensor([1.; 2.]) / dsharp.tensor(5.)
        let t2Correct = dsharp.tensor([0.2; 0.4])

        let t3 = dsharp.tensor([1.; 2.]) / 5.
        let t3Correct = dsharp.tensor([0.2; 0.4])

        let t4 = 5. / dsharp.tensor([1.; 2.])
        let t4Correct = dsharp.tensor([5.; 2.5])

        Assert.True(t1.allclose(t1Correct, 0.01))
        Assert.True(t2.allclose(t2Correct, 0.01))
        Assert.True(t3.allclose(t3Correct, 0.01))
        Assert.True(t4.allclose(t4Correct, 0.01))

    [<Test>]
    member this.TestTensorPowTT () =
        let t1 = dsharp.tensor([1.; 2.]) ** dsharp.tensor([3.; 4.])
        let t1Correct = dsharp.tensor([1.; 16.])

        let t2 = dsharp.tensor([1.; 2.]) ** dsharp.tensor(5.)
        let t2Correct = dsharp.tensor([1.; 32.])

        let t3 = dsharp.tensor(5.) ** dsharp.tensor([1.; 2.])
        let t3Correct = dsharp.tensor([5.; 25.])

        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)

    [<Test>]
    member this.TestTensorMatMulT2T2 () =
        let t1 = dsharp.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
        let t2 = dsharp.tensor([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])

        let t3 = dsharp.matmul(t1, t2)
        let t3Correct = dsharp.tensor([[118.0367; 56.6266];
                                        [190.5926; 90.8155];
                                        [134.3925; 64.1030]])

        Assert.True(t3.allclose(t3Correct, 0.01))

    [<Test>]
    member this.TestTensorDot () =
        let t1 = dsharp.tensor([8.0766, 3.3030, -2.1732, 8.9448, 1.1028])
        let t2 = dsharp.tensor([5.1067, -0.0681, 7.4633, -3.6027, 9.0070])
        let t3 = dsharp.dot(t1, t2)
        let t3Correct = dsharp.tensor(2.5081)
        Assert.True(t3.allclose(t3Correct, 0.01))

    [<Test>]
    member this.TestTensorDiagonal () =
        let t1 = dsharp.arange(6.).view([2; 3])
        let t1a = dsharp.diagonal(t1)
        let t1b = dsharp.diagonal(t1, offset=1)
        let t1c = dsharp.diagonal(t1, offset=2)
        let t1d = dsharp.diagonal(t1, offset= -1)
        let t1aCorrect = dsharp.tensor([0.,4.])
        let t1bCorrect = dsharp.tensor([1.,5.])
        let t1cCorrect = dsharp.tensor([2.])
        let t1dCorrect = dsharp.tensor([3.])
        let t2 = dsharp.arange(9.).view([3;3])
        let t2a = dsharp.diagonal(t2)
        let t2aCorrect = dsharp.tensor([0.,4.,8.])
        Assert.AreEqual(t1aCorrect, t1a)
        Assert.AreEqual(t1bCorrect, t1b)
        Assert.AreEqual(t1cCorrect, t1c)
        Assert.AreEqual(t1dCorrect, t1d)
        Assert.AreEqual(t2aCorrect, t2a)

    [<Test>]
    member this.TestTensorTrace () =
        let t1 = dsharp.arange(6.).view([2; 3])
        let t1a = dsharp.trace(t1)
        let t1aCorrect = dsharp.tensor(4.)
        let t2 = dsharp.arange(9.).view([3;3])
        let t2a = dsharp.trace(t2)
        let t2aCorrect = dsharp.tensor(12.)
        Assert.AreEqual(t1aCorrect, t1a)
        Assert.AreEqual(t2aCorrect, t2a)

    [<Test>]
    member this.TestTensorConv1D () =
        let t1 = dsharp.tensor([[[0.3460; 0.4414; 0.2384; 0.7905; 0.2267];
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
        let t2 = dsharp.tensor([[[0.4941; 0.8710; 0.0606];
                                 [0.2831; 0.7930; 0.5602];
                                 [0.0024; 0.1236; 0.4394];
                                 [0.9086; 0.1277; 0.2450]];

                                [[0.5196; 0.1349; 0.0282];
                                 [0.1749; 0.6234; 0.5502];
                                 [0.7678; 0.0733; 0.3396];
                                 [0.6023; 0.6546; 0.3439]]])

        let t3 = t1.conv1d(t2)
        let t3Correct = dsharp.tensor([[[2.8516; 2.0732; 2.6420];
                                         [2.3239; 1.7078; 2.7450]];

                                        [[3.0127; 2.9651; 2.5219];
                                         [3.0899; 3.1496; 2.4110]];

                                        [[3.4749; 2.9038; 2.7131];
                                         [2.7692; 2.9444; 3.2554]]])

        let t3p1 = t1.conv1d(t2, padding=1)
        let t3p1Correct = dsharp.tensor([[[1.4392; 2.8516; 2.0732; 2.6420; 2.1177];
                                         [1.4345; 2.3239; 1.7078; 2.7450; 2.1474]];

                                        [[2.4208; 3.0127; 2.9651; 2.5219; 1.2960];
                                         [1.5544; 3.0899; 3.1496; 2.4110; 1.8567]];

                                        [[1.2965; 3.4749; 2.9038; 2.7131; 1.7408];
                                         [1.3549; 2.7692; 2.9444; 3.2554; 1.2120]]])

        let t3p2 = t1.conv1d(t2, padding=2)
        let t3p2Correct = dsharp.tensor([[[0.6333; 1.4392; 2.8516; 2.0732; 2.6420; 2.1177; 1.0258];
                                         [0.6539; 1.4345; 2.3239; 1.7078; 2.7450; 2.1474; 1.2585]];

                                        [[0.5982; 2.4208; 3.0127; 2.9651; 2.5219; 1.2960; 1.0620];
                                         [0.5157; 1.5544; 3.0899; 3.1496; 2.4110; 1.8567; 1.3182]];

                                        [[0.3165; 1.2965; 3.4749; 2.9038; 2.7131; 1.7408; 0.5275];
                                         [0.3861; 1.3549; 2.7692; 2.9444; 3.2554; 1.2120; 0.7428]]])

        let t3s2 = t1.conv1d(t2, stride=2)
        let t3s2Correct = dsharp.tensor([[[2.8516; 2.6420];
                                         [2.3239; 2.7450]];

                                        [[3.0127; 2.5219];
                                         [3.0899; 2.4110]];

                                        [[3.4749; 2.7131];
                                         [2.7692; 3.2554]]])

        let t3s3 = t1.conv1d(t2, stride=3)
        let t3s3Correct = dsharp.tensor([[[2.8516];
                                         [2.3239]];

                                        [[3.0127];
                                         [3.0899]];

                                        [[3.4749];
                                         [2.7692]]])

        let t3s2p1 = t1.conv1d(t2, stride=2, padding=1)
        let t3s2p1Correct = dsharp.tensor([[[1.4392; 2.0732; 2.1177];
                                             [1.4345; 1.7078; 2.1474]];

                                            [[2.4208; 2.9651; 1.2960];
                                             [1.5544; 3.1496; 1.8567]];

                                            [[1.2965; 2.9038; 1.7408];
                                             [1.3549; 2.9444; 1.2120]]])

        let t3s3p2 = t1.conv1d(t2, stride=3, padding=2)
        let t3s3p2Correct = dsharp.tensor([[[0.6333; 2.0732; 1.0258];
                                             [0.6539; 1.7078; 1.2585]];

                                            [[0.5982; 2.9651; 1.0620];
                                             [0.5157; 3.1496; 1.3182]];

                                            [[0.3165; 2.9038; 0.5275];
                                             [0.3861; 2.9444; 0.7428]]])
        
        let t3d2 = t1.conv1d(t2, dilation=2)
        let t3d2Correct = dsharp.tensor([[[2.8030];
                                         [2.4735]];

                                        [[2.9226];
                                         [3.1868]];

                                        [[2.8469];
                                         [2.4790]]])

        let t3p2d3 = t1.conv1d(t2, padding=2, dilation=3)
        let t3p2d3Correct = dsharp.tensor([[[2.1121; 0.8484; 2.2709];
                                             [1.6692; 0.5406; 1.8381]];

                                            [[2.5078; 1.2137; 0.9173];
                                             [2.2395; 1.1805; 1.1954]];

                                            [[1.5215; 1.3946; 2.1327];
                                             [1.0732; 1.3014; 2.0696]]])

        let t3s3p6d3 = t1.conv1d(t2, stride=3, padding=6, dilation=3)
        let t3s3p6d3Correct = dsharp.tensor([[[0.6333; 1.5018; 2.2709; 1.0580];
                                             [0.6539; 1.5130; 1.8381; 1.0479]];

                                            [[0.5982; 1.7459; 0.9173; 0.2709];
                                             [0.5157; 0.8537; 1.1954; 0.7027]];

                                            [[0.3165; 1.4118; 2.1327; 1.1949];
                                             [0.3861; 1.5697; 2.0696; 0.8520]]])

        let t3b1 = t1.[0].unsqueeze(0).conv1d(t2)
        let t3b1Correct = t3Correct.[0].unsqueeze(0)
        let t3b1s2 = t1.[0].unsqueeze(0).conv1d(t2, stride = 2)
        let t3b1s2Correct = t3s2Correct.[0].unsqueeze(0)

        Assert.True(t3.allclose(t3Correct, 0.01))
        Assert.True(t3p1.allclose(t3p1Correct, 0.01))
        Assert.True(t3p2.allclose(t3p2Correct, 0.01))
        Assert.True(t3s2.allclose(t3s2Correct, 0.01))
        Assert.True(t3s3.allclose(t3s3Correct, 0.01))
        Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01))
        Assert.True(t3s3p2.allclose(t3s3p2Correct, 0.01))
        Assert.True(t3d2.allclose(t3d2Correct, 0.01))
        Assert.True(t3p2d3.allclose(t3p2d3Correct, 0.01))
        Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01))
        Assert.True(t3b1.allclose(t3b1Correct, 0.01))
        Assert.True(t3b1s2.allclose(t3b1s2Correct, 0.01))

    [<Test>]
    member this.TestTensorConv2D () =
        let t1 = dsharp.tensor([[[[ 10.7072,  -5.0993,   3.6884,   2.0982],
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
        let t2 = dsharp.tensor([[[[-5.6745, -1.9422,  4.1369],
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

        let t3 = t1.conv2d(t2)
        let t3Correct = dsharp.tensor([[[[  10.6089;   -1.4459];
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

        let t3p1 = t1.conv2d(t2, padding=1)
        let t3p1Correct = dsharp.tensor([[[[  86.6988;    8.1164;  -85.8172;   69.5001];
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

        let t3p12 = t1.conv2d(t2, padding=[|1; 2|])
        let t3p12Correct = dsharp.tensor([[[[   7.5867;   86.6988;    8.1164;  -85.8172;   69.5001;  -35.4485];
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

        let t3s2 = t1.conv2d(t2, stride=2)
        let t3s2Correct = dsharp.tensor([[[[  10.6089]];

                                         [[  97.8425]];

                                         [[ 427.2891]]];


                                        [[[-127.6157]];

                                         [[ 104.2333]];

                                         [[-106.0468]]]])

        let t3s13 = t1.conv2d(t2, stride=[|1; 3|])
        let t3s13Correct = dsharp.tensor([[[[  10.6089];
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

        let t3s2p1 = t1.conv2d(t2, stride=2, padding=1)
        let t3s2p1Correct = dsharp.tensor([[[[  86.6988;  -85.8172];
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

        let t3s23p32 = t1.conv2d(t2, stride=[2; 3], padding=[3; 2])
        let t3s23p32Correct = dsharp.tensor([[[[   0.0000,    0.0000],
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
        
        let t3p1d2 = t1.conv2d(t2, padding=1, dilation=2)
        let t3p1d2Correct = dsharp.tensor([[[[ -72.7697,  -34.7305],
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

        let t3p22d23 = t1.conv2d(t2, padding=[2;2], dilation=[2;3])
        let t3p22d23Correct = dsharp.tensor([[[[-3.2693e+01, -4.3192e+01],
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

        let t3s3p6d3 = t1.conv2d(t2, stride=3, padding=6, dilation=3)
        let t3s3p6d3Correct = dsharp.tensor([[[[  78.0793,   88.7191,  -32.2774,   12.5512],
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

        let t3b1 = t1.[0].unsqueeze(0).conv2d(t2)
        let t3b1Correct = t3Correct.[0].unsqueeze(0)
        let t3b1s2 = t1.[0].unsqueeze(0).conv2d(t2, stride = 2)
        let t3b1s2Correct = t3s2Correct.[0].unsqueeze(0)

        // Assert.True(false)
        Assert.True(t3.allclose(t3Correct, 0.01))
        Assert.True(t3p1.allclose(t3p1Correct, 0.01))
        Assert.True(t3p12.allclose(t3p12Correct, 0.01))
        Assert.True(t3s2.allclose(t3s2Correct, 0.01))
        Assert.True(t3s13.allclose(t3s13Correct, 0.01))
        Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01))
        Assert.True(t3s23p32.allclose(t3s23p32Correct, 0.01))
        Assert.True(t3p1d2.allclose(t3p1d2Correct, 0.01))
        Assert.True(t3p22d23.allclose(t3p22d23Correct, 0.01))
        Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01))
        Assert.True(t3b1.allclose(t3b1Correct, 0.01))
        Assert.True(t3b1s2.allclose(t3b1s2Correct, 0.01))

    [<Test>]
    member this.TestTensorNegT () =
        let t1 = dsharp.tensor([1.; 2.; 3.])
        let t1Neg = -t1
        let t1NegCorrect = dsharp.tensor([-1.; -2.; -3.])

        Assert.AreEqual(t1NegCorrect, t1Neg)

    [<Test>]
    member this.TestTensorSumT () =
        let t1 = dsharp.tensor([1.; 2.; 3.])
        let t1Sum = t1.sum()
        let t1SumCorrect = dsharp.tensor(6.)

        let t2 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t2Sum = t2.sum()
        let t2SumCorrect = dsharp.tensor(10.)

        Assert.AreEqual(t1SumCorrect, t1Sum)
        Assert.AreEqual(t2SumCorrect, t2Sum)

    [<Test>]
    member this.TestTensorSumCollapseT () =
        let t1 = dsharp.tensor([1.; 2.; 3.])
        let t1Sum = t1.sumToSize([| |])
        let t1SumCorrect = dsharp.tensor(6.)

        Assert.AreEqual(t1SumCorrect, t1Sum)

        let t2 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t2Sum = t2.sumToSize([| |])
        let t2SumCorrect = dsharp.tensor(10.)

        Assert.AreEqual(t2SumCorrect, t2Sum)

        let t3 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t3Sum = t3.sumToSize([| 2 |])
        let t3SumCorrect = dsharp.tensor( [4.; 6.] )

        Assert.AreEqual(t3SumCorrect, t3Sum)

        let t4 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t4Sum = t4.sumToSize([| 1; 2 |])
        let t4SumCorrect = dsharp.tensor( [ [4.; 6.] ] )

        Assert.AreEqual(t4SumCorrect, t4Sum)

        let t5 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t5Sum = t5.sumToSize([| 2; 1 |])
        let t5SumCorrect = dsharp.tensor( [ [3.]; [7.] ] )

        Assert.AreEqual(t5SumCorrect, t5Sum)

        // Systematically test all legitimate reductions of 2x2x2 to smaller sizes
        let t6 = dsharp.tensor([ [[1.; 2.]; [3.; 4.] ]; [[5.; 6.]; [7.; 8.] ] ])
        let systematicResults = 
            [| for i1 in 0..2 do 
                  for i2 in (if i1 = 0 then 0 else 1)..2 do
                     for i3 in (if i2 = 0 then 0 else 1)..2 do
                        let newShape = 
                            [| if i1 > 0 then yield i1
                               if i2 > 0 then yield i2
                               if i3 > 0 then yield i3 |]
                        yield (newShape, t6.sumToSize(newShape)) |]
        
        let expectedResults = 
            [|([||], dsharp.tensor 36.);
              ([|1|], dsharp.tensor [36.]);
              ([|2|], dsharp.tensor [16.; 20.]);
              ([|1; 1|], dsharp.tensor [[36.]]);
              ([|1; 2|], dsharp.tensor [[16.; 20.]]);
              ([|2; 1|], dsharp.tensor [[14.]; [22.]]);
              ([|2; 2|], dsharp.tensor [[6.; 8.]; [10.; 12.]]);
              ([|1; 1; 1|], dsharp.tensor [[[36.]]]);
              ([|1; 1; 2|], dsharp.tensor [[[16.; 20.]]]);
              ([|1; 2; 1|], dsharp.tensor [[[14.]; [22.]]]);
              ([|1; 2; 2|], dsharp.tensor [[[6.; 8.]; [10.; 12.]]]);
              ([|2; 1; 1|], dsharp.tensor [[[10.]]; [[26.]]]);
              ([|2; 1; 2|], dsharp.tensor [[[4.; 6.]]; [[12.; 14.]]]);
              ([|2; 2; 1|], dsharp.tensor [[[3.]; [7.]]; [[11.]; [15.]]]);
              ([|2; 2; 2|], dsharp.tensor [[[1.; 2.]; [3.; 4.]]; [[5.; 6.]; [7.; 8.]]])|]

        Assert.AreEqual(systematicResults, expectedResults)

    [<Test>]
    member this.TestTensorSumT2Dim0 () =
        let t1 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t1Sum = t1.sumT2Dim0()
        let t1SumCorrect = dsharp.tensor([4.; 6.])

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
        let t = dsharp.tensor([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
        let tSum0 = t.sum(0)
        let tSum0Correct = dsharp.tensor([[14., 16., 18., 20.], [22., 24., 26., 28.], [30., 32., 34., 36.]])
        let tSum1 = t.sum(1)
        let tSum1Correct = dsharp.tensor([[15., 18., 21., 24.], [51., 54., 57., 60.]])
        let tSum2 = t.sum(2)
        let tSum2Correct = dsharp.tensor([[10., 26., 42.], [58., 74., 90.]])

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
        let t = dsharp.tensor([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tSum0 = t.sum(0, keepDim=true)
        let tSum0Correct = dsharp.tensor([[[14.; 16.; 18.; 20.]; [22.; 24.; 26.; 28.]; [30.; 32.; 34.; 36.]]])
        let tSum1 = t.sum(1, keepDim=true)
        let tSum1Correct = dsharp.tensor([[[15.; 18.; 21.; 24.]]; [[51.; 54.; 57.; 60.]]])
        let tSum2 = t.sum(2, keepDim=true)
        let tSum2Correct = dsharp.tensor([[[10.]; [26.]; [42.]]; [[58.]; [74.]; [90.]]])

        Assert.AreEqual(tSum0Correct, tSum0)
        Assert.AreEqual(tSum1Correct, tSum1)
        Assert.AreEqual(tSum2Correct, tSum2)

    [<Test>]
    member this.TestTensorMean () =
        let t = dsharp.tensor([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(12.5)

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
        let tMean0 = t.mean(0)
        let tMean0Correct = dsharp.tensor([[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]])
        let tMean1 = t.mean(1)
        let tMean1Correct = dsharp.tensor([[5.; 6.; 7.; 8.]; [17.; 18.; 19.; 20.]])
        let tMean2 = t.mean(2)
        let tMean2Correct = dsharp.tensor([[2.5; 6.5; 10.5]; [14.5; 18.5; 22.5]])

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
        let tMeanKeepDim0 = t.mean(0, keepDim=true)
        let tMeanKeepDim0Correct = dsharp.tensor([[[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]]])
        let tMeanKeepDim1 = t.mean(1, keepDim=true)
        let tMeanKeepDim1Correct = dsharp.tensor([[[5.; 6.; 7.; 8.]]; [[17.; 18.; 19.; 20.]]])
        let tMeanKeepDim2 = t.mean(2, keepDim=true)
        let tMeanKeepDim2Correct = dsharp.tensor([[[2.5]; [6.5]; [10.5]]; [[14.5]; [18.5]; [22.5]]])

        Assert.AreEqual(tMeanKeepDim0, tMeanKeepDim0Correct)
        Assert.AreEqual(tMeanKeepDim1, tMeanKeepDim1Correct)
        Assert.AreEqual(tMeanKeepDim2, tMeanKeepDim2Correct)

    [<Test>]
    member this.TestTensorStddev () =
        let t = dsharp.tensor([[[0.3787;0.7515;0.2252;0.3416];
          [0.6078;0.4742;0.7844;0.0967];
          [0.1416;0.1559;0.6452;0.1417]];
 
         [[0.0848;0.4156;0.5542;0.4166];
          [0.5187;0.0520;0.4763;0.1509];
          [0.4767;0.8096;0.1729;0.6671]]])
        let tStddev = t.stddev()
        let tStddevCorrect = dsharp.tensor(0.2398)

        Assert.True(tStddev.allclose(tStddevCorrect, 0.01))

        // stddev, dim={0,1,2,3}, keepDim=true
        let tStddev0 = t.stddev(0)
        let tStddev0Correct = dsharp.tensor([[0.2078; 0.2375; 0.2326; 0.0530];
         [0.0630; 0.2985; 0.2179; 0.0383];
         [0.2370; 0.4623; 0.3339; 0.3715]])
        let tStddev1 = t.stddev(1)
        let tStddev1Correct = dsharp.tensor([[0.2331; 0.2981; 0.2911; 0.1304];
         [0.2393; 0.3789; 0.2014; 0.2581]])
        let tStddev2 = t.stddev(2)
        let tStddev2Correct = dsharp.tensor([[0.2277; 0.2918; 0.2495];
         [0.1996; 0.2328; 0.2753]])

        Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
        Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
        Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))

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
        let tStddev0 = t.stddev(0, keepDim=true)
        let tStddev0Correct = dsharp.tensor([[[0.2078; 0.2375; 0.2326; 0.0530];[0.0630; 0.2985; 0.2179; 0.0383];[0.2370; 0.4623; 0.3339; 0.3715]]])
        let tStddev1 = t.stddev(1, keepDim=true)
        let tStddev1Correct = dsharp.tensor([[[0.2331; 0.2981; 0.2911; 0.1304]];[[0.2393; 0.3789; 0.2014; 0.2581]]])
        let tStddev2 = t.stddev(2, keepDim=true)
        let tStddev2Correct = dsharp.tensor([[[0.2277]; [0.2918]; [0.2495]];[[0.1996]; [0.2328]; [0.2753]]])

        Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
        Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
        Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))

    [<Test>]
    member this.TestTensorVariance () =
        (* Python:
        import torch
        input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
        input.var()
        *)
        let t = dsharp.tensor([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
        let tVariance = t.variance()
        let tVarianceCorrect = dsharp.tensor(0.0575)

        Assert.True(tVariance.allclose(tVarianceCorrect, 0.01))

        // Variance, dim={0,1,2,3}
        (* Python:
        input.var(0)
        # --> tensor([[0.0432, 0.0564, 0.0541, 0.0028],[0.0040, 0.0891, 0.0475, 0.0015],[0.0561, 0.2137, 0.1115, 0.1380]])
        input.var(1)
        # --> tensor([[0.0543, 0.0888, 0.0847, 0.0170],[0.0573, 0.1436, 0.0406, 0.0666]])
        input.var(2)
        # --> tensor([[0.0519, 0.0852, 0.0622],[0.0398, 0.0542, 0.0758]])
        *)
        let tVariance0 = t.variance(0)
        let tVariance0Correct = dsharp.tensor([[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]])
        let tVariance1 = t.variance(1)
        let tVariance1Correct = dsharp.tensor([[0.0543; 0.0888; 0.0847; 0.0170];[0.0573; 0.1436; 0.0406; 0.0666]])
        let tVariance2 = t.variance(2)
        let tVariance2Correct = dsharp.tensor([[0.0519; 0.0852; 0.0622];[0.0398; 0.0542; 0.0758]])

        Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
        Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
        Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))

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
        let tVariance0 = t.variance(0, keepDim=true)
        let tVariance0Correct = dsharp.tensor([[[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]]])
        let tVariance1 = t.variance(1, keepDim=true)
        let tVariance1Correct = dsharp.tensor([[[0.0543; 0.0888; 0.0847; 0.0170]];[[0.0573; 0.1436; 0.0406; 0.0666]]])
        let tVariance2 = t.variance(2, keepDim=true)
        let tVariance2Correct = dsharp.tensor([[[0.0519];[0.0852];[0.0622]];[[0.0398];[0.0542];[0.0758]]])

        Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
        Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
        Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))

    [<Test>]
    member this.TestTensorTransposeT2 () =
        let t1 = dsharp.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]])
        let t1Transpose = t1.transpose()
        let t1TransposeCorrect = dsharp.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])

        let t2 = dsharp.tensor([[1.; 2.]; [3.; 4.]])
        let t2TransposeTranspose = t2.transpose().transpose()
        let t2TransposeTransposeCorrect = t2

        Assert.AreEqual(t1TransposeCorrect, t1Transpose)
        Assert.AreEqual(t2TransposeTransposeCorrect, t2TransposeTranspose)

    [<Test>]
    member this.TestTensorSignT () =
        let t1 = dsharp.tensor([-1.; -2.; 0.; 3.])
        let t1Sign = t1.sign()
        let t1SignCorrect = dsharp.tensor([-1.; -1.; 0.; 1.])

        Assert.AreEqual(t1SignCorrect, t1Sign)

    [<Test>]
    member this.TestTensorFloorT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Floor = t1.floor()
        let t1FloorCorrect = dsharp.tensor([0.; 0.; 0.; 0.; 0.])

        Assert.True(t1Floor.allclose(t1FloorCorrect, 0.01))

    [<Test>]
    member this.TestTensorCeilT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Ceil = t1.ceil()
        let t1CeilCorrect = dsharp.tensor([1.; 1.; 1.; 1.; 1.])

        Assert.True(t1Ceil.allclose(t1CeilCorrect, 0.01))

    [<Test>]
    member this.TestTensorRoundT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Round = t1.round()
        let t1RoundCorrect = dsharp.tensor([1.; 0.; 0.; 1.; 1.])

        Assert.True(t1Round.allclose(t1RoundCorrect, 0.01))

    [<Test>]
    member this.TestTensorAbsT () =
        let t1 = dsharp.tensor([-1.; -2.; 0.; 3.])
        let t1Abs = t1.abs()
        let t1AbsCorrect = dsharp.tensor([1.; 2.; 0.; 3.])

        Assert.AreEqual(t1AbsCorrect, t1Abs)

    [<Test>]
    member this.TestTensorReluT () =
        let t1 = dsharp.tensor([-1.; -2.; 0.; 3.; 10.])
        let t1Relu = t1.relu()
        let t1ReluCorrect = dsharp.tensor([0.; 0.; 0.; 3.; 10.])

        Assert.AreEqual(t1ReluCorrect, t1Relu)

    [<Test>]
    member this.TestTensorLeakyRelu () =
        let t1 = dsharp.tensor([-1.; -2.; 0.; 3.; 10.])
        let t1LeakyRelu = t1.leakyRelu()
        let t1LeakyReluCorrect = dsharp.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])

        Assert.AreEqual(t1LeakyReluCorrect, t1LeakyRelu)

    [<Test>]
    member this.TestTensorSigmoidT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Sigmoid = t1.sigmoid()
        let t1SigmoidCorrect = dsharp.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])

        Assert.True(t1Sigmoid.allclose(t1SigmoidCorrect, 0.01))

    [<Test>]
    member this.TestTensorSoftplusT () =
        let t1 = dsharp.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05])
        let t1Softplus = t1.softplus()
        let t1SoftplusCorrect = dsharp.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])

        Assert.True(t1Softplus.allclose(t1SoftplusCorrect, 0.01))

    [<Test>]
    member this.TestTensorExpT () =
        let t1 = dsharp.tensor([0.9139; -0.5907;  1.9422; -0.7763; -0.3274])
        let t1Exp = t1.exp()
        let t1ExpCorrect = dsharp.tensor([2.4940; 0.5539; 6.9742; 0.4601; 0.7208])

        Assert.True(t1Exp.allclose(t1ExpCorrect, 0.01))

    [<Test>]
    member this.TestTensorLogT () =
        let t1 = dsharp.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
        let t1Log = t1.log()
        let t1LogCorrect = dsharp.tensor([-2.0516; -0.5426; -0.4301; -0.9727; -0.9100])

        Assert.True(t1Log.allclose(t1LogCorrect, 0.01))

    [<Test>]
    member this.TestTensorLog10T () =
        let t1 = dsharp.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
        let t1Log10 = t1.log10()
        let t1Log10Correct = dsharp.tensor([-0.8911; -0.2357; -0.1868; -0.4224; -0.3952])

        Assert.True(t1Log10.allclose(t1Log10Correct, 0.01))

    [<Test>]
    member this.TestTensorSqrtT () =
        let t1 = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
        let t1Sqrt = t1.sqrt()
        let t1SqrtCorrect = dsharp.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])

        Assert.True(t1Sqrt.allclose(t1SqrtCorrect, 0.01))

    [<Test>]
    member this.TestTensorSinT () =
        let t1 = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
        let t1Sin = t1.sin()
        let t1SinCorrect = dsharp.tensor([-0.9828;  0.9991; -0.3698; -0.7510;  0.9491])

        Assert.True(t1Sin.allclose(t1SinCorrect, 0.01))

    [<Test>]
    member this.TestTensorCosT () =
        let t1 = dsharp.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
        let t1Cos = t1.cos()
        let t1CosCorrect = dsharp.tensor([-0.1849;  0.0418; -0.9291;  0.6603;  0.3150])

        Assert.True(t1Cos.allclose(t1CosCorrect, 0.01))

    [<Test>]
    member this.TestTensorTanT () =
        let t1 = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Tan = t1.tan()
        let t1TanCorrect = dsharp.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])

        Assert.True(t1Tan.allclose(t1TanCorrect, 0.01))

    [<Test>]
    member this.TestTensorSinhT () =
        let t1 = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Sinh = t1.sinh()
        let t1SinhCorrect = dsharp.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])

        Assert.True(t1Sinh.allclose(t1SinhCorrect, 0.01))

    [<Test>]
    member this.TestTensorCoshT () =
        let t1 = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Cosh = t1.cosh()
        let t1CoshCorrect = dsharp.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])

        Assert.True(t1Cosh.allclose(t1CoshCorrect, 0.01))

    [<Test>]
    member this.TestTensorTanhT () =
        let t1 = dsharp.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
        let t1Tanh = t1.tanh()
        let t1TanhCorrect = dsharp.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])

        Assert.True(t1Tanh.allclose(t1TanhCorrect, 0.01))

    [<Test>]
    member this.TestTensorAsinT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Asin = t1.asin()
        let t1AsinCorrect = dsharp.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])

        Assert.True(t1Asin.allclose(t1AsinCorrect, 0.01))

    [<Test>]
    member this.TestTensorAcosT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Acos = t1.acos()
        let t1AcosCorrect = dsharp.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])

        Assert.True(t1Acos.allclose(t1AcosCorrect, 0.01))

    [<Test>]
    member this.TestTensorAtanT () =
        let t1 = dsharp.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
        let t1Atan = t1.atan()
        let t1AtanCorrect = dsharp.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])

        Assert.True(t1Atan.allclose(t1AtanCorrect, 0.01))

    [<Test>]
    member this.TestTensorSlice () =
        let t1 = dsharp.tensor([1.;2.])
        let t1s1 = t1.[0]
        let t1s2 = t1.[*]
        let t1s1Correct = dsharp.tensor(1.)
        let t1s2Correct = dsharp.tensor([1.;2.])

        let t2 = dsharp.tensor([[1.;2.];[3.;4.]])
        let t2s1 = t2.[0]
        let t2s2 = t2.[*]
        let t2s3 = t2.[0,0]
        let t2s4 = t2.[0,*]
        let t2s5 = t2.[*,0]
        let t2s6 = t2.[*,*]
        let t2s1Correct = dsharp.tensor([1.;2.])
        let t2s2Correct = dsharp.tensor([[1.;2.];[3.;4.]])
        let t2s3Correct = dsharp.tensor(1.)
        let t2s4Correct = dsharp.tensor([1.;2.])
        let t2s5Correct = dsharp.tensor([1.;3.])
        let t2s6Correct = dsharp.tensor([[1.;2.];[3.;4.]])

        let t2b = dsharp.tensor([[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]])
        let t2bs1 = t2b.[1..,2..]
        let t2bs1Correct = dsharp.tensor([[7.;8.];[11.;12.]])
        let t2bs2 = t2b.[1..2,2..3]
        let t2bs2Correct = dsharp.tensor([[7.;8.];[11.;12.]])

        let t3 = dsharp.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
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
        let t3s1Correct  = dsharp.tensor([[1.;2.];[3.;4.]])
        let t3s2Correct  = dsharp.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
        let t3s3Correct  = dsharp.tensor([1.;2.])
        let t3s4Correct  = dsharp.tensor([[1.;2.];[3.;4.]])
        let t3s5Correct  = dsharp.tensor([[1.;2.];[5.;6.]])
        let t3s6Correct  = dsharp.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
        let t3s7Correct  = dsharp.tensor(1.)
        let t3s8Correct  = dsharp.tensor([1.;2.])
        let t3s9Correct  = dsharp.tensor([1.;3.])
        let t3s10Correct = dsharp.tensor([[1.;2.];[3.;4.]])
        let t3s11Correct = dsharp.tensor([1.;5.])
        let t3s12Correct = dsharp.tensor([[1.;2.];[5.;6.]])
        let t3s13Correct = dsharp.tensor([[1.;3.];[5.;7.]])
        let t3s14Correct = dsharp.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])

        let t4 = dsharp.tensor([[[[1.]]; 
                                 [[2.]]; 
                                 [[3.]]]; 
                                [[[4.]]; 
                                 [[5.]]; 
                                 [[6.]]]])
        let t4s1 = t4.[0]
        let t4s2 = t4.[0,*,*,*]
        let t4s1Correct = dsharp.tensor([[[1]];
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
        let t1 = dsharp.tensor([[-0.2754;  0.0172;  0.7105];
            [-0.1890;  1.7664;  0.5377];
            [-0.5313; -2.2530; -0.6235];
            [ 0.6776;  1.5844; -0.5686]])
        let t2 = dsharp.tensor([[-111.8892;   -7.0328];
            [  18.7557;  -86.2308]])
        let t3 = t1.addSlice([0;1], t2)
        let t3Correct = dsharp.tensor([[  -0.2754; -111.8720;   -6.3222];
            [  -0.1890;   20.5221;  -85.6932];
            [  -0.5313;   -2.2530;   -0.6235];
            [   0.6776;    1.5844;   -0.5686]])

        Assert.True(t3.allclose(t3Correct, 0.01))

    [<Test>]
    member this.TestTensorExpandT () =
        let t1 = dsharp.tensor(1.0)
        let t1Expand = t1.expand([2;3])
        let t1ExpandCorrect = dsharp.tensor([[1.;1.;1.];[1.;1.;1.]])
        Assert.AreEqual(t1ExpandCorrect, t1Expand)

        let t2 = dsharp.tensor([1.0])
        let t2Expand = t2.expand([2;3])
        let t2ExpandCorrect = dsharp.tensor([[1.;1.;1.];[1.;1.;1.]])

        Assert.AreEqual(t2ExpandCorrect, t2Expand)

        let t3 = dsharp.tensor([1.; 2.]) // 2
        let t3Expand = t3.expand([3;2]) // 3x2
        let t3ExpandCorrect = dsharp.tensor([[1.;2.];[1.;2.];[1.;2.]]) // 3x2

        Assert.AreEqual(t3ExpandCorrect, t3Expand)

        let t4 = dsharp.tensor([[1.]; [2.]]) // 2x1
        let t4Expand = t4.expand([2;2]) // 2x2
        let t4ExpandCorrect = dsharp.tensor([[1.;1.];[2.;2.]])

        Assert.AreEqual(t4ExpandCorrect, t4Expand)

        let t5 = dsharp.tensor([[1.]; [2.]]) // 2x1
        let t5Expand = t5.expand([2;2;2]) // 2x2x2
        let t5ExpandCorrect = dsharp.tensor([[[1.;1.];[2.;2.]];[[1.;1.];[2.;2.]]])

        Assert.AreEqual(t5ExpandCorrect, t5Expand)

    [<Test>]
    member this.TestTensorSqueezeT () =
        let t1 = dsharp.tensor([[[1.; 2.]]; [[3.;4.]]])
        let t1Squeeze = t1.squeeze()
        let t1SqueezeCorrect = dsharp.tensor([[1.;2.];[3.;4.]])

        Assert.True(t1Squeeze.allclose(t1SqueezeCorrect, 0.01))

    [<Test>]
    member this.TestTensorUnsqueezeT () =
        let t1 = dsharp.tensor([[1.;2.];[3.;4.]])
        let t1Unsqueeze = t1.unsqueeze(1)
        let t1UnsqueezeCorrect = dsharp.tensor([[[1.;2.]]; [[3.;4.]]])

        Assert.True(t1Unsqueeze.allclose(t1UnsqueezeCorrect, 0.01))

    [<Test>]
    member this.TestTensorFlipT () =
        let t1 = dsharp.tensor([[1.;2.];[3.;4.]])
        let t2 = t1.flip([|0|])
        let t2Correct = dsharp.tensor([[3.;4.]; [1.;2.]])
        let t3 = t1.flip([|1|])
        let t3Correct = dsharp.tensor([[2.;1.]; [4.;3.]])
        let t4 = t1.flip([|0; 1|])
        let t4Correct = dsharp.tensor([[4.;3.]; [2.;1.]])
        let t5 = t1.flip([|0; 1|]).flip([|0; 1|])
        let t5Correct = dsharp.tensor([[1.;2.]; [3.;4.]])

        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t4Correct, t4)
        Assert.AreEqual(t5Correct, t5)

    [<Test>]
    member this.TestTensorDilateT () =
        let t1 = dsharp.tensor([[1.;2.]; [3.;4.]])
        let t2 = t1.dilate([|1; 2|])
        let t2Correct = dsharp.tensor([[1.;0.;2.];[3.;0.;4.]])
        let t3 = t1.dilate([|2; 2|])
        let t3Correct = dsharp.tensor([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])
        let t4 = dsharp.tensor([1.;2.;3.;4.])
        let t5 = t4.dilate([|3|])
        let t5Correct = dsharp.tensor([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])

        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)
        Assert.AreEqual(t5Correct, t5)

    [<Test>]
    member this.TestTensorUndilateT () =
        let t1 = dsharp.tensor([[1.;0.;2.];[3.;0.;4.]])
        let t2 = t1.undilate([|1; 2|])
        let t2Correct = dsharp.tensor([[1.;2.]; [3.;4.]])
        let t3 = dsharp.tensor([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])
        let t4 = t3.undilate([|2; 2|])
        let t4Correct = dsharp.tensor([[1.;2.]; [3.;4.]])
        let t5 = dsharp.tensor([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])
        let t6 = t5.undilate([|3|])
        let t6Correct = dsharp.tensor([1.;2.;3.;4.])

        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t4Correct, t4)
        Assert.AreEqual(t6Correct, t6)

    [<Test>]
    member this.TestTensorView () =
        let t = dsharp.rand([10;10])
        let t1Shape = t.view(-1).shape
        let t1ShapeCorrect = [|100|]
        let t2Shape = t.view([-1;50]).shape
        let t2ShapeCorrect = [|2;50|]
        let t3Shape = t.view([2;-1;50]).shape
        let t3ShapeCorrect = [|2;1;50|]
        let t4Shape = t.view([2;-1;10]).shape
        let t4ShapeCorrect = [|2;5;10|]
        
        Assert.AreEqual(t1ShapeCorrect, t1Shape)
        Assert.AreEqual(t2ShapeCorrect, t2Shape)
        Assert.AreEqual(t3ShapeCorrect, t3Shape)
        Assert.AreEqual(t4ShapeCorrect, t4Shape)

    [<Test>]
    member this.TestTensorFlatten () =
        let t1 = dsharp.rand([5;5;5;5])
        let t1f1shape = dsharp.flatten(t1).shape
        let t1f1shapeCorrect = [|625|]
        let t1f2shape = dsharp.flatten(t1, startDim=1).shape
        let t1f2shapeCorrect = [|5; 125|]
        let t1f3shape = dsharp.flatten(t1, startDim=1, endDim=2).shape
        let t1f3shapeCorrect = [|5; 25; 5|]

        let t2 = dsharp.rand(5)
        let t2fshape = dsharp.flatten(t2).shape
        let t2fshapeCorrect = [|5|]

        let t3 = dsharp.tensor(2.5)
        let t3fshape = dsharp.flatten(t3).shape
        let t3fshapeCorrect = [||]

        Assert.AreEqual(t1f1shapeCorrect, t1f1shape)
        Assert.AreEqual(t1f2shapeCorrect, t1f2shape)
        Assert.AreEqual(t1f3shapeCorrect, t1f3shape)
        Assert.AreEqual(t2fshapeCorrect, t2fshape)
        Assert.AreEqual(t3fshapeCorrect, t3fshape)

    [<Test>]
    member this.TestTensorMax () =
        let t1 = dsharp.tensor([4.;1.;20.;3.])
        let t1Max = t1.max()
        let t1MaxCorrect = dsharp.tensor(20.)

        let t2 = dsharp.tensor([[1.;4.];[2.;3.]])
        let t2Max = t2.max()
        let t2MaxCorrect = dsharp.tensor(4.)

        let t3 = dsharp.tensor([[[ 7.6884; 65.9125;  4.0114];
             [46.7944; 61.5331; 40.1627];
             [48.3240;  4.9910; 50.1571]];

            [[13.4777; 65.7656; 36.8161];
             [47.8268; 42.2229;  5.6115];
             [43.4779; 77.8675; 95.7660]];

            [[59.8422; 47.1146; 36.7614];
             [71.6328; 18.5912; 27.7328];
             [49.9120; 60.3023; 53.0838]]])
        let t3Max = t3.max()
        let t3MaxCorrect = dsharp.tensor(95.7660)
        
        let t4 = dsharp.tensor([[[[8.8978; 8.0936];
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
        let t4Max = t4.max()
        let t4MaxCorrect = dsharp.tensor(9.7456)

        Assert.AreEqual(t1MaxCorrect, t1Max)
        Assert.AreEqual(t2MaxCorrect, t2Max)
        Assert.AreEqual(t3MaxCorrect, t3Max)
        Assert.AreEqual(t4MaxCorrect, t4Max)


    [<Test>]
    member this.TestTensorMin () =
        let t1 = dsharp.tensor([4.;1.;20.;3.])
        let t1Min = t1.min()
        let t1MinCorrect = dsharp.tensor(1.)

        let t2 = dsharp.tensor([[1.;4.];[2.;3.]])
        let t2Min = t2.min()
        let t2MinCorrect = dsharp.tensor(1.)

        let t3 = dsharp.tensor([[[ 7.6884; 65.9125;  4.0114];
             [46.7944; 61.5331; 40.1627];
             [48.3240;  4.9910; 50.1571]];

            [[13.4777; 65.7656; 36.8161];
             [47.8268; 42.2229;  5.6115];
             [43.4779; 77.8675; 95.7660]];

            [[59.8422; 47.1146; 36.7614];
             [71.6328; 18.5912; 27.7328];
             [49.9120; 60.3023; 53.0838]]])
        let t3Min = t3.min()
        let t3MinCorrect = dsharp.tensor(4.0114)
       
        let t4 = dsharp.tensor([[[[8.8978; 8.0936];
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
        let t4Min = t4.min()
        let t4MinCorrect = dsharp.tensor(0.5370)

        Assert.AreEqual(t1MinCorrect, t1Min)
        Assert.AreEqual(t2MinCorrect, t2Min)
        Assert.AreEqual(t3MinCorrect, t3Min)
        Assert.AreEqual(t4MinCorrect, t4Min)

    [<Test>]
    member this.TestTensorMaxBinary () =
        let t1 = dsharp.tensor([[-4.9385; 12.6206; 10.1783];
            [-2.9624; 17.6992;  2.2506];
            [-2.3536;  8.0772; 13.5639]])
        let t2 = dsharp.tensor([[  0.7027;  22.3251; -11.4533];
            [  3.6887;   4.3355;   3.3767];
            [  0.1203;  -5.4088;   1.5658]])
        let t3 = dsharp.max(t1, t2)
        let t3Correct = dsharp.tensor([[ 0.7027; 22.3251; 10.1783];
            [ 3.6887; 17.6992;  3.3767];
            [ 0.1203;  8.0772; 13.5639]])

        Assert.True(t3.allclose(t3Correct, 0.01))

    [<Test>]
    member this.TestTensorMinBinary () =
        let t1 = dsharp.tensor([[-4.9385; 12.6206; 10.1783];
            [-2.9624; 17.6992;  2.2506];
            [-2.3536;  8.0772; 13.5639]])
        let t2 = dsharp.tensor([[  0.7027;  22.3251; -11.4533];
            [  3.6887;   4.3355;   3.3767];
            [  0.1203;  -5.4088;   1.5658]])
        let t3 = dsharp.min(t1, t2)
        let t3Correct = dsharp.tensor([[ -4.9385;  12.6206; -11.4533];
            [ -2.9624;   4.3355;   2.2506];
            [ -2.3536;  -5.4088;   1.5658]])

        Assert.True(t3.allclose(t3Correct, 0.01))

    [<Test>]
    member this.TestTensorSoftmax () =
        let t1 = dsharp.tensor([2.7291; 0.0607; 0.8290])
        let t1Softmax0 = t1.softmax(0)
        let t1Softmax0Correct = dsharp.tensor([0.8204; 0.0569; 0.1227])

        let t2 = dsharp.tensor([[1.3335; 1.6616; 2.4874; 6.1722];
            [3.3478; 9.3019; 1.0844; 8.9874];
            [8.6300; 1.8842; 9.1387; 9.1321]])
        let t2Softmax0 = t2.softmax(0)
        let t2Softmax0Correct = dsharp.tensor([[6.7403e-04; 4.8014e-04; 1.2904e-03; 2.7033e-02];
            [5.0519e-03; 9.9892e-01; 3.1723e-04; 4.5134e-01];
            [9.9427e-01; 5.9987e-04; 9.9839e-01; 5.2163e-01]])
        let t2Softmax1 = t2.softmax(1)
        let t2Softmax1Correct = dsharp.tensor([[7.5836e-03; 1.0528e-02; 2.4044e-02; 9.5784e-01];
            [1.4974e-03; 5.7703e-01; 1.5573e-04; 4.2131e-01];
            [2.3167e-01; 2.7240e-04; 3.8528e-01; 3.8277e-01]])

        let t3 = dsharp.tensor([[[3.0897; 2.0902];
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
             
        let t3Softmax0 = t3.softmax(0)
        let t3Softmax0Correct = dsharp.tensor([[[2.4662e-03; 3.7486e-03];
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
        let t3Softmax1 = t3.softmax(1)
        let t3Softmax1Correct = dsharp.tensor([[[1.8050e-01; 1.2351e-03];
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
        let t3Softmax2 = t3.softmax(2)
        let t3Softmax2Correct = dsharp.tensor([[[7.3096e-01; 2.6904e-01];
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

        Assert.True(t1Softmax0.allclose(t1Softmax0Correct, 0.001))
        Assert.True(t2Softmax0.allclose(t2Softmax0Correct, 0.001))
        Assert.True(t2Softmax1.allclose(t2Softmax1Correct, 0.001))
        Assert.True(t3Softmax0.allclose(t3Softmax0Correct, 0.001))
        Assert.True(t3Softmax1.allclose(t3Softmax1Correct, 0.001))
        Assert.True(t3Softmax2.allclose(t3Softmax2Correct, 0.001))


    [<Test>]
    member this.TestTensorLogsoftmax () =
        let t1 = dsharp.tensor([2.7291, 0.0607, 0.8290])
        let t1Logsoftmax0 = t1.logsoftmax(0)
        let t1Logsoftmax0Correct = dsharp.tensor([-0.1980, -2.8664, -2.0981])

        let t2 = dsharp.tensor([[1.3335, 1.6616, 2.4874, 6.1722],
                                [3.3478, 9.3019, 1.0844, 8.9874],
                                [8.6300, 1.8842, 9.1387, 9.1321]])
        let t2Logsoftmax0 = t2.logsoftmax(0)
        let t2Logsoftmax0Correct = dsharp.tensor([[-7.3022e+00, -7.6414e+00, -6.6529e+00, -3.6107e+00],
                                                    [-5.2879e+00, -1.0806e-03, -8.0559e+00, -7.9552e-01],
                                                    [-5.7426e-03, -7.4188e+00, -1.6088e-03, -6.5082e-01]])
        let t2Logsoftmax1 = t2.logsoftmax(1)
        let t2Logsoftmax1Correct = dsharp.tensor([[-4.8818, -4.5537, -3.7279, -0.0431],
                                                    [-6.5040, -0.5499, -8.7674, -0.8644],
                                                    [-1.4624, -8.2082, -0.9537, -0.9603]])

        let t3 = dsharp.tensor([[[3.0897, 2.0902],
                                 [2.4055, 1.2437],
                                 [2.1253, 8.7802],
                                 [4.3856, 3.4456]],

                                [[8.6233, 6.9789],
                                 [4.9583, 9.9497],
                                 [2.6964, 1.6048],
                                 [2.1182, 2.1071]],

                                [[8.1097, 6.9804],
                                 [8.1223, 6.3030],
                                 [0.1873, 8.7840],
                                 [9.3609, 0.6493]]])
             
        let t3Logsoftmax0 = t3.logsoftmax(0)
        let t3Logsoftmax0Correct = dsharp.tensor([[[-6.0050e+00, -5.5864e+00],
                                                     [-5.7613e+00, -8.7319e+00],
                                                     [-1.0696e+00, -6.9543e-01],
                                                     [-4.9829e+00, -2.8011e-01]],

                                                    [[-4.7143e-01, -6.9765e-01],
                                                     [-3.2085e+00, -2.5904e-02],
                                                     [-4.9850e-01, -7.8708e+00],
                                                     [-7.2503e+00, -1.6186e+00]],

                                                    [[-9.8503e-01, -6.9615e-01],
                                                     [-4.4540e-02, -3.6726e+00],
                                                     [-3.0076e+00, -6.9163e-01],
                                                     [-7.5929e-03, -3.0764e+00]]])
        let t3Logsoftmax1 = t3.logsoftmax(1)
        let t3Logsoftmax1Correct = dsharp.tensor([[[-1.7120e+00, -6.6966e+00],
                                                     [-2.3962e+00, -7.5431e+00],
                                                     [-2.6764e+00, -6.5767e-03],
                                                     [-4.1609e-01, -5.3412e+00]],

                                                    [[-2.9332e-02, -3.0214e+00],
                                                     [-3.6943e+00, -5.0591e-02],
                                                     [-5.9562e+00, -8.3955e+00],
                                                     [-6.5344e+00, -7.8932e+00]],

                                                    [[-1.7061e+00, -2.0257e+00],
                                                     [-1.6935e+00, -2.7031e+00],
                                                     [-9.6285e+00, -2.2207e-01],
                                                     [-4.5492e-01, -8.3568e+00]]])
        let t3Logsoftmax2 = t3.logsoftmax(2)
        let t3Logsoftmax2Correct = dsharp.tensor([[[-3.1340e-01, -1.3129e+00],
                                                     [-2.7226e-01, -1.4341e+00],
                                                     [-6.6562e+00, -1.2869e-03],
                                                     [-3.2976e-01, -1.2698e+00]],

                                                    [[-1.7658e-01, -1.8210e+00],
                                                     [-4.9982e+00, -6.7731e-03],
                                                     [-2.8944e-01, -1.3810e+00],
                                                     [-6.8761e-01, -6.9871e-01]],

                                                    [[-2.8010e-01, -1.4094e+00],
                                                     [-1.5026e-01, -1.9696e+00],
                                                     [-8.5969e+00, -1.8464e-04],
                                                     [-1.6461e-04, -8.7118e+00]]])
        Assert.True(t1Logsoftmax0.allclose(t1Logsoftmax0Correct, 0.01))
        Assert.True(t2Logsoftmax0.allclose(t2Logsoftmax0Correct, 0.01))
        Assert.True(t2Logsoftmax1.allclose(t2Logsoftmax1Correct, 0.01))
        Assert.True(t3Logsoftmax0.allclose(t3Logsoftmax0Correct, 0.01))
        Assert.True(t3Logsoftmax1.allclose(t3Logsoftmax1Correct, 0.01))
        Assert.True(t3Logsoftmax2.allclose(t3Logsoftmax2Correct, 0.01))

    [<Test>]
    member this.TestTensorLogsumexp () =
        let t1 = dsharp.tensor([2.7291, 0.0607, 0.8290])
        let t1Logsumexp0 = t1.logsumexp(0)
        let t1Logsumexp0Correct = dsharp.tensor(2.9271)
        let t1Logsumexp0keepdim = t1.logsumexp(0, keepDim=true)
        let t1Logsumexp0keepdimCorrect = dsharp.tensor([2.9271])

        let t2 = dsharp.tensor([[1.3335, 1.6616, 2.4874, 6.1722],
                                [3.3478, 9.3019, 1.0844, 8.9874],
                                [8.6300, 1.8842, 9.1387, 9.1321]])
        let t2Logsumexp0 = t2.logsumexp(0)
        let t2Logsumexp0Correct = dsharp.tensor([8.6357, 9.3030, 9.1403, 9.7829])
        let t2Logsumexp0keepdim = t2.logsumexp(0, keepDim=true)
        let t2Logsumexp0keepdimCorrect = dsharp.tensor([[8.6357, 9.3030, 9.1403, 9.7829]])
        let t2Logsumexp1 = t2.logsumexp(1)
        let t2Logsumexp1Correct = dsharp.tensor([ 6.2153,  9.8518, 10.0924])
        let t2Logsumexp1keepdim = t2.logsumexp(1, keepDim=true)
        let t2Logsumexp1keepdimCorrect = dsharp.tensor([[ 6.2153],
                                                        [ 9.8518],
                                                        [10.0924]])

        let t3 = dsharp.tensor([[[3.0897, 2.0902],
                                 [2.4055, 1.2437],
                                 [2.1253, 8.7802],
                                 [4.3856, 3.4456]],

                                [[8.6233, 6.9789],
                                 [4.9583, 9.9497],
                                 [2.6964, 1.6048],
                                 [2.1182, 2.1071]],

                                [[8.1097, 6.9804],
                                 [8.1223, 6.3030],
                                 [0.1873, 8.7840],
                                 [9.3609, 0.6493]]])
             
        let t3Logsumexp0 = t3.logsumexp(0)
        let t3Logsumexp0Correct = dsharp.tensor([[9.0947, 7.6766],
                                                    [8.1668, 9.9756],
                                                    [3.1949, 9.4756],
                                                    [9.3685, 3.7257]])
        let t3Logsumexp0keepdim = t3.logsumexp(0, keepDim=true)
        let t3Logsumexp0keepdimCorrect = dsharp.tensor([[[9.0947, 7.6766],
                                                         [8.1668, 9.9756],
                                                         [3.1949, 9.4756],
                                                         [9.3685, 3.7257]]])                                                    
        let t3Logsumexp1 = t3.logsumexp(1)
        let t3Logsumexp1Correct = dsharp.tensor([[ 4.8017,  8.7868],
                                                    [ 8.6526, 10.0003],
                                                    [ 9.8158,  9.0061]])
        let t3Logsumexp1keepdim = t3.logsumexp(1, keepDim=true)
        let t3Logsumexp1keepdimCorrect = dsharp.tensor([[[ 4.8017,  8.7868]],

                                                        [[ 8.6526, 10.0003]],

                                                        [[ 9.8158,  9.0061]]])
        let t3Logsumexp2 = t3.logsumexp(2)
        let t3Logsumexp2Correct = dsharp.tensor([[3.4031, 2.6778, 8.7815, 4.7154],
                                                    [8.7999, 9.9565, 2.9858, 2.8058],
                                                    [8.3898, 8.2726, 8.7842, 9.3611]])
        let t3Logsumexp2keepdim = t3.logsumexp(2, keepDim=true)
        let t3Logsumexp2keepdimCorrect = dsharp.tensor([[[3.4031],
                                                         [2.6778],
                                                         [8.7815],
                                                         [4.7154]],

                                                        [[8.7999],
                                                         [9.9565],
                                                         [2.9858],
                                                         [2.8058]],

                                                        [[8.3898],
                                                         [8.2726],
                                                         [8.7842],
                                                         [9.3611]]])

        let t4 = dsharp.tensor([[167.385696, -146.549866, 168.850235, -41.856903, -56.691696, -78.774994, 42.035625, 97.490936, -42.763878, -2.130855], 
                                 [-62.961613, -497.529846, 371.218231, -30.224543, 368.146393, -325.945068, -292.102631, -24.760872, 130.348282, -193.775909]])
        let t4Logsumexp1 = t4.logsumexp(dim=1)
        let t4Logsumexp1Correct = dsharp.tensor([169.0582, 371.2635])
        Assert.True(t1Logsumexp0.allclose(t1Logsumexp0Correct, 0.001))
        Assert.True(t2Logsumexp0.allclose(t2Logsumexp0Correct, 0.001))
        Assert.True(t2Logsumexp1.allclose(t2Logsumexp1Correct, 0.001))
        Assert.True(t3Logsumexp0.allclose(t3Logsumexp0Correct, 0.001))
        Assert.True(t3Logsumexp1.allclose(t3Logsumexp1Correct, 0.001))
        Assert.True(t3Logsumexp2.allclose(t3Logsumexp2Correct, 0.001))
        Assert.True(t1Logsumexp0keepdim.allclose(t1Logsumexp0keepdimCorrect, 0.001))
        Assert.True(t2Logsumexp0keepdim.allclose(t2Logsumexp0keepdimCorrect, 0.001))
        Assert.True(t2Logsumexp1keepdim.allclose(t2Logsumexp1keepdimCorrect, 0.001))
        Assert.True(t3Logsumexp0keepdim.allclose(t3Logsumexp0keepdimCorrect, 0.001))
        Assert.True(t3Logsumexp1keepdim.allclose(t3Logsumexp1keepdimCorrect, 0.001))
        Assert.True(t3Logsumexp2keepdim.allclose(t3Logsumexp2keepdimCorrect, 0.001))
        Assert.True(t4Logsumexp1.allclose(t4Logsumexp1Correct, 0.75))

    [<Test>]
    member this.TestTensorNllLoss () =
        let t1a = dsharp.tensor([[0.15,0.85],[0.5,0.5],[0.8,0.2]]).log()
        let t1b = dsharp.tensor([0,1,1])
        let t1w = dsharp.tensor([-1.2,0.6])
        let l1 = dsharp.nllLoss(t1a, t1b)
        let l1Correct = dsharp.tensor(1.3999)
        let l2 = dsharp.nllLoss(t1a, t1b, weight=t1w)
        let l2Correct = dsharp.tensor(-0.8950)
        let l3 = dsharp.nllLoss(t1a, t1b, reduction="none")
        let l3Correct = dsharp.tensor([1.8971, 0.6931, 1.6094])
        let l4 = dsharp.nllLoss(t1a, t1b, reduction="none", weight=t1w)
        let l4Correct = dsharp.tensor([-2.2765,  0.4159,  0.9657])
        let l5 = dsharp.nllLoss(t1a, t1b, reduction="sum")
        let l5Correct = dsharp.tensor(4.1997)
        let l6 = dsharp.nllLoss(t1a, t1b, reduction="sum", weight=t1w)
        let l6Correct = dsharp.tensor(-0.8950)

        let t2a = dsharp.tensor([[[[-1.9318, -1.9386, -0.9488, -0.8787],
                                      [-1.1891, -2.4614, -1.0514, -1.1577],
                                      [-1.1977, -1.2468, -0.8123, -1.2226],
                                      [-0.9584, -2.1857, -0.9079, -1.5362]],

                                     [[-0.5465, -0.3825, -1.2375, -0.8330],
                                      [-2.4107, -0.8157, -0.9717, -1.0601],
                                      [-0.9040, -1.3655, -1.6613, -1.0334],
                                      [-0.8829, -1.4097, -1.5420, -1.9021]],

                                     [[-1.2868, -1.7491, -1.1311, -1.8975],
                                      [-0.5013, -0.7500, -1.3016, -1.0807],
                                      [-1.2271, -0.7824, -1.0044, -1.0505],
                                      [-1.5950, -0.4410, -0.9606, -0.4533]]],


                                    [[[-1.9389, -2.4012, -1.0333, -1.4381],
                                      [-1.5336, -1.6488, -2.1201, -1.5972],
                                      [-1.2268, -1.2666, -0.7287, -1.1079],
                                      [-1.3558, -1.0362, -1.2035, -1.0245]],

                                     [[-0.5721, -0.3562, -1.0314, -0.8208],
                                      [-0.4922, -0.5392, -0.9215, -0.5276],
                                      [-1.3011, -0.6734, -0.9661, -0.5593],
                                      [-0.6594, -0.9271, -1.0346, -0.7122]],

                                     [[-1.2316, -1.5651, -1.2460, -1.1315],
                                      [-1.7548, -1.4939, -0.7297, -1.5724],
                                      [-0.8335, -1.5690, -1.9886, -2.3212],
                                      [-1.4912, -1.3883, -1.0658, -1.8940]]]])
        let t2b = dsharp.tensor([[[2, 0, 1, 2],
                                     [2, 0, 1, 0],
                                     [2, 1, 0, 1],
                                     [1, 2, 1, 1]],

                                    [[2, 0, 2, 0],
                                     [0, 1, 0, 2],
                                     [2, 0, 2, 1],
                                     [1, 1, 1, 2]]])
        let t2w = dsharp.tensor([ 1.1983, -0.2633, -0.3064])
        let l7 = dsharp.nllLoss(t2a, t2b)
        let l7Correct = dsharp.tensor(1.3095)
        let l8 = dsharp.nllLoss(t2a, t2b, weight=t2w)
        let l8Correct = dsharp.tensor(2.4610)
        let l9 = dsharp.nllLoss(t2a, t2b, reduction="none")
        let l9Correct = dsharp.tensor([[[1.2868, 1.9386, 1.2375, 1.8975],
                                         [0.5013, 2.4614, 0.9717, 1.1577],
                                         [1.2271, 1.3655, 0.8123, 1.0334],
                                         [0.8829, 0.4410, 1.5420, 1.9021]],

                                        [[1.2316, 2.4012, 1.2460, 1.4381],
                                         [1.5336, 0.5392, 2.1201, 1.5724],
                                         [0.8335, 1.2666, 1.9886, 0.5593],
                                         [0.6594, 0.9271, 1.0346, 1.8940]]])
        let l10 = dsharp.nllLoss(t2a, t2b, reduction="none", weight=t2w)
        let l10Correct = dsharp.tensor([[[-0.3943,  2.3231, -0.3258, -0.5814],
                                         [-0.1536,  2.9496, -0.2558,  1.3872],
                                         [-0.3760, -0.3595,  0.9734, -0.2721],
                                         [-0.2324, -0.1351, -0.4059, -0.5007]],

                                        [[-0.3774,  2.8775, -0.3818,  1.7233],
                                         [ 1.8378, -0.1419,  2.5406, -0.4818],
                                         [-0.2554,  1.5179, -0.6093, -0.1472],
                                         [-0.1736, -0.2440, -0.2724, -0.5804]]])
        let l11 = dsharp.nllLoss(t2a, t2b, reduction="sum")
        let l11Correct = dsharp.tensor(41.9042)
        let l12 = dsharp.nllLoss(t2a, t2b, reduction="sum", weight=t2w)
        let l12Correct = dsharp.tensor(10.4726)

        Assert.True(l1Correct.allclose(l1, 0.001))
        Assert.True(l2Correct.allclose(l2, 0.001))
        Assert.True(l3Correct.allclose(l3, 0.001))
        Assert.True(l4Correct.allclose(l4, 0.001))
        Assert.True(l5Correct.allclose(l5, 0.001))
        Assert.True(l6Correct.allclose(l6, 0.001))
        Assert.True(l7Correct.allclose(l7, 0.001))
        Assert.True(l8Correct.allclose(l8, 0.001))
        Assert.True(l9Correct.allclose(l9, 0.001))
        Assert.True(l10Correct.allclose(l10, 0.001))
        Assert.True(l11Correct.allclose(l11, 0.001))
        Assert.True(l12Correct.allclose(l12, 0.001))

    [<Test>]
    member this.TestTensorCrossEntropyLoss () =
        let t1a = dsharp.tensor([[-0.6596,  0.3078, -0.2525, -0.2593, -0.2354],
                                    [ 0.4708,  0.6073,  1.5621, -1.4636,  0.9769],
                                    [ 0.5078,  0.0579,  1.0054,  0.3532,  1.1819],
                                    [ 1.5425, -0.2887,  1.0716, -1.3946,  0.8806]])
        let t1b = dsharp.tensor([3, 1, 0, 4])
        let t1w = dsharp.tensor([-1.4905,  0.5929,  1.0018, -1.0858, -0.5993])
        let l1 = dsharp.crossEntropyLoss(t1a, t1b)
        let l1Correct = dsharp.tensor(1.7059)
        let l2 = dsharp.crossEntropyLoss(t1a, t1b, weight=t1w)
        let l2Correct = dsharp.tensor(1.6969)
        let l3 = dsharp.crossEntropyLoss(t1a, t1b, reduction="none")
        let l3Correct = dsharp.tensor([1.6983, 1.7991, 1.8085, 1.5178])
        let l4 = dsharp.crossEntropyLoss(t1a, t1b, reduction="none", weight=t1w)
        let l4Correct = dsharp.tensor([-1.8439,  1.0666, -2.6956, -0.9096])
        let l5 = dsharp.crossEntropyLoss(t1a, t1b, reduction="sum")
        let l5Correct = dsharp.tensor(6.8237)
        let l6 = dsharp.crossEntropyLoss(t1a, t1b, reduction="sum", weight=t1w)
        let l6Correct = dsharp.tensor(-4.3825)

        Assert.True(l1Correct.allclose(l1, 0.001))
        Assert.True(l2Correct.allclose(l2, 0.001))
        Assert.True(l3Correct.allclose(l3, 0.001))
        Assert.True(l4Correct.allclose(l4, 0.001))
        Assert.True(l5Correct.allclose(l5, 0.001))
        Assert.True(l6Correct.allclose(l6, 0.001))

    [<Test>]
    member this.TestTensorMseLoss () =
        let t1a = dsharp.tensor([-0.2425,  0.2643,  0.7070,  1.2049,  1.6245])
        let t1b = dsharp.tensor([-1.0742,  1.5874,  0.6509,  0.8715,  0.0692])
        let l1 = dsharp.mseLoss(t1a, t1b)
        let l1Correct = dsharp.tensor(0.9951)
        let l2 = dsharp.mseLoss(t1a, t1b, reduction="none")
        let l2Correct = dsharp.tensor([0.6917, 1.7507, 0.0031, 0.1112, 2.4190])
        let l3 = dsharp.mseLoss(t1a, t1b, reduction="sum")
        let l3Correct = dsharp.tensor(4.9756)

        let t2a = dsharp.tensor([[ 0.6650,  0.5049, -0.7356,  0.5312, -0.6574],
                                 [ 1.0133,  0.9106,  0.1523,  0.2662,  1.1438],
                                 [ 0.3641, -1.8525, -0.0822, -1.0361,  0.2723]])
        let t2b = dsharp.tensor([[-1.0001, -1.4867, -0.3340, -0.2590,  0.1395],
                                 [-2.0158,  0.8281,  1.1726, -0.2359,  0.5007],
                                 [ 1.3242,  0.5215,  1.4293, -1.4235,  0.2473]])
        let l4 = dsharp.mseLoss(t2a, t2b)
        let l4Correct = dsharp.tensor(1.8694)
        let l5 = dsharp.mseLoss(t2a, t2b, reduction="none")
        let l5Correct = dsharp.tensor([[2.7726e+00, 3.9663e+00, 1.6130e-01, 6.2438e-01, 6.3511e-01],
                                        [9.1753e+00, 6.8075e-03, 1.0409e+00, 2.5207e-01, 4.1352e-01],
                                        [9.2194e-01, 5.6358e+00, 2.2848e+00, 1.5011e-01, 6.2556e-04]])
        let l6 = dsharp.mseLoss(t2a, t2b, reduction="sum")
        let l6Correct = dsharp.tensor(28.0416)

        Assert.True(l1Correct.allclose(l1, 0.01, 0.01))
        Assert.True(l2Correct.allclose(l2, 0.01, 0.01))
        Assert.True(l3Correct.allclose(l3, 0.01, 0.01))
        Assert.True(l4Correct.allclose(l4, 0.01, 0.01))
        Assert.True(l5Correct.allclose(l5, 0.01, 0.01))
        Assert.True(l6Correct.allclose(l6, 0.01, 0.01))

    [<Test>]
    member this.TestTensorDepth () =
        let t0 = dsharp.tensor([1.;2.])
        let t0Depth = t0.depth
        let t0DepthCorrect = 0
        let t1 = dsharp.tensor([1.;2.]).reverseDiff()
        let t1Depth = t1.depth
        let t1DepthCorrect = 1
        let t2 = dsharp.tensor([1.;2.]).reverseDiff().reverseDiff()
        let t2Depth = t2.depth
        let t2DepthCorrect = 2
        let t3 = dsharp.tensor([1.;2.]).reverseDiff().reverseDiff().forwardDiff(dsharp.tensor([1.; 1.]))
        let t3Depth = t3.depth
        let t3DepthCorrect = 3

        Assert.AreEqual(t0DepthCorrect, t0Depth)
        Assert.AreEqual(t1DepthCorrect, t1Depth)
        Assert.AreEqual(t2DepthCorrect, t2Depth)
        Assert.AreEqual(t3DepthCorrect, t3Depth)

    [<Test>]
    member this.FSharpCoreOps () =
        let t = dsharp.tensor([0.1; 0.2; 0.3])
        let add = t + t
        let addCorrect = t.add(t)
        let sub = t - t
        let subCorrect = t.sub(t)
        let mul = t * t
        let mulCorrect = t.mul(t)
        let div = t / t
        let divCorrect = t.div(t)
        let pow = t ** t
        let powCorrect = t.pow(t)
        let neg = -t
        let negCorrect = t.neg()
        // sign t not supported because FSharp.Core sign operator returns int
        let floor = floor t
        let floorCorrect = t.floor()
        let ceil = ceil t
        let ceilCorrect = t.ceil()
        let round = round t
        let roundCorrect = t.round()
        let abs = abs t
        let absCorrect = t.abs()
        let exp = exp t
        let expCorrect = t.exp()
        let log = log t
        let logCorrect = t.log()
        let log10 = log10 t
        let log10Correct = t.log10()
        let sqrt = sqrt t
        let sqrtCorrect = t.sqrt()
        let sin = sin t
        let sinCorrect = t.sin()
        let cos = cos t
        let cosCorrect = t.cos()
        let tan = tan t
        let tanCorrect = t.tan()
        let sinh = sinh t
        let sinhCorrect = t.sinh()
        let cosh = cosh t
        let coshCorrect = t.cosh()
        let tanh = tanh t
        let tanhCorrect = t.tanh()
        let asin = asin t
        let asinCorrect = t.asin()
        let acos = acos t
        let acosCorrect = t.acos()
        let atan = atan t
        let atanCorrect = t.atan()
        
        Assert.AreEqual(addCorrect, add)
        Assert.AreEqual(subCorrect, sub)
        Assert.AreEqual(mulCorrect, mul)
        Assert.AreEqual(divCorrect, div)
        Assert.AreEqual(powCorrect, pow)
        Assert.AreEqual(negCorrect, neg)
        Assert.AreEqual(floorCorrect, floor)
        Assert.AreEqual(ceilCorrect, ceil)
        Assert.AreEqual(roundCorrect, round)
        Assert.AreEqual(absCorrect, abs)
        Assert.AreEqual(expCorrect, exp)
        Assert.AreEqual(logCorrect, log)
        Assert.AreEqual(log10Correct, log10)
        Assert.AreEqual(sqrtCorrect, sqrt)
        Assert.AreEqual(sinCorrect, sin)
        Assert.AreEqual(cosCorrect, cos)
        Assert.AreEqual(tanCorrect, tan)
        Assert.AreEqual(sinhCorrect, sinh)
        Assert.AreEqual(coshCorrect, cosh)
        Assert.AreEqual(tanhCorrect, tanh)
        Assert.AreEqual(asinCorrect, asin)
        Assert.AreEqual(acosCorrect, acos)
        Assert.AreEqual(atanCorrect, atan)


