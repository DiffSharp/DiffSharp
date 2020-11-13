namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Util
open System

[<TestFixture>]
type TestTensor () =
    [<SetUp>]
    member _.Setup () =
        ()

    member _.TestTensorCreateAllTensorTypesGeneric (ofDouble: double -> 'T) =
      // Test creating these types of tensors
      for combo in Combos.All do 
        let t0 = combo.tensor(ofDouble 1.)
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.CheckEqual(t0ShapeCorrect, t0.shape)
        Assert.CheckEqual(t0DimCorrect, t0.dim)
        Assert.CheckEqual(combo.dtype, t0.dtype)

        let t1 = combo.tensor([ofDouble 1.; ofDouble 2.; ofDouble 3.])
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        Assert.CheckEqual(t1ShapeCorrect, t1.shape)
        Assert.CheckEqual(t1DimCorrect, t1.dim)
        Assert.CheckEqual(combo.dtype, t1.dtype)

        let t2 = combo.tensor([[ofDouble 1.; ofDouble 2.; ofDouble 3.]; [ofDouble 4.; ofDouble 5.; ofDouble 6.]])
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2
        Assert.CheckEqual(t2ShapeCorrect, t2.shape)
        Assert.CheckEqual(t2DimCorrect, t2.dim)
        Assert.CheckEqual(combo.dtype, t2.dtype)

        let t3 = combo.tensor([[[ofDouble 1.; ofDouble 2.; ofDouble 3.]; [ofDouble 4.; ofDouble 5.; ofDouble 6.]]])
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3

        Assert.CheckEqual(t3ShapeCorrect, t3.shape)
        Assert.CheckEqual(t3DimCorrect, t3.dim)
        Assert.CheckEqual(combo.dtype, t3.dtype)

        let t4 = combo.tensor([[[[ofDouble 1.; ofDouble 2.]]]])
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4

        Assert.CheckEqual(t4ShapeCorrect, t4.shape)
        Assert.CheckEqual(t4DimCorrect, t4.dim)
        Assert.CheckEqual(combo.dtype, t4.dtype)

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromFloat64Data() =
        this.TestTensorCreateAllTensorTypesGeneric id

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromFloat32Data() =
        this.TestTensorCreateAllTensorTypesGeneric float32

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt32Data() =
        this.TestTensorCreateAllTensorTypesGeneric int32

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt8Data() =
        this.TestTensorCreateAllTensorTypesGeneric int8

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt16Data() =
        this.TestTensorCreateAllTensorTypesGeneric int16

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt64Data() =
        this.TestTensorCreateAllTensorTypesGeneric int64

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromBoolData() =
        this.TestTensorCreateAllTensorTypesGeneric (fun i -> abs i >= 1.0)

        let t1 = dsharp.tensor([true, true])
        Assert.CheckEqual(Dtype.Bool, t1.dtype)

        let t2 = dsharp.tensor([true, false])
        Assert.CheckEqual(Dtype.Bool, t2.dtype)

        let t3 = dsharp.tensor([true; false])
        Assert.CheckEqual(Dtype.Bool, t3.dtype)

        let t4 = dsharp.tensor([true; false], dtype=Dtype.Float32)
        Assert.CheckEqual(Dtype.Float32, t4.dtype)

    [<Test>]
    member _.TestTensorHandle () =
        for combo in Combos.Float32 do
           if combo.backend = Backend.Reference then
               let t1 = combo.tensor([1.0f ; 1.0f ])
               Assert.CheckEqual([| 1.0f ; 1.0f |], (t1.primalRaw.Handle :?> float32[]))

    [<Test>]
    member _.TestTensorCreate0 () =
      for combo in Combos.AllDevicesAndBackends do
        let t0 = combo.tensor(1.)
        let t0Shape = t0.shape
        let t0Dim = t0.dim
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.CheckEqual(t0DimCorrect, t0Dim)
        Assert.CheckEqual(t0ShapeCorrect, t0Shape)

    [<Test>]
    member _.TestTensorCreate1 () =
      for combo in Combos.AllDevicesAndBackends do
        // create from double list
        let t1 = combo.tensor([1.; 2.; 3.])
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        Assert.CheckEqual(t1ShapeCorrect, t1.shape)
        Assert.CheckEqual(t1DimCorrect, t1.dim)

        // create from double[]
        let t1Array = combo.tensor([| 1.; 2.; 3. |])

        Assert.CheckEqual(t1ShapeCorrect, t1Array.shape)
        Assert.CheckEqual(t1DimCorrect, t1Array.dim)

        // create from seq<double>
        let t1Seq = combo.tensor(seq { 1.; 2.; 3. })

        Assert.CheckEqual(t1ShapeCorrect, t1Seq.shape)
        Assert.CheckEqual(t1DimCorrect, t1Seq.dim)

    [<Test>]
    member _.TestTensorCreate2 () =
      for combo in Combos.AllDevicesAndBackends do
        let t2Values = [[1.; 2.; 3.]; [4.; 5.; 6.]]
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2
        // let t2DtypeCorrect = Dtype.Float32
        let t2ValuesCorrect = array2D (List.map (List.map float32) t2Values)

        // create from double list list
        let t2 = combo.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]])
        Assert.CheckEqual(t2ShapeCorrect, t2.shape)
        Assert.CheckEqual(t2DimCorrect, t2.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2.toArray() :?> float32[,])

        // create from double array list
        let t2ArrayList = combo.tensor([[|1.; 2.; 3.|]; [|4.; 5.; 6.|]])
        Assert.CheckEqual(t2ShapeCorrect, t2ArrayList.shape)
        Assert.CheckEqual(t2DimCorrect, t2ArrayList.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2ArrayList.toArray() :?> float32[,])

        // create from double list array
        let t2ListArray = combo.tensor([| [1.; 2.; 3.]; [4.; 5.; 6.] |])
        Assert.CheckEqual(t2ShapeCorrect, t2ListArray.shape)
        Assert.CheckEqual(t2DimCorrect, t2ListArray.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2ListArray.toArray() :?> float32[,])

        // create from double[][]
        let t2ArrayArray = combo.tensor([| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.CheckEqual(t2ShapeCorrect, t2ArrayArray.shape)
        Assert.CheckEqual(t2DimCorrect, t2ArrayArray.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2ArrayArray.toArray() :?> float32[,])

        // create from double[,]
        let t2Array2D = combo.tensor(array2D [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.CheckEqual(t2ShapeCorrect, t2Array2D.shape)
        Assert.CheckEqual(t2DimCorrect, t2Array2D.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2Array2D.toArray() :?> float32[,])

        // create from seq<double[]>
        let t2ArraySeq = combo.tensor(seq { yield [| 1.; 2.; 3. |]; yield [| 4.; 5.; 6.|] })
        Assert.CheckEqual(t2ShapeCorrect, t2ArraySeq.shape)
        Assert.CheckEqual(t2DimCorrect, t2ArraySeq.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2ArraySeq.toArray() :?> float32[,])

        // create from seq<seq<double>>
        let t2SeqSeq = combo.tensor(seq { seq { 1.; 2.; 3. }; seq { 4.; 5.; 6.} })
        Assert.CheckEqual(t2ShapeCorrect, t2SeqSeq.shape)
        Assert.CheckEqual(t2DimCorrect, t2SeqSeq.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2SeqSeq.toArray() :?> float32[,])

        // create from (double * double * double) list list
        let t2TupleListList = combo.tensor([ [ 1., 2., 3. ]; [ 4., 5., 6. ] ])
        Assert.CheckEqual(t2ShapeCorrect, t2TupleListList.shape)
        Assert.CheckEqual(t2DimCorrect, t2TupleListList.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2TupleListList.toArray() :?> float32[,])

        // create from ((double * double * double) list * (double * double * double) list) list
        let t2TupleListTupleList = combo.tensor([ [ 1., 2., 3. ], [ 4., 5., 6. ] ])
        Assert.CheckEqual(t2ShapeCorrect, t2TupleListTupleList.shape)
        Assert.CheckEqual(t2DimCorrect, t2TupleListTupleList.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2TupleListTupleList.toArray() :?> float32[,])

        // create from (double * double * double)[]
        let t2TupleArray = combo.tensor([| [ 1., 2., 3. ]; [ 4., 5., 6. ] |])
        Assert.CheckEqual(t2ShapeCorrect, t2TupleArray.shape)
        Assert.CheckEqual(t2DimCorrect, t2TupleArray.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2TupleArray.toArray() :?> float32[,])

        // create from ((double * double * double) [] * (double * double * double) []) []
        let t2TupleArrayTupleArray = combo.tensor([| [| 1., 2., 3. |], [| 4., 5., 6. |] |])
        Assert.CheckEqual(t2ShapeCorrect, t2TupleArrayTupleArray.shape)
        Assert.CheckEqual(t2DimCorrect, t2TupleArrayTupleArray.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2TupleArrayTupleArray.toArray() :?> float32[,])
        Assert.CheckEqual(t2ValuesCorrect, t2TupleArrayTupleArray.toArray() :?> float32[,])

        // create from (double * double * double)seq
        let t2TupleArray = combo.tensor(seq { [ 1., 2., 3. ]; [ 4., 5., 6. ] })
        Assert.CheckEqual(t2ShapeCorrect, t2TupleArray.shape)
        Assert.CheckEqual(t2DimCorrect, t2TupleArray.dim)
        Assert.CheckEqual(t2ValuesCorrect, t2TupleArray.toArray() :?> float32[,])

        let t2TupleOfList = combo.tensor [[2.], [3.], [4.]]
        Assert.CheckEqual([| 3; 1 |], t2TupleOfList.shape)
        Assert.CheckEqual(array2D [ [2.f]; [3.f]; [4.f] ], t2TupleOfList.toArray() :?> float32[,])

    [<Test>]
    member _.TestTensorCreate3 () =
      for combo in Combos.AllDevicesAndBackends do
        let t3Values = [[[1.; 2.; 3.]; [4.; 5.; 6.]]]
        let t3 = combo.tensor(t3Values)
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3
        let t3ValuesCorrect = array3D (List.map (List.map (List.map float32)) t3Values)

        Assert.CheckEqual(t3ShapeCorrect, t3.shape)
        Assert.CheckEqual(t3DimCorrect, t3.dim)
        Assert.CheckEqual(t3ValuesCorrect, t3.toArray() :?> float32[,,])

    [<Test>]
    member _.TestTensorCreate4 () =
      for combo in Combos.AllDevicesAndBackends do
        let t4Values = [[[[1.; 2.]]]]
        let t4 = combo.tensor(t4Values)
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4
        let t4ValuesCorrect = array4D (List.map (List.map (List.map (List.map float32))) t4Values)

        Assert.CheckEqual(t4ShapeCorrect, t4.shape)
        Assert.CheckEqual(t4DimCorrect, t4.dim)
        Assert.CheckEqual(t4ValuesCorrect, t4.toArray() :?> float32[,,,])

    [<Test>]
    member this.TestTensorCreateFromTensor4 () =
        let t4Values = [[[[dsharp.tensor 1.; dsharp.tensor 2.]]]]
        let t4 = dsharp.tensor(t4Values)
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4
        let t4ValuesCorrect = array4D (List.map (List.map (List.map (List.map float32))) t4Values)

        Assert.AreEqual(t4ShapeCorrect, t4.shape)
        Assert.AreEqual(t4DimCorrect, t4.dim)
        Assert.AreEqual(t4ValuesCorrect, t4.toArray())

    [<Test>]
    member _.TestTensorToArray () =
        for combo in Combos.All do 
            let a = array2D [[1.; 2.]; [3.; 4.]]
            let t = combo.tensor(a)
            let tToArrayCorrect = combo.arrayCreator2D a
            Assert.CheckEqual(tToArrayCorrect, t.toArray())

    [<Test>]
    member _.TestTensorSaveSaveAndLoadToSpecificConfiguration () =
        let fileName = System.IO.Path.GetTempFileName()
        for combo in Combos.All do 
            let a = combo.tensor([[1,2],[3,4]])
            a.save(fileName)
            let b = combo.load(fileName)
            Assert.CheckEqual(a, b)

    [<Test>]
    member _.TestTensorSaveLoadBackToDefaultConfiguarionThenMoveToCombo () =
        let fileName = System.IO.Path.GetTempFileName()
        for combo in Combos.All do 
            let a = combo.tensor([[1,2],[3,4]])
            a.save(fileName)
            let b = Tensor.load(fileName)
            let bInCombo = combo.move(b)
            Assert.CheckEqual(a, bInCombo)

    [<Test>]
    member _.TestTensorSaveLoadBackToDefaultConfiguarion () =
        let fileName = System.IO.Path.GetTempFileName()
        for combo in Combos.All do 
            let a = combo.tensor([[1,2],[3,4]])
            a.save(fileName)
            let aInDefault = a.move(device=Device.Default, backend=Backend.Default)
            let b = Tensor.load(fileName, dtype = combo.dtype)
            Assert.CheckEqual(aInDefault, b)

    [<Test>]
    member _.TestTensorSaveLoadConfiguarion () =
        let fileName = System.IO.Path.GetTempFileName()
        let a = dsharp.tensor([[1,2],[3,4]])
        a.save(fileName)
        for combo in Combos.All do 
            let aInCombo = combo.move(a)
            let b = combo.load(fileName)
            Assert.CheckEqual(aInCombo, b)

    [<Test>]
    member _.TestTensorClone () =
        for combo in Combos.All do 
            let a = combo.randint(0,100,[10;10])
            let b = a.clone()
            Assert.CheckEqual(a, b)
            Assert.CheckEqual(a.dtype, b.dtype)

    [<Test>]
    member _.TestTensorFull () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1a = combo.full([2;3], 2)
            let t1b = combo.ones([2;3]) * 2
            let t2a = combo.full([], 2)
            let t2b = combo.ones([]) * 2
            Assert.CheckEqual(t1a, t1b)
            Assert.CheckEqual(t2a, t2b)

        for combo in Combos.All do 
            let t1 = combo.full([2], 1)
            let t1Expected = combo.tensor([1,1])
            Assert.CheckEqual(t1, t1Expected)

    [<Test>]
    member _.TestTensorZero () =
        for combo in Combos.All do 
            let t1 = combo.zero()
            let t1Expected = combo.tensor(0)
            Assert.CheckEqual(t1, t1Expected)
            Assert.CheckEqual(t1.shape, ([| |]: int32[]) )
            Assert.CheckEqual(t1.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorZerosDisposal () =
        for i in 0..1024 do
            let _ = dsharp.zeros([1024; 1024])
            printfn "%A" i
            //System.GC.Collect()


    [<Test>]
    member _.TestTensorZeros () =
        for combo in Combos.All do 
            let t0 = combo.zeros([])
            let t0Expected = combo.tensor(0)
            Assert.CheckEqual(t0.shape, ([| |]: int32[]) )
            Assert.CheckEqual(t0.dtype, combo.dtype)
            Assert.CheckEqual(t0, t0Expected)

            let t1 = combo.zeros([2])
            let t1Expected = combo.tensor([0,0])
            Assert.CheckEqual(t1.shape, ([| 2 |]: int32[]) )
            Assert.CheckEqual(t1.dtype, combo.dtype)
            Assert.CheckEqual(t1, t1Expected)

    [<Test>]
    member _.TestTensorEmpty () =
        for combo in Combos.All do 
            let t0 = combo.empty([])
            Assert.CheckEqual(t0.shape, ([| |]: int32[]) )
            Assert.CheckEqual(t0.dtype, combo.dtype)

            let t1 = combo.empty([2])
            Assert.CheckEqual(t1.shape, ([| 2 |]: int32[]) )
            Assert.CheckEqual(t1.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorOne () =
        for combo in Combos.All do 
            let t1 = combo.one()
            let t1Expected = combo.tensor(1)
            Assert.CheckEqual(t1, t1Expected)
            Assert.CheckEqual(t1.dtype, combo.dtype)
            Assert.CheckEqual(t1.shape, ([| |]: int32[]) )

    [<Test>]
    member _.TestTensorOnes () =
        for combo in Combos.All do 
            let t0 = combo.ones([])
            let t0Expected = combo.tensor(1)
            Assert.CheckEqual(t0.shape, ([| |]: int32[]) )
            Assert.CheckEqual(t0.dtype, combo.dtype)
            Assert.CheckEqual(t0, t0Expected)

            let t1 = combo.ones([2])
            let t1Expected = combo.tensor([1,1])
            Assert.CheckEqual(t1, t1Expected)
    [<Test>]
    member _.TestTensorIsTensor () =
        for combo in Combos.All do 
            let a = 2.
            let b = combo.tensor(2.)
            Assert.True(not (dsharp.isTensor(a)))
            Assert.True(dsharp.isTensor(b))

    [<Test>]
    member _.TestTensorConvert () =
        for combo in Combos.IntegralAndFloatingPoint do
            let v = 2.
            let t = combo.tensor(v)
            let tsingle = single t
            let tdouble = double t
            let tint16 = int16 t
            let tint32 = int32 t
            let tint64 = int64 t
            let tsingleCorrect = single v
            let tdoubleCorrect = double v
            let tint16Correct = int16 v
            let tint32Correct = int32 v
            let tint64Correct = int64 v
            Assert.CheckEqual(tsingleCorrect, tsingle)
            Assert.CheckEqual(tdoubleCorrect, tdouble)
            Assert.CheckEqual(tint16Correct, tint16)
            Assert.CheckEqual(tint32Correct, tint32)
            Assert.CheckEqual(tint64Correct, tint64)

        for combo in Combos.IntegralAndFloatingPoint do
            let v = 2.
            let t = combo.tensor(v)
            let tsingle = t.toSingle()
            let tdouble = t.toDouble()
            let tint16 = t.toInt16()
            let tint32 = t.toInt32()
            let tint64 = t.toInt64()
            let tsingleCorrect = single v
            let tdoubleCorrect = double v
            let tint16Correct = int16 v
            let tint32Correct = int32 v
            let tint64Correct = int64 v
            Assert.CheckEqual(tsingleCorrect, tsingle)
            Assert.CheckEqual(tdoubleCorrect, tdouble)
            Assert.CheckEqual(tint16Correct, tint16)
            Assert.CheckEqual(tint32Correct, tint32)
            Assert.CheckEqual(tint64Correct, tint64)

        for combo in Combos.Bool do
            let v = true
            let t = combo.tensor(v)
            let tbool = t.toBool()
            let tboolCorrect = v
            Assert.CheckEqual(tboolCorrect, tbool)

    [<Test>]
    member _.TestTensorOnehot () =
        for combo in Combos.All do 
            let t0 = combo.onehot(3, 0)
            let t1 = combo.onehot(3, 1)
            let t2 = combo.onehot(3, 2)
            let t0Correct = combo.tensor([1,0,0])
            let t1Correct = combo.tensor([0,1,0])
            let t2Correct = combo.tensor([0,0,1])
            Assert.CheckEqual(t0Correct, t0)
            Assert.CheckEqual(t1Correct, t1)
            Assert.CheckEqual(t2Correct, t2)

    [<Test>]
    // Test the underlying GetItem on the RawPrimal, useful when testing backends
    member _.TestTensorGetItemOnPrimal () =
      for combo in Combos.IntegralAndFloatingPoint do 
        let t0 = combo.tensor(2.)
        Assert.CheckEqual(2.0, t0.toDouble())

        let t1 = combo.tensor([2., 3., 4., 5., 6.])
        Assert.CheckEqual(2.0, t1.primalRaw.GetItem(0).toDouble())
        Assert.CheckEqual(3.0, t1.primalRaw.GetItem(1).toDouble())
        Assert.CheckEqual(4.0, t1.primalRaw.GetItem(2).toDouble())
        Assert.CheckEqual(5.0, t1.primalRaw.GetItem(3).toDouble())
        Assert.CheckEqual(6.0, t1.primalRaw.GetItem(4).toDouble())

        let t2 = combo.tensor([[2.]; [3.]])
        Assert.CheckEqual(2.0, t2.primalRaw.GetItem(0, 0).toDouble())
        Assert.CheckEqual(3.0, t2.primalRaw.GetItem(1, 0).toDouble())

        let t2b = combo.tensor([[1.;2.]; [3.;4.]])
        Assert.CheckEqual(1.0, t2b.primalRaw.GetItem(0, 0).toDouble())
        Assert.CheckEqual(2.0, t2b.primalRaw.GetItem(0, 1).toDouble())
        Assert.CheckEqual(3.0, t2b.primalRaw.GetItem(1, 0).toDouble())
        Assert.CheckEqual(4.0, t2b.primalRaw.GetItem(1, 1).toDouble())

        let t3 = combo.tensor([[[2.; 3.]]])
        Assert.CheckEqual(2.0, t3.primalRaw.GetItem(0, 0, 0).toDouble())
        Assert.CheckEqual(3.0, t3.primalRaw.GetItem(0, 0, 1).toDouble())

        let t4 = combo.tensor([[[[1.]]]])
        Assert.CheckEqual(1.0, t4.primalRaw.GetItem(0, 0, 0, 0).toDouble())

    [<Test>]
    // Test the underlying GetItem on the RawPrimal, useful when testing backends
    member _.TestTensorGetSliceOnPrimal () =
      for combo in Combos.IntegralAndFloatingPoint do 
        let t0 = combo.tensor(2.)
        Assert.CheckEqual(2.0, t0.toDouble())

        let t1 = combo.tensor([ 0 .. 10 ])
        let t1slice1 = t1.primalRaw.GetSlice(array2D [ [ 3; 4; 0 ] ])
        let t1slice2 = t1.primalRaw.GetSlice(array2D [ [ 3; 3; 0 ] ])

        Assert.CheckEqual(3, (t1slice1.GetItem(0) |> Convert.ToInt32))
        Assert.CheckEqual(4, (t1slice1.GetItem(1) |> Convert.ToInt32))
        Assert.CheckEqual(1, t1slice1.Dim)
        Assert.CheckEqual(2, t1slice1.Shape.[0])

        Assert.CheckEqual(3, (t1slice2.GetItem(0) |> Convert.ToInt32))
        Assert.CheckEqual(1, t1slice2.Dim)
        Assert.CheckEqual(1, t1slice2.Shape.[0])

        // TODO: slicing reducing down to scalar
        //let t1slice3 = t1.primalRaw.GetSlice(array2D [ [ 3; 3; 1 ] ])
        //Assert.CheckEqual(3, t1slice3.GetItem(0))
        //Assert.CheckEqual(0, t1slice3.Dim)

        let t2 = combo.tensor([ for i in 0 .. 10 -> [ i*10 .. i*10+10 ] ])
        let t2slice1 = t2.primalRaw.GetSlice(array2D [ [ 3; 5; 0 ]; [ 3; 5; 0 ] ])

        Assert.CheckEqual(33, t2slice1.GetItem(0, 0) |> Convert.ToInt32)
        Assert.CheckEqual(34, t2slice1.GetItem(0, 1) |> Convert.ToInt32)
        Assert.CheckEqual(35, t2slice1.GetItem(0, 2) |> Convert.ToInt32)
        Assert.CheckEqual(43, t2slice1.GetItem(1, 0) |> Convert.ToInt32)
        Assert.CheckEqual(44, t2slice1.GetItem(1, 1) |> Convert.ToInt32)
        Assert.CheckEqual(45, t2slice1.GetItem(1, 2) |> Convert.ToInt32)
        Assert.CheckEqual(53, t2slice1.GetItem(2, 0) |> Convert.ToInt32)
        Assert.CheckEqual(54, t2slice1.GetItem(2, 1) |> Convert.ToInt32)
        Assert.CheckEqual(55, t2slice1.GetItem(2, 2) |> Convert.ToInt32)

        let t2slice2 = t2.primalRaw.GetSlice(array2D [ [ 3; 5; 0 ]; [ 3; 3; 1 ] ])
        Assert.CheckEqual(33, t2slice2.GetItem(0) |> Convert.ToInt32)
        Assert.CheckEqual(43, t2slice2.GetItem(1) |> Convert.ToInt32)
        Assert.CheckEqual(53, t2slice2.GetItem(2) |> Convert.ToInt32)

        let t2slice3 = t2.primalRaw.GetSlice(array2D [ [ 3; 3; 1 ]; [ 3; 5; 0 ] ])
        Assert.CheckEqual(33, t2slice3.GetItem(0) |> Convert.ToInt32)
        Assert.CheckEqual(34, t2slice3.GetItem(1) |> Convert.ToInt32)
        Assert.CheckEqual(35, t2slice3.GetItem(2) |> Convert.ToInt32)


    [<Test>]
    // Test cases of indexing where indexing returns a scalar
    member _.TestTensorIndexItemAsScalarTensor () =
      for combo in Combos.IntegralAndFloatingPoint do 
        let t0 = combo.tensor(2.)
        Assert.CheckEqual(2.0, t0.toDouble())

        let t1 = combo.tensor([2., 3., 4., 5., 6.])
        let t1_0 = t1.[0]
        let t1_1 = t1.[1]
        let t1_0_s = t1_0.toDouble()
        let t1_1_s = t1_1.toDouble()
        Assert.CheckEqual(2.0, t1_0_s)
        Assert.CheckEqual(3.0, t1_1_s)
        Assert.CheckEqual(4.0, (t1.[2].toDouble()))
        Assert.CheckEqual(5.0, (t1.[3].toDouble()))

        let t2 = combo.tensor([[2.]; [3.]])
        Assert.CheckEqual(2.0, (t2.[0,0].toDouble()))
        Assert.CheckEqual(3.0, (t2.[1,0].toDouble()))

        let t2b = combo.tensor([[1.;2.]; [3.;4.]])
        Assert.CheckEqual(1.0, (t2b.[0,0].toDouble()))
        Assert.CheckEqual(2.0, (t2b.[0,1].toDouble()))
        Assert.CheckEqual(3.0, (t2b.[1,0].toDouble()))
        Assert.CheckEqual(4.0, (t2b.[1,1].toDouble()))

        let t3 = combo.tensor([[[2.; 3.]]])
        Assert.CheckEqual(2.0, (t3.[0,0,0].toDouble()))
        Assert.CheckEqual(3.0, (t3.[0,0,1].toDouble()))

        let t4 = combo.tensor([[[[1.]]]])
        Assert.CheckEqual(1.0, (t4.[0,0,0,0].toDouble()))

    [<Test>]
    member _.TestTensorArange () =
        for combo in Combos.All do
            let t = combo.arange(5.)
            let tCorrect = combo.tensor([0.,1.,2.,3.,4.])
            Assert.CheckEqual(tCorrect, t)

            let t2 = combo.arange(5., 1.5, 0.5)
            let t2Correct = combo.tensor([1.5,2.,2.5,3.,3.5,4.,4.5])
            Assert.CheckEqual(t2Correct, t2)

            let t3 = combo.arange(5)
            let t3Correct = combo.tensor([0,1,2,3,4], dtype=Dtype.Int32)
            Assert.CheckEqual(t3Correct, t3)

    [<Test>]
    member _.TestTensorZeroSize () =
        for combo in Combos.All do
            let t = combo.tensor([])
            let tshape = t.shape
            let tshapeCorrect = [|0|]
            let tdtype = t.dtype
            let tdtypeCorrect = combo.dtype
            Assert.CheckEqual(tshapeCorrect, tshape)
            Assert.CheckEqual(tdtypeCorrect, tdtype)

            let t = combo.tensor([||])
            let tshape = t.shape
            let tshapeCorrect = [|0|]
            let tdtype = t.dtype
            let tdtypeCorrect = combo.dtype
            Assert.CheckEqual(tshapeCorrect, tshape)
            Assert.CheckEqual(tdtypeCorrect, tdtype)

        for combo in Combos.IntegralAndFloatingPoint do
            let t = combo.tensor([])

            let tAdd = t + 2
            let tAddCorrect = t
            Assert.CheckEqual(tAddCorrect, tAdd)

            let tMul = t * 2
            let tMulCorrect = t
            Assert.CheckEqual(tMulCorrect, tMul)

            let tSum = t.sum()
            let tSumCorrect = tSum.zeroLike()
            Assert.CheckEqual(tSumCorrect, tSum)

            let tClone = t.clone()
            let tCloneCorrect = t
            Assert.CheckEqual(tCloneCorrect, tClone)

        for combo in Combos.IntegralAndFloatingPoint do
            let t = combo.tensor([])

            let tSub = t - 2
            let tSubCorrect = t
            Assert.CheckEqual(tSubCorrect, tSub)

            let tDiv = t / 2
            let tDivCorrect = t
            Assert.CheckEqual(tDivCorrect, tDiv)

            let tNeg = -t
            let tNegCorrect = t
            Assert.CheckEqual(tNegCorrect, tNeg)

            let tAbs = dsharp.abs(t)
            let tAbsCorrect = t
            Assert.CheckEqual(tAbsCorrect, tAbs)

            let tSign = dsharp.sign(t)
            let tSignCorrect = t
            Assert.CheckEqual(tSignCorrect, tSign)

        for combo in Combos.FloatingPoint do
            let t = combo.tensor([])

            let tPow = t ** 2
            let tPowCorrect = t
            Assert.CheckEqual(tPowCorrect, tPow)

    [<Test>]
    member _.TestTensorEye () =
        for combo in Combos.All do
            let t = combo.eye(3)
            let tCorrect = combo.tensor([[1., 0., 0.],
                                          [0., 1., 0.],
                                          [0., 0., 1.]])
            Assert.True(tCorrect.allclose(t))

            let t = combo.eye(3, 2)
            let tCorrect = combo.tensor([[1., 0.],
                                          [0., 1.],
                                          [0., 0.]])
            Assert.True(tCorrect.allclose(t))

            let t = combo.eye(2, 3)
            let tCorrect = combo.tensor([[1., 0., 0.],
                                          [0., 1., 0.]])
            Assert.True(tCorrect.allclose(t))

            let t = combo.eye(2, 0)
            let tCorrect = combo.tensor([])
            Assert.True(tCorrect.allclose(t))
        
    [<Test>]
    member _.TestTensorMultinomial () =
        for combo in Combos.FloatingPoint do
            let p1 = combo.tensor([0.2,0.3,0.5])
            let m1 = dsharp.multinomial(p1, numSamples=3000)
            let m1dtype = m1.dtype
            let m1dtypeCorrect = Dtype.Int32
            let m1mean = m1.float().mean()
            let m1stddev = m1.float().stddev()
            let m1meanCorrect = combo.tensor(1.3001).float()
            let m1stddevCorrect = combo.tensor(0.7810).float()
            Assert.CheckEqual(m1dtypeCorrect, m1dtype)
            Assert.True(m1meanCorrect.allclose(m1mean, 0.1))
            Assert.True(m1stddevCorrect.allclose(m1stddev, 0.1))

            let p2 = combo.tensor([[0.2,0.3,0.5],[0.8,0.1,0.1]])
            let m2 = dsharp.multinomial(p2, numSamples=3000)
            let m2dtype = m2.dtype
            let m2dtypeCorrect = Dtype.Int32
            let m2mean = m2.float().mean(dim=1)
            let m2stddev = m2.float().stddev(dim=1)
            let m2meanCorrect = combo.tensor([1.3001, 0.3001]).float()
            let m2stddevCorrect = combo.tensor([0.7810, 0.6404]).float()
            Assert.CheckEqual(m2dtypeCorrect, m2dtype)
            Assert.True(m2meanCorrect.allclose(m2mean, 0.15))
            Assert.True(m2stddevCorrect.allclose(m2stddev, 0.15))

    [<Test>]
    member _.TestTensorBernoulli () =
        for combo in Combos.FloatingPoint do
            let p1 = combo.tensor([0.1,0.5,0.9])
            let b1 = dsharp.bernoulli(p1.expand([2500;3]))
            let b1mean = b1.mean(dim=0)
            let b1meanCorrect = p1
            Assert.True(b1meanCorrect.allclose(b1mean, 0.1, 0.1))

            let p2 = combo.tensor([[0.2,0.4],[0.9, 0.5]])
            let b2 = dsharp.bernoulli(p2.expand([2500;2;2]))
            let b2mean = b2.mean(dim=0)
            let b2meanCorrect = p2
            Assert.True(b2meanCorrect.allclose(b2mean, 0.1, 0.1))

    [<Test>]
    member _.TestTensorDropout () =
        for combo in Combos.FloatingPoint do
            for p in [0.; 0.2; 0.8; 1.] do
                let t = combo.ones([100;100])
                let d = dsharp.dropout(t, p)
                let m = d.mean() |> float
                let mCorrect = 1. - p
                Assert.True(abs(mCorrect - m) < 0.1)

    [<Test>]
    member _.TestTensorDropout2d () =
        for combo in Combos.FloatingPoint do
            for p in [0.; 0.2; 0.8; 1.] do
                let t = combo.ones([100;100;8;8])
                let d = dsharp.dropout2d(t, p)
                let m = d.mean() |> float
                let mCorrect = 1. - p
                Assert.True(abs(mCorrect - m) < 0.1)

    [<Test>]
    member _.TestTensorDropout3d () =
        for combo in Combos.FloatingPoint do
            for p in [0.; 0.2; 0.8; 1.] do
                let t = combo.ones([100;100;8;8;8])
                let d = dsharp.dropout3d(t, p)
                let m = d.mean() |> float
                let mCorrect = 1. - p
                Assert.True(abs(mCorrect - m) < 0.1)

    [<Test>]
    member _.TestTensorToString () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t0 = combo.tensor(2.)
            let t1 = combo.tensor([[2.]; [2.]])
            let t2 = combo.tensor([[[2.; 2.]]])
            let t3 = combo.tensor([[1.;2.]; [3.;4.]])
            let t4 = combo.tensor([[[[1.]]]])
            let t0String = t0.ToString()
            let t1String = t1.ToString()
            let t2String = t2.ToString()
            let t3String = t3.ToString()
            let t4String = t4.ToString()
            let suffix = 
                match combo.dtype with 
                | Bool -> failwith "unexpected bool dtype in test"
                | Byte -> ""
                | Int8 -> ""
                | Int16 -> ""
                | Int32 -> ""
                | Int64 -> ""
                | Float32 -> ".000000"
                | Float64 -> ".000000"
            let dtypeText = 
                if combo.dtype = Dtype.Default then
                    ""
                else
                    sprintf ",dtype=%s" (combo.dtype.ToString())
            let deviceText = 
                if combo.device = Device.Default then
                    ""
                else
                    sprintf ",device=%s" (combo.device.ToString())
            let backendText = 
                if combo.backend = Backend.Default then
                    ""
                else
                    sprintf ",backend=%s" (combo.backend.ToString())

            let extraText = dtypeText + deviceText + backendText
            let t0StringCorrect = sprintf "tensor(2%s%s)" suffix extraText
            let t1StringCorrect = sprintf "tensor([[2%s],\n        [2%s]]%s)" suffix suffix extraText
            let t2StringCorrect = sprintf "tensor([[[2%s, 2%s]]]%s)" suffix suffix extraText
            let t3StringCorrect = sprintf "tensor([[1%s, 2%s],\n        [3%s, 4%s]]%s)" suffix suffix suffix suffix extraText
            let t4StringCorrect = sprintf "tensor([[[[1%s]]]]%s)" suffix extraText
            Assert.CheckEqual(t0StringCorrect, t0String)
            Assert.CheckEqual(t1StringCorrect, t1String)
            Assert.CheckEqual(t2StringCorrect, t2String)
            Assert.CheckEqual(t3StringCorrect, t3String)
            Assert.CheckEqual(t4StringCorrect, t4String)

        let t0Bool = dsharp.tensor([ 0.5; 1.0 ], dtype=Dtype.Bool)
        let t0BoolToString = t0Bool.ToString()
        let t0BoolToStringCorrect = sprintf "tensor([false, true],dtype=Bool)" 
        Assert.CheckEqual(t0BoolToString, t0BoolToStringCorrect)

        let t1Bool = dsharp.tensor([ false; true ], dtype=Dtype.Bool)
        let t1BoolToString = t1Bool.ToString()
        let t1BoolToStringCorrect = sprintf "tensor([false, true],dtype=Bool)" 
        Assert.CheckEqual(t1BoolToString, t1BoolToStringCorrect)

    [<Test>]
    member _.TestTensorEqual () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1A = combo.tensor(-1.)
            let t1B = combo.tensor(1.)
            let t1C = combo.tensor(1.)
            let t1At1BEqual = t1A = t1B
            let t1At1BEqualCorrect = false
            let t1Bt1CEqual = t1B = t1C
            let t1Bt1CEqualCorrect = true

            Assert.CheckEqual(t1At1BEqualCorrect, t1At1BEqual)
            Assert.CheckEqual(t1Bt1CEqualCorrect, t1Bt1CEqual)

            // Systematic testing. The tensors below are listed in expected order of comparison
            let t2S =
                [ combo.tensor( 0. )
                  combo.tensor( 1. )
                  combo.tensor([ 1.] )
                  combo.tensor([ 2.] )
                  combo.tensor([ 1.; 1.] )
                  combo.tensor([ 1.; 2. ] )
                  combo.tensor([ 2.; 1. ] ) 
                  combo.tensor([ [ 1.; 1.] ]) ]

            // Check the F# generic '=' gives expected results
            let equalsResults = [| for a in t2S -> [| for b in t2S -> a = b |] |]
            let equalsCorrect = [| for i in 0..t2S.Length-1 -> [| for j in 0..t2S.Length-1 -> (i=j) |] |]

            Assert.CheckEqual(equalsResults, equalsCorrect)

    // Bool
        for combo in Combos.Bool do 
            let t1A = combo.tensor(false)
            let t1B = combo.tensor(true)
            let t1C = combo.tensor(true)
            let t1At1BEqual = t1A = t1B
            let t1At1BEqualCorrect = false
            let t1Bt1CEqual = t1B = t1C
            let t1Bt1CEqualCorrect = true

            Assert.CheckEqual(t1At1BEqualCorrect, t1At1BEqual)
            Assert.CheckEqual(t1Bt1CEqualCorrect, t1Bt1CEqual)

        for combo in Combos.All do 
            for dtype2 in Dtypes.All do 
                 if combo.dtype <> dtype2 then 
                     isInvalidOp (fun () -> combo.tensor(1) = combo.tensor(1, dtype=dtype2))

    [<Test>]
    member _.TestTensorHash () =
        for combo in Combos.IntegralAndFloatingPoint do 

            // Systematic testing. The tensors below are listed in expected order of comparison
            let t2S =
                [ combo.tensor( 0. )
                  combo.tensor( 1. )
                  combo.tensor([ 1.] )
                  combo.tensor([ 2.] )
                  combo.tensor([ 1.; 1.] )
                  combo.tensor([ 1.; 2. ] )
                  combo.tensor([ 2.; 1. ] ) 
                  combo.tensor([ [ 1.; 1.] ]) ]

            // Check the F# generic hashes are the same for identical tensors, and different for this small sample of tensors
            let hashSameResults = [| for a in t2S -> [| for b in t2S -> hash a = hash b |] |]
            let hashSameCorrect = [| for i in 0..t2S.Length-1 -> [| for j in 0..t2S.Length-1 -> (i=j) |] |]

            Assert.CheckEqual(hashSameResults, hashSameCorrect)

            // Check reallocating an identical tensor doesn't change the hash
            let t2a = combo.tensor([ 1.] )
            let t2b = combo.tensor([ 1.] )
            Assert.CheckEqual(t2a.GetHashCode(), t2b.GetHashCode())

            // Check adding `ForwardDiff` doesn't change the hash or equality
            Assert.CheckEqual(t2a.forwardDiff(combo.tensor([1.])).GetHashCode(), t2a.GetHashCode())
            Assert.CheckEqual(true, (t2a.forwardDiff(combo.tensor([1.]))) = t2a)

            // Check adding `ReverseDiff` doesn't change the hash or equality
            Assert.CheckEqual(t2a.reverseDiff().GetHashCode(), t2a.GetHashCode())
            Assert.CheckEqual(true, (t2a.reverseDiff()) = t2a)

    [<Test>]
    member _.TestTensorCompare () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1A = combo.tensor(2.)
            let t1B = combo.tensor(3.)
            let t1At1BLess = t1A < t1B
            let t1At1BLessCorrect = true

            Assert.CheckEqual(t1At1BLessCorrect, t1At1BLess)

    // Bool
        for combo in Combos.Bool do 
            let t1A = combo.tensor(false)
            let t1B = combo.tensor(true)
            let t1At1BLess = t1A < t1B
            let t1At1BLessCorrect = true

            Assert.CheckEqual(t1At1BLessCorrect, t1At1BLess)

    [<Test>]
    member _.TestTensorMove () =
        for combo1 in Combos.All do
            for combo2 in Combos.All do
                // printfn "%A %A" (combo1.dtype, combo1.device, combo1.backend) (combo2.dtype, combo2.device, combo2.backend)
                let t1 = combo1.tensor([0, 1, 2, 3])
                let t2 = t1.move(combo2.dtype, combo2.device, combo2.backend)
                let t2b = t2.move(combo1.dtype, combo1.device, combo1.backend)
                Assert.CheckEqual(combo2.dtype, t2.dtype)
                Assert.CheckEqual(combo2.device, t2.device)
                Assert.CheckEqual(combo2.backend, t2.backend)
                if combo2.dtype <> Dtype.Bool then // Conversion to bool is irreversible for tensor([0, 1, 2, 3])
                    Assert.CheckEqual(t1, t2b)

    [<Test>]
    member _.TestTensorMoveDefaultBackend () =
        // Check that device and backend are not changed if not specified in move
        for combo1 in Combos.All do
            let t1 = combo1.tensor([0, 1, 2, 3])
            let t1b = t1.move(combo1.dtype, ?backend=None, ?device=None)
            Assert.CheckEqual(combo1.backend, t1b.backend)
            Assert.CheckEqual(combo1.device, t1b.device)

    [<Test>]
    member _.TestTensorCast () =
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in Dtypes.IntegralAndFloatingPoint do 
                let t1 = combo.tensor([1.; 2.; 3.; 5.])
                let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=dtype2)
                let t1Cast = t1.cast(dtype2)
                let t2Cast = t2.cast(combo.dtype)

                Assert.CheckEqual(t1Cast.dtype, dtype2)
                Assert.CheckEqual(t2Cast.dtype, combo.dtype)
                Assert.CheckEqual(t1Cast, t2)
                Assert.CheckEqual(t1, t2Cast)

        for combo in Combos.IntegralAndFloatingPoint do 
            let t1Bool = combo.tensor([true; false], dtype=Dtype.Bool)
            let t2Bool = combo.tensor([1.; 0.])
            let t1BoolCast = t1Bool.cast(combo.dtype)
            let t2BoolCast = t2Bool.cast(Dtype.Bool)

            Assert.CheckEqual(t1BoolCast.dtype, combo.dtype)
            Assert.CheckEqual(t2BoolCast.dtype, Dtype.Bool)
            Assert.CheckEqual(t1BoolCast, t2Bool)
            Assert.CheckEqual(t1Bool, t2BoolCast)

        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Int8)
            let t1Cast = t1.int8()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Int8)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Int16)
            let t1Cast = t1.int16()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Int16)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Int32)
            let t1Cast = t1.int32()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Int32)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Int32)
            let t1Cast = t1.int()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Int32)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Int64)
            let t1Cast = t1.int64()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Int64)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Float32)
            let t1Cast = t1.float32()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Float32)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Float64)
            let t1Cast = t1.float64()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Float64)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Float64)
            let t1Cast = t1.float()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Float64)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=Dtype.Float64)
            let t1Cast = t1.double()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Float64)
            Assert.CheckEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 0.])
            let t2 = combo.tensor([1.; 0.], dtype=Dtype.Bool)
            let t1Cast = t1.bool()

            Assert.CheckEqual(t1Cast.dtype, Dtype.Bool)
            Assert.CheckEqual(t1Cast, t2)

    [<Test>]
    member _.TestTensorBool () =
        for tys in Combos.Bool do
            let t1 = tys.tensor([1; 0; 1; 0], dtype=Bool)

            Assert.CheckEqual([| true; false; true; false |], t1.toArray() :?> bool[])
            Assert.CheckEqual(Bool, t1.dtype)

            let t2 = tys.tensor([true; false; true; false], dtype=Bool)

            Assert.CheckEqual([| true; false; true; false |], t2.toArray() :?> bool[])
            Assert.CheckEqual(Bool, t2.dtype)

    [<Test>]
    member _.TestTensorLtTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Lt = t1.lt(t2)
            let t1t2LtCorrect = combo.tensor([0.; 1.; 1.; 0.], dtype=Dtype.Bool)

            Assert.CheckEqual(t1t2LtCorrect, t1t2Lt)
            Assert.CheckEqual(Dtype.Bool, t1t2Lt.dtype)

        for combo in Combos.Bool do 
            // Test bool type separately
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolLt = t1Bool.lt(t2Bool)
            let t1Boolt2BoolLtCorrect = combo.tensor([false; false; true; false ], dtype=Dtype.Bool)

            Assert.CheckEqual(t1Boolt2BoolLtCorrect, t1Boolt2BoolLt)

    [<Test>]
    member _.TestTensorLeTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Le = t1.le(t2)
            let t1t2LeCorrect = combo.tensor([1.; 1.; 1.; 0.], dtype=Dtype.Bool)

            Assert.CheckEqual(t1t2LeCorrect, t1t2Le)
            Assert.CheckEqual(Dtype.Bool, t1t2Le.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolLe = t1Bool.le(t2Bool)
            let t1Boolt2BoolLeCorrect = combo.tensor([true; false; true; true ], dtype=Dtype.Bool)

            Assert.CheckEqual(t1Boolt2BoolLeCorrect, t1Boolt2BoolLe)

    [<Test>]
    member _.TestTensorGtTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Gt = t1.gt(t2)
            let t1t2GtCorrect = combo.tensor([0.; 0.; 0.; 1.], dtype=Dtype.Bool)

            Assert.CheckEqual(t1t2GtCorrect, t1t2Gt)
            Assert.CheckEqual(Dtype.Bool, t1t2Gt.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolGt = t1Bool.gt(t2Bool)
            let t1Boolt2BoolGtCorrect = combo.tensor([false; true; false; false ], dtype=Dtype.Bool)

            Assert.CheckEqual(t1Boolt2BoolGtCorrect, t1Boolt2BoolGt)

    [<Test>]
    member _.TestTensorGeTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Ge = t1.ge(t2)
            let t1t2GeCorrect = combo.tensor([1.; 0.; 0.; 1.], dtype=Dtype.Bool)

            Assert.CheckEqual(t1t2GeCorrect, t1t2Ge)
            Assert.CheckEqual(Dtype.Bool, t1t2Ge.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            // Test bool type separately
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolGe = t1Bool.ge(t2Bool)
            let t1Boolt2BoolGeCorrect = combo.tensor([true; true; false; true ], dtype=Dtype.Bool)

            Assert.CheckEqual(t1Boolt2BoolGeCorrect, t1Boolt2BoolGe)

    [<Test>]
    member _.TestTensorIsinf () =
        // isinf always returns bool tensor
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; infinity; 3.; -infinity])
            let i = dsharp.isinf(t)
            let iCorrect = combo.tensor([0.; 1.; 0.; 1.], dtype=Dtype.Bool)
            Assert.CheckEqual(iCorrect, i)

        // Integer tensors always return 0 for isinf
        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.isinf(t)
            let iCorrect = combo.tensor([0.; 0.; 0.], dtype=Dtype.Bool)
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorIsnan () =
        // isnan always returns bool tensor
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; nan; 3.; nan])
            let i = dsharp.isnan(t)
            let iCorrect = combo.tensor([false; true; false; true], dtype=Dtype.Bool)
            Assert.CheckEqual(iCorrect, i)

        // Integer and bool tensors always return false for isnan
        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.isnan(t)
            let iCorrect = combo.tensor([0.; 0.; 0.], dtype=Dtype.Bool)
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorOnesLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.onesLike([2])
            let iCorrect = combo.tensor([1.; 1.])
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorZerosLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.zerosLike([2])
            let iCorrect = combo.tensor([0.; 0.])
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorFullLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.fullLike(4.0, [2])
            let iCorrect = combo.tensor([4.; 4.])
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorZeroLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.zeroLike()
            let iCorrect = combo.tensor(0.)
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorOneLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.oneLike()
            let iCorrect = combo.tensor(1.)
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorRandLike() =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.randLike([2])
            Assert.CheckEqual(i.shape, [|2|])
            Assert.CheckEqual(i.dtype, t.dtype)
            Assert.CheckEqual(i.dtype, combo.dtype)

        for combo in Combos.Bool do
            let t = combo.tensor([1.; 2.; 3.; 4.])
            isInvalidOp(fun () -> t.randLike([2]))

    [<Test>]
    member _.TestTensorRandnLike() =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.randnLike([2])
            Assert.CheckEqual(i.shape, [|2|])
            Assert.CheckEqual(i.dtype, t.dtype)
            Assert.CheckEqual(i.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            let t = combo.tensor([1.; 2.; 3.; 4.])
            isInvalidOp(fun () -> t.randnLike([2]))

    [<Test>]
    member _.TestTensorHasinf () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; infinity; 3.; -infinity])
            let t1i = dsharp.hasinf(t1)
            let t1iCorrect = true
            let t2 = combo.tensor([1.; 2.; 3.; 4.])
            let t2i = dsharp.hasinf(t2)
            let t2iCorrect = false
            Assert.CheckEqual(t1iCorrect, t1i)
            Assert.CheckEqual(t2iCorrect, t2i)

        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.hasinf(t)
            let iCorrect = false
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorHasnan () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; nan; 3.; nan])
            let t1i = dsharp.hasnan(t1)
            let t1iCorrect = true
            let t2 = combo.tensor([1.; 2.; 3.; 4.])
            let t2i = dsharp.hasnan(t2)
            let t2iCorrect = false
            Assert.CheckEqual(t1iCorrect, t1i)
            Assert.CheckEqual(t2iCorrect, t2i)

        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.hasnan(t)
            let iCorrect = false
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorAddTT () =
        // Test all pairs of non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in Dtypes.IntegralAndFloatingPoint do 
                match Dtype.widen combo.dtype dtype2 with 
                | None -> ()
                | Some dtypeRes -> 
                let t1 = combo.tensor([1.; 2.]) + combo.tensor([3.; 4.], dtype=dtype2)
                let t1Correct = combo.tensor([4.; 6.], dtype=dtypeRes)

                let t2 = combo.tensor([1.; 2.]) + combo.tensor(5., dtype=dtype2)
                let t2Correct = combo.tensor([6.; 7.], dtype=dtypeRes)

                Assert.CheckEqual(t1Correct, t1)
                Assert.CheckEqual(t2Correct, t2)
                Assert.CheckEqual(t1.dtype, dtypeRes)
                Assert.CheckEqual(t2.dtype, dtypeRes)

    [<Test>]
    member _.TestTensorAddTTScalarBroadcasting () =
        // Test scalar broadcasting 
        for combo in Combos.IntegralAndFloatingPoint do 
            let t3 = combo.tensor([1; 2]) + 5
            let t3Correct = combo.tensor([6; 7])

            let t4 = combo.tensor([1; 2]) + 5
            let t4Correct = combo.tensor([6; 7])

            let t5 = combo.tensor([1; 2]) + 5
            let t5Correct = combo.tensor([6; 7])

            Assert.CheckEqual(t3Correct, t3)
            Assert.CheckEqual(t4Correct, t4)
            Assert.CheckEqual(t5Correct, t5)
            Assert.CheckEqual(t3.dtype, combo.dtype)
            Assert.CheckEqual(t4.dtype, combo.dtype)
            Assert.CheckEqual(t5.dtype, combo.dtype)

        // Bool tensors support addition returning bool
        //
        //   t = torch.tensor([[True]], dtype=torch.bool)
        //   t + t
        //
        //   tensor([[True]])

        for combo in Combos.Bool do 
            let t5a = combo.tensor([true; false])
            let t5b = combo.tensor([true; true])
            let t5 = t5a + t5b
            let t5Correct = combo.tensor([true; true])
            Assert.CheckEqual(t5, t5Correct)

    [<Test>]
    member _.TestTensorAddTT_BroadcastingSystematic () =
      for combo in Combos.IntegralAndFloatingPoint do 

        // Check all broadcasts into 2x2
        // 2x2 * 1  (broadcast --> 2x2)
        // 2x2 * 2  (broadcast --> 2x2)
        // 2x2 * 2x1  (broadcast --> 2x2)
        // 2x2 * 1x2  (broadcast --> 2x2)
        let t6a = combo.tensor([ [1.; 2.]; [3.; 4.] ])
        for t6b in [ combo.tensor([ 5.0 ])
                     combo.tensor([ 5.0; 5.0 ])
                     combo.tensor([ [5.0]; [5.0] ])
                     combo.tensor([ [5.0; 5.0] ]) ] do
            let t6 = t6a + t6b
            let t6Commute = t6b + t6a
            let t6Correct = combo.tensor([ [6.; 7.]; [8.; 9.] ])

            Assert.CheckEqual(t6Correct, t6)
            Assert.CheckEqual(t6Correct, t6Commute)

        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 + 1  (broadcast --> 2x3x4)
        // 2x3x4 + 4  (broadcast --> 2x3x4)
        // 2x3x4 + 1x1  (broadcast --> 2x3x4)
        // 2x3x4 + 3x1  (broadcast --> 2x3x4)
        // 2x3x4 + 1x4  (broadcast --> 2x3x4)
        // etc.
        let t7a = combo.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
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
                  let t7b = combo.tensor(arrayND shape (fun is -> double (Array.sum is) + 2.0))
                  let t7 = t7a + t7b
                  let t7Commute = t7b + t7a
                  yield (t7b, t7), (t7b, t7Commute) |]
            |> Array.unzip

        let t7Expected =
            [|(combo.tensor 2.,                                                       combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [2.],                                                     combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [2., 3., 4., 5.],                                         combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (combo.tensor [[2.]],                                                   combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [[2., 3., 4., 5.]],                                       combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (combo.tensor [[2.], [3.], [4.]],                                       combo.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[15., 16., 17., 18.], [20., 21., 22., 23.], [25., 26., 27., 28.]]]);
              (combo.tensor [[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]],   combo.tensor [[[3., 5., 7., 9.], [8., 10., 12., 14.], [13., 15., 17., 19.]], [[15., 17., 19., 21.], [20., 22., 24., 26.], [25., 27., 29., 31.]]]);
              (combo.tensor [[[2.]]],                                                 combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [[[2., 3., 4., 5.]]],                                     combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (combo.tensor [[[2.], [3.], [4.]]],                                     combo.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[15., 16., 17., 18.], [20., 21., 22., 23.], [25., 26., 27., 28.]]]);
              (combo.tensor [[[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]], combo.tensor [[[3., 5., 7., 9.], [8., 10., 12., 14.], [13., 15., 17., 19.]], [[15., 17., 19., 21.], [20., 22., 24., 26.], [25., 27., 29., 31.]]]);
              (combo.tensor [[[2.]], [[3.]]],                                         combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[16., 17., 18., 19.], [20., 21., 22., 23.], [24., 25., 26., 27.]]]);
              (combo.tensor [[[2., 3., 4., 5.]], [[3., 4., 5., 6.]]],                 combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[16., 18., 20., 22.], [20., 22., 24., 26.], [24., 26., 28., 30.]]]);
              (combo.tensor [[[2.], [3.], [4.]], [[3.], [4.], [5.]]],                 combo.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[16., 17., 18., 19.], [21., 22., 23., 24.], [26., 27., 28., 29.]]])|]


        Assert.CheckEqual(t7Expected, t7Results)
        Assert.CheckEqual(t7Expected, t7CommuteResults)



    [<Test>]
    member _.TestTensorStackTs () =
      for combo in Combos.All do 
        let t0a = combo.tensor(1.)
        let t0b = combo.tensor(3.)
        let t0c = combo.tensor(5.)
        let t0 = Tensor.stack([t0a;t0b;t0c])
        let t0Correct = combo.tensor([1.;3.;5.])

        let t1a = combo.tensor([1.; 2.])
        let t1b = combo.tensor([3.; 4.])
        let t1c = combo.tensor([5.; 6.])
        let t1 = Tensor.stack([t1a;t1b;t1c])

        let t2a = combo.tensor([ [1.; 2.] ])
        let t2b = combo.tensor([ [3.; 4.] ])
        let t2c = combo.tensor([ [5.; 6.] ])
        let t2_dim0 = Tensor.stack([t2a;t2b;t2c], dim=0)
        let t2_dim1 = Tensor.stack([t2a;t2b;t2c], dim=1)
        let t2_dim2 = Tensor.stack([t2a;t2b;t2c], dim=2)
        let t2Correct_dim0 = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t2Correct_dim1 = combo.tensor([[[1.;2.];[3.;4.];[5.;6.]]])
        let t2Correct_dim2 = combo.tensor([[[1.;3.;5.];[2.;4.;6.]]])

        let t1Correct = combo.tensor([[1.;2.];[3.;4.];[5.;6.]])

        Assert.CheckEqual(t0Correct, t0)
        Assert.CheckEqual(t1Correct, t1)
        Assert.CheckEqual(t0.dtype, combo.dtype)
        Assert.CheckEqual(t1.dtype, combo.dtype)

        Assert.CheckEqual(t2Correct_dim0, t2_dim0)
        Assert.CheckEqual(t2Correct_dim1, t2_dim1)
        Assert.CheckEqual(t2Correct_dim2, t2_dim2)

    [<Test>]
    member _.TestTensorUnstackT () =
        for combo in Combos.All do 
            let t0a = combo.tensor(1.)
            let t0b = combo.tensor(3.)
            let t0c = combo.tensor(5.)
            let t0Correct = [t0a;t0b;t0c]
            let t0 = Tensor.stack(t0Correct).unstack()

            let t1a = combo.tensor([1.; 2.])
            let t1b = combo.tensor([3.; 4.])
            let t1c = combo.tensor([5.; 6.])
            let t1Correct = [t1a;t1b;t1c]
            let t1 = Tensor.stack(t1Correct).unstack()

            // 3x1x2
            let t2a = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t2 = t2a.unstack()
            let t2_dim1 = t2a.unstack(dim=1)
            let t2_dim2 = t2a.unstack(dim=2)
            // 3 of 1x2
            let t2Correct = [combo.tensor [[1.;2.]]; combo.tensor [[3.;4.]]; combo.tensor [[5.;6.]]]
            // 1 of 3x2
            let t2Correct_dim1 = [combo.tensor [[1.;2.];[3.;4.];[5.;6.]]]
            // 2 of 3x1
            let t2Correct_dim2 = [combo.tensor [[1.];[3.];[5.]]; combo.tensor [[2.];[4.];[6.]]]

            Assert.CheckEqual(t0Correct, Seq.toList t0)
            Assert.CheckEqual(t1Correct, Seq.toList t1)
            for t in t1 do 
                Assert.CheckEqual(t.dtype, combo.dtype)
            Assert.CheckEqual(t2Correct, Array.toList t2)
            Assert.CheckEqual(t2Correct_dim1, Array.toList t2_dim1)
            Assert.CheckEqual(t2Correct_dim2, Array.toList t2_dim2)

    [<Test>]
    member _.TestTensorCatTs () =
        for combo in Combos.All do 

            let t0a = combo.tensor([1.; 2.])
            let t0 = Tensor.cat([t0a])
            let t0Correct = combo.tensor([1.;2.])

            Assert.CheckEqual(t0Correct, t0)

            let t1a = combo.tensor([1.; 2.]) // 2
            let t1b = combo.tensor([3.; 4.]) // 2
            let t1c = combo.tensor([5.; 6.]) // 2
            let t1 = Tensor.cat([t1a;t1b;t1c]) // 6
            let t1_dim0 = Tensor.cat([t1a;t1b;t1c],dim=0) // 6
            let t1Correct = combo.tensor([1.;2.;3.;4.;5.;6.])

            Assert.CheckEqual(t1Correct, t1)
            Assert.CheckEqual(t1Correct, t1_dim0)

            let t2a = combo.tensor([ [1.; 2.] ]) // 1x2
            let t2b = combo.tensor([ [3.; 4.] ]) // 1x2
            let t2c = combo.tensor([ [5.; 6.] ]) // 1x2
            let t2 = Tensor.cat([t2a;t2b;t2c]) // 3x2
            let t2_dim0 = Tensor.cat([t2a;t2b;t2c], dim=0) // 3x2
            let t2_dim1 = Tensor.cat([t2a;t2b;t2c], dim=1) // 1x6
            let t2Correct_dim0 = combo.tensor([[1.;2.];[3.;4.];[5.;6.]]) // 3x2
            let t2Correct_dim1 = combo.tensor([[1.;2.;3.;4.;5.;6.]]) // 1x6

            Assert.CheckEqual(t2Correct_dim0, t2)
            Assert.CheckEqual(t2Correct_dim0, t2_dim0)
            Assert.CheckEqual(t2Correct_dim1, t2_dim1)

            // irregular sizes dim0
            let t3a = combo.tensor([ [1.; 2.] ]) // 1x2
            let t3b = combo.tensor([ [3.; 4.];[5.; 6.] ]) // 2x2
            let t3c = combo.tensor([ [7.; 8.] ]) // 1x2
            let t3 = Tensor.cat([t3a;t3b;t3c]) // 4x2
            let t3Correct = combo.tensor([[1.;2.];[3.;4.];[5.;6.];[7.;8.]]) // 4x2

            Assert.CheckEqual(t3Correct, t3)

            // irregular sizes dim1
            let t4a = combo.tensor([ [1.]; [2.] ]) // 2x1
            let t4b = combo.tensor([ [3.; 4.];[5.; 6.] ]) // 2x2
            let t4c = combo.tensor([ [7.]; [8.] ]) // 2x1
            let t4_dim1 = Tensor.cat([t4a;t4b;t4c],dim=1) // 2x4
            let t4Correct_dim1 = combo.tensor([[1.;3.;4.;7.];[2.;5.;6.;8.]]) // 2x4

            Assert.CheckEqual(t4Correct_dim1, t4_dim1)

    [<Test>]
    member _.TestTensorSplitT_Basics () =
        
        for combo in Combos.All do 
            //6 --> 2;2;2
            let t1in = combo.tensor([1.;2.;3.;4.;5.;6.]) // 6
            let t1 = t1in.split([2;2;2]) |> Seq.toList // 3 of 2
            let t1Correct = [combo.tensor([1.; 2.]);combo.tensor([3.; 4.]);combo.tensor([5.; 6.])]

            Assert.CheckEqual(t1Correct, t1)

            // 3x1x2
            let t2in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t2 = t2in.split(sizes=[1;1;1], dim=0)  |> Seq.toList // 3 of 1x1x2
            let t2Correct = [combo.tensor [[[1.;2.]]]; combo.tensor [[[3.;4.]]]; combo.tensor [[[5.;6.]]]]

            Assert.CheckEqual(t2Correct, t2)

            let t3in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t3 = t3in.split(sizes=[1;2], dim=0)  |> Seq.toList // 2 of 1x1x2 and 2x1x2
            let t3Correct = [combo.tensor [[[1.;2.]]]; combo.tensor [[[3.;4.]];[[5.;6.]]]]

            Assert.CheckEqual(t3Correct, t3)

            let t4in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t4 = t4in.split(sizes=[1], dim=1)  |> Seq.toList // 1 of 3x1x2
            let t4Correct = [combo.tensor [[[1.;2.]];[[3.;4.]];[[5.;6.]]]] // 1 of 3x1x2

            Assert.CheckEqual(t4Correct, t4)

            let t5in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t5 = t5in.split(sizes=[1;1], dim=2)  |> Seq.toList // 2 of 3x1x1
            let t5Correct = [combo.tensor [[[1.]];[[3.]];[[5.]]]; combo.tensor [[[2.]];[[4.]];[[6.]]]] // 2 of 3x1x1

            Assert.CheckEqual(t5Correct, t5)

            //systematic split of 6 
            let t6vs = [1..6]
            let t6in = combo.tensor(t6vs) // 6
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
                          [if p1 > 0 then combo.tensor(t6vs.[0..p1-1]);
                           if p2 > 0 then combo.tensor(t6vs.[p1..p1+p2-1]);
                           if p3 > 0 then combo.tensor(t6vs.[p1+p2..])]

                      Assert.CheckEqual(t6Correct, t6)


            //systematic split of 2x6 along dim1
            let t7vs1 = [1..6]
            let t7vs2 = [7..12]
            let t7in = combo.tensor([ t7vs1; t7vs2] ) // 2x6
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
                          [if p1 > 0 then combo.tensor([ t7vs1.[0..p1-1];     t7vs2.[0..p1-1] ]);
                           if p2 > 0 then combo.tensor([ t7vs1.[p1..p1+p2-1]; t7vs2.[p1..p1+p2-1] ]);
                           if p3 > 0 then combo.tensor([ t7vs1.[p1+p2..];     t7vs2.[p1+p2..] ])]

                      Assert.CheckEqual(t7Correct, t7)



    [<Test>]
    member _.TestTensorAddT2T1 () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([[1.; 2.]; [3.; 4.]]) + combo.tensor([5.; 6.])
            let t1Correct = combo.tensor([[6.; 8.]; [8.; 10.]])

            Assert.CheckEqual(t1Correct, t1)
            Assert.CheckEqual(t1.dtype, combo.dtype)

        for combo in Combos.Bool do 
            // check broadcast for bool tensor 0 --> [2]
            let t6a = combo.tensor([true; false])
            let t6b = combo.tensor(true)
            let t6 = t6a + t6b
            let t6Correct = combo.tensor([true; true])
            Assert.CheckEqual(t6, t6Correct)

            // check broadcast for bool tensor [1] --> [2]
            let t7a = combo.tensor([true; false])
            let t7b = combo.tensor([true])
            let t7 = t7a + t7b
            let t7Correct = combo.tensor([true; true])
            Assert.CheckEqual(t7, t7Correct)


    [<Test>]
    member _.TestTensorSubTT () =
        // Test all pairs of non-bool types, for widening
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in Dtypes.IntegralAndFloatingPoint do 
                match Dtype.widen combo.dtype dtype2 with 
                | None -> ()
                | Some dtypeRes -> 

                let t1 = combo.tensor([1.; 2.]) - combo.tensor([3.; 4.], dtype=dtype2)
                let t1Correct = combo.tensor([-2.; -2.], dtype=dtypeRes)

                Assert.CheckEqual(t1Correct, t1)
                Assert.CheckEqual(t1.dtype, dtypeRes)

                let t2 = combo.tensor([1.; 2.]) - combo.tensor(5., dtype=dtype2)
                let t2Correct = combo.tensor([-4.; -3.], dtype=dtypeRes)

                Assert.CheckEqual(t2Correct, t2)
                Assert.CheckEqual(t2.dtype, dtypeRes)

        // Test scalar broadcast
        for combo in Combos.IntegralAndFloatingPoint do 
            let t3 = combo.tensor([1; 2]) - 5
            let t3Correct = combo.tensor([-4; -3])

            Assert.CheckEqual(t3Correct, t3)
            Assert.CheckEqual(t3.dtype, combo.dtype)

            let t4 = 5 - combo.tensor([1; 2])
            let t4Correct = combo.tensor([4; 3])

            Assert.CheckEqual(t4Correct, t4)
            Assert.CheckEqual(t4.dtype, combo.dtype)

            let t5 = combo.tensor([1; 2]) - 5
            let t5Correct = combo.tensor([-4; -3])

            Assert.CheckEqual(t5Correct, t5)
            Assert.CheckEqual(t5.dtype, combo.dtype)

        for combo in Combos.Bool do 
            // Bool tensors do not support subtraction
            //
            //   torch.tensor([[True]], dtype=torch.bool) - torch.tensor([[True]], dtype=torch.bool)
            //
            // RuntimeError: Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.

            let t5a = combo.tensor([true; false])
            let t5b = combo.tensor([true; true])
            isInvalidOp(fun () -> t5a - t5b)

    [<Test>]
    member _.TestTensorMulTT () =
        // Test all pairs of non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in Dtypes.IntegralAndFloatingPoint do 
                match Dtype.widen combo.dtype dtype2 with 
                | None -> ()
                | Some dtypeRes -> 
                let t1 = combo.tensor([1.; 2.]) * combo.tensor([3.; 4.], dtype=dtype2)
                let t1Correct = combo.tensor([3.; 8.], dtype=dtypeRes)

                Assert.CheckEqual(t1Correct, t1)
                Assert.CheckEqual(t1.dtype, dtypeRes)

                let t2 = combo.tensor([1.; 2.]) * combo.tensor(5., dtype=dtype2)
                let t2Correct = combo.tensor([5.; 10.], dtype=dtypeRes)

                Assert.CheckEqual(t2Correct, t2)
                Assert.CheckEqual(t2.dtype, dtypeRes)

        // Test scalar broadcasting 
        for combo in Combos.FloatingPoint do 
            let t3 = combo.tensor([1.; 2.]) * 5.f
            let t3Correct = combo.tensor([5.; 10.])

            Assert.CheckEqual(t3Correct, t3)

            let t4 = 5. * combo.tensor([1.; 2.])
            let t4Correct = combo.tensor([5.; 10.])

            Assert.CheckEqual(t4Correct, t4)
            Assert.CheckEqual(t3.dtype, combo.dtype)
            Assert.CheckEqual(t4.dtype, combo.dtype)

        for combo in Combos.Integral do 
            let t3 = combo.tensor([1; 2]) * 5
            let t3Correct = combo.tensor([5; 10])

            Assert.CheckEqual(t3Correct, t3)
            Assert.CheckEqual(t3.dtype, combo.dtype)

            let t4 = 5 * combo.tensor([1; 2])
            let t4Correct = combo.tensor([5; 10])

            Assert.CheckEqual(t4Correct, t4)
            Assert.CheckEqual(t4.dtype, combo.dtype)

            // Multiplying integer tensors by a floating point number always
            // results in float32. THis is the same behaviour as Torch
            let t5 = 5.0 * combo.tensor([1; 2])
            let t5Correct = combo.tensor([5; 10], dtype=Dtype.Float32)

            Assert.CheckEqual(t5Correct, t5)
            Assert.CheckEqual(t5.dtype, Dtype.Float32)

        // Bool tensors support multiplication giving bool tensor
        //
        //    torch.ones(10, dtype=torch.bool) * torch.ones(10, dtype=torch.bool)
        //
        //    tensor([True, True, True, True, True, True, True, True, True, True])
        for combo in Combos.Bool do 
            let t1 = combo.tensor([true; true])
            let t2 = combo.tensor([true; false])
            let i = t1 * t2
            let iCorrect = combo.tensor([true; false])
            Assert.CheckEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorMulTT_BroadcastSystematic () =
      for combo in Combos.FloatingPoint do 
        // 2x2 * 1  (broadcast --> 2x2)
        // 2x2 * 2  (broadcast --> 2x2)
        // 2x2 * 2x1  (broadcast --> 2x2)
        // 2x2 * 1x2  (broadcast --> 2x2)
        let t5a = combo.tensor([ [1.; 2.]; [3.; 4.] ])
        for t5b in [ combo.tensor([ 5.0 ])
                     combo.tensor([ 5.0; 5.0 ])
                     combo.tensor([ [5.0]; [5.0] ])
                     combo.tensor([ [5.0; 5.0] ]) ] do
            let t5 = t5a * t5b
            let t5Commute = t5b * t5a
            let t5Correct = combo.tensor([ [5.; 10.]; [15.; 20.] ])

            Assert.CheckEqual(t5Correct, t5)
            Assert.CheckEqual(t5Correct, t5Commute)

        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 * 1  (broadcast --> 2x3x4)
        // 2x3x4 * 4  (broadcast --> 2x3x4)
        // 2x3x4 * 1x1  (broadcast --> 2x3x4)
        // 2x3x4 * 3x1  (broadcast --> 2x3x4)
        // 2x3x4 * 1x4  (broadcast --> 2x3x4)
        // etc.
        let t6a = combo.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
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
                  let t6b = combo.tensor( arrayND shape (fun is -> double (Array.sum is) + 2.0))
                  let t6 = t6a * t6b
                  let t6Commute = t6b * t6a
                  yield (t6b, t6 ), (t6b, t6Commute ) |]
            |> Array.unzip

        let t6Expected =
            [|(combo.tensor 2.,                                                      combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [2.],                                                    combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [2., 3., 4., 5.],                                        combo.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (combo.tensor [[2.]],                                                  combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [[2., 3., 4., 5.]],                                      combo.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (combo.tensor [[2.], [3.], [4.]],                                      combo.tensor [[[2., 4., 6., 8.], [15., 18., 21., 24.], [36., 40., 44., 48.]], [[26., 28., 30., 32.], [51., 54., 57., 60.], [84., 88., 92., 96.]]]);
              (combo.tensor [[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]],  combo.tensor [[[2., 6., 12., 20.], [15., 24., 35., 48.], [36., 50., 66., 84.]], [[26., 42., 60., 80.], [51., 72., 95., 120.], [84., 110., 138., 168.]]]);
              (combo.tensor [[[2.]]],                                                combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [[[2., 3., 4., 5.]]],                                    combo.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (combo.tensor [[[2.], [3.], [4.]]],                                    combo.tensor [[[2., 4., 6., 8.], [15., 18., 21., 24.], [36., 40., 44., 48.]], [[26., 28., 30., 32.], [51., 54., 57., 60.], [84., 88., 92., 96.]]]);
              (combo.tensor [[[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]],combo.tensor [[[2., 6., 12., 20.], [15., 24., 35., 48.], [36., 50., 66., 84.]], [[26., 42., 60., 80.], [51., 72., 95., 120.], [84., 110., 138., 168.]]]);
              (combo.tensor [[[2.]], [[3.]]],                                        combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[39., 42., 45., 48.], [51., 54., 57., 60.], [63., 66., 69., 72.]]]);
              (combo.tensor [[[2., 3., 4., 5.]], [[3., 4., 5., 6.]]],                combo.tensor [[[2., 6., 12., 20.],  [10., 18., 28., 40.], [18., 30., 44., 60.]], [[39., 56., 75., 96.], [51., 72., 95., 120.], [63., 88., 115., 144.]]]);
              (combo.tensor [[[2.], [3.], [4.]], [[3.], [4.], [5.]]],                combo.tensor [[[2., 4., 6., 8.],  [15., 18., 21., 24.], [36., 40., 44., 48.]], [[39., 42., 45., 48.], [68., 72., 76., 80.], [105., 110., 115., 120.]]]); |]

        Assert.CheckEqual(t6Expected, t6Results)
        Assert.CheckEqual(t6Expected, t6CommuteResults)


    [<Test>]
    member _.TestTensorDivTT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; 2.]) / combo.tensor([3.; 4.])
            let t1Correct = combo.tensor([0.333333; 0.5])

            let t2 = combo.tensor([1.; 2.]) / combo.tensor(5.)
            let t2Correct = combo.tensor([0.2; 0.4])

            let t3 = combo.tensor([1.; 2.]) / 5.
            let t3Correct = combo.tensor([0.2; 0.4])

            let t4 = 5. / combo.tensor([1.; 2.])
            let t4Correct = combo.tensor([5.; 2.5])

            Assert.True(t1.allclose(t1Correct, 0.01))
            Assert.True(t2.allclose(t2Correct, 0.01))
            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.True(t4.allclose(t4Correct, 0.01))
            Assert.CheckEqual(t1.dtype, combo.dtype)
            Assert.CheckEqual(t2.dtype, combo.dtype)
            Assert.CheckEqual(t3.dtype, combo.dtype)
            Assert.CheckEqual(t4.dtype, combo.dtype)

        // Integer tensors support integer division
        for combo in Combos.Integral do 
            let t1a = combo.tensor([2; 3; 4])
            let t1b = combo.tensor([1; 2; 3])
            let i1 = t1a / t1b
            let i1Correct = combo.tensor([2; 1; 1])
            Assert.CheckEqual(i1Correct, i1)

            let t2a = combo.tensor(6)
            let t2b = combo.tensor([1; 2; 3])
            let i2 = t2a / t2b
            let i2Correct = combo.tensor([6; 3; 2])
            Assert.CheckEqual(i2Correct, i2)

            let t3a = combo.tensor([6; 12; 18])
            let t3b = combo.tensor(3)
            let i3 = t3a / t3b
            let i3Correct = combo.tensor([2; 4; 6])
            Assert.CheckEqual(i3Correct, i3)

        // Bool tensors don't support /
        //
        //    torch.ones(10, dtype=torch.bool) / torch.ones(10, dtype=torch.bool)
        //
        //    RuntimeError: "div_cpu" not implemented for 'Bool'
        for combo in Combos.Bool do 
            let t2 = combo.tensor([true; false])
            isInvalidOp(fun () -> t2 / t2)

    [<Test>]
    member _.TestTensorPowTT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; 2.]) ** combo.tensor([3.; 4.])
            let t1Correct = combo.tensor([1.; 16.])

            Assert.CheckEqual(t1Correct, t1)
            Assert.CheckEqual(t1.dtype, combo.dtype)
            let t2 = combo.tensor([1.; 2.]) ** combo.tensor(5.)
            let t2Correct = combo.tensor([1.; 32.])

            Assert.CheckEqual(t2Correct, t2)
            Assert.CheckEqual(t2.dtype, combo.dtype)

            let t3 = combo.tensor(5.) ** combo.tensor([1.; 2.])
            let t3Correct = combo.tensor([5.; 25.])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.CheckEqual(t3.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            let t1 = combo.tensor([1.0])
            isInvalidOp(fun () -> t1 ** t1)

            let t2a = combo.tensor([1.0])
            let t2b = combo.tensor(1.0)
            isInvalidOp(fun () -> t2a ** t2b)

            let t3a = combo.tensor(1.0)
            let t3b = combo.tensor([1.0])
            isInvalidOp(fun () -> t3a ** t3b)

    [<Test>]
    member _.TestTensorMatMulT2T2 () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                   [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                   [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
            let t2 = combo.tensor([[5.1067; 0.0681];
                                   [7.4633; 3.6027];
                                   [9.0070; 7.3012];
                                   [2.6639; 2.8728];
                                   [7.9229; 2.3695]])

            let t3 = t1.matmul(t2)
            let t3Correct = combo.tensor([[118.0367; 56.6266];
                                          [190.5926; 90.8155];
                                          [134.3925; 64.1030]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.CheckEqual(t3.dtype, combo.dtype)

        for combo in Combos.Integral do 
            let t1 = combo.tensor([[1; 2]])
            let t2 = combo.tensor([[3]; [4]])

            let t3 = t1.matmul(t2)
            let t3Correct = combo.tensor([[11]])

            Assert.True(t3.allclose(t3Correct, 0.0))
            Assert.CheckEqual(t3.dtype, combo.dtype)

        // Matmul of Bool tensor not allowed
        //
        //    t = torch.tensor([[True]], dtype=torch.bool)
        //    t.matmul(t)
        //
        // RuntimeError: _th_mm not supported on CPUType for Bool

        for combo in Combos.Bool do 
            let t3a = combo.tensor([[true]])
            isInvalidOp(fun () -> t3a.matmul(t3a))

    [<Test>]
    member _.TestTensorDot () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([8.0766, 3.3030, -2.1732, 8.9448, 1.1028])
            let t2 = combo.tensor([5.1067, -0.0681, 7.4633, -3.6027, 9.0070])
            let t3 = dsharp.dot(t1, t2)
            let t3Correct = combo.tensor(2.5081)
            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.CheckEqual(t3.dtype, combo.dtype)

        for combo in Combos.Integral do 
            let t1 = combo.tensor([1; 2])
            let t2 = combo.tensor([3; 4])

            let t3 = dsharp.dot(t1, t2)
            let t3Correct = combo.tensor(11)

            Assert.True(t3.allclose(t3Correct, 0.0))
            Assert.CheckEqual(t3.dtype, combo.dtype)

        for combo in Combos.Bool do 
            let t3a = combo.tensor([true])
            isInvalidOp(fun () -> dsharp.dot(t3a, t3a))

    [<Test>]
    member _.TestTensorDiagonal () =
        for combo in Combos.All do
            let t1 = combo.arange(6.).view([2; 3])
            let t1a = dsharp.diagonal(t1)
            let t1b = dsharp.diagonal(t1, offset=1)
            let t1c = dsharp.diagonal(t1, offset=2)
            let t1d = dsharp.diagonal(t1, offset= -1)
            let t1aCorrect = combo.tensor([0.,4.])
            let t1bCorrect = combo.tensor([1.,5.])
            let t1cCorrect = combo.tensor([2.])
            let t1dCorrect = combo.tensor([3.])
            let t2 = combo.arange(9.).view([3;3])
            let t2a = dsharp.diagonal(t2)
            let t2aCorrect = combo.tensor([0.,4.,8.])
            Assert.CheckEqual(t1aCorrect, t1a)
            Assert.CheckEqual(t1bCorrect, t1b)
            Assert.CheckEqual(t1cCorrect, t1c)
            Assert.CheckEqual(t1dCorrect, t1d)
            Assert.CheckEqual(t2aCorrect, t2a)

    [<Test>]
    member _.TestTensorTrace () =
        for combo in Combos.FloatingPoint do
            let t1 = combo.arange(6.).view([2; 3])
            let t1a = dsharp.trace(t1)
            let t1aCorrect = combo.tensor(4.)
            let t2 = combo.arange(9.).view([3;3])
            let t2a = dsharp.trace(t2)
            let t2aCorrect = combo.tensor(12.)
            Assert.CheckEqual(t1aCorrect, t1a)
            Assert.CheckEqual(t2aCorrect, t2a)

        for combo in Combos.Integral do
            let t1 = combo.arange(6.).view([2; 3])
            let t1a = dsharp.trace(t1)
            let t1aCorrect = combo.tensor(4., dtype=Dtype.Int64)
            let t2 = combo.arange(9.).view([3;3])
            let t2a = dsharp.trace(t2)
            let t2aCorrect = combo.tensor(12., dtype=Dtype.Int64)
            Assert.CheckEqual(t1aCorrect, t1a)
            Assert.CheckEqual(t2aCorrect, t2a)

        for combo in Combos.Bool do
            let t1a = combo.tensor([[true]]).trace()
            let t1aCorrect = combo.tensor(1., dtype=Dtype.Int64)
            Assert.CheckEqual(t1aCorrect, t1a)

    [<Test>]
    member _.TestTensorMatMul11 () =
        let t1 = dsharp.tensor([8.0766; 3.3030; 2.1732; 8.9448; 1.1028])
        let t2 = dsharp.tensor([5.1067; 7.4633; 3.6027; 9.0070; 7.3012])
        let t3 = t1.matmul(t2)
        let t3Correct = t1.dot(t2)

        Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul12 () =
        let t1 = dsharp.tensor([8.0766; 3.3030; 2.1732; 8.9448; 1.1028])
        let t2 = dsharp.tensor([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])
        let t3 = t1.matmul(t2)
        let t3Correct = t1.expand([1;5]).matmul(t2).squeeze(0)

        Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul13 () =
        // 5 --> 1x5 --> 3x1x5 (batching expansion)
        let t1 = dsharp.tensor([8.0766; 3.3030; 2.1732; 8.9448; 1.1028])
        
        // 3x5x2 (batch dimension is 3)
        let t2 = dsharp.tensor([[[5.1067; 0.0681];
                                 [7.4633; 3.6027];
                                 [9.0070; 7.3012];
                                 [2.6639; 2.8728];
                                 [7.9229; 2.3695]];
                                [[1.1067; 0.0681];
                                 [2.4633; 3.6027];
                                 [3.0070; 7.3012];
                                 [4.6639; 2.8728];
                                 [5.9229; 2.3695]];
                                [[7.1067; 0.0681];
                                 [8.4633; 3.6027];
                                 [7.0070; 7.3012];
                                 [8.6639; 2.8728];
                                 [7.9229; 2.3695]]])
        let t3 = t1.matmul(t2)
        let t3Correct = t1.expand([3;1;5]).matmul(t2).squeeze(1)

        Assert.AreEqual([|3;2|], t3.shape)
        Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul21 () =
        let t1 = dsharp.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [5.1067; 7.4633; 3.6027; 9.0070; 7.3012]])
        let t2 = dsharp.tensor([0.0681; 3.6027; 7.3012; 2.8728; 2.3695])
        let t3 = t1.matmul(t2)
        let t3Correct = t1.matmul(t2.unsqueeze(1)).squeeze(1)

        Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul31 () =
        //2 x 2 x 5
        let t1 = dsharp.tensor([[[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                 [5.1067; 7.4633; 3.6027; 9.0070; 7.3012]];
                                [[9.0766; 4.3030; 2.1732; 8.9448; 1.1028];
                                 [3.1067; 5.4633; 3.6027; 9.0070; 7.3012]]])
        
        // 5 --> 5x1 (matmul expand) -> 2x5x1 (batch expand)
        let t2 = dsharp.tensor([0.0681; 3.6027; 7.3012; 2.8728; 2.3695])
        // 2x2x5 * 2x5x1 --> 2x2x1 --> 2x2 (reverse matmul expand)
        let t3 = t1.matmul(t2)
        let t3Correct = t1.matmul(t2.unsqueeze(1)).squeeze(2)

        Assert.AreEqual([|2;2|], t3.shape)
        Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul33 () =
        let t1 = dsharp.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
        let t2 = dsharp.tensor([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])

        let t1Expanded = t1.expand([| 6;3;5 |])
        let t2Expanded = t2.expand([| 6;5;2 |])
        let t3Unexpanded = t1.matmul(t2)
        let t3 = t1Expanded.matmul(t2Expanded)
        let t3Correct = t3Unexpanded.expand([| 6;3;2 |])

        Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member this.TestTensorMatMul44 () =
        let t1 = dsharp.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
        let t2 = dsharp.tensor([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])

        let t1Expanded = t1.expand([| 2;6;3;5 |])
        let t2Expanded = t2.expand([| 2;6;5;2 |])
        let t3Unexpanded = t1.matmul(t2)
        let t3 = t1Expanded.matmul(t2Expanded)
        let t3Correct = t3Unexpanded.expand([| 2;6;3;2 |])

        Assert.True(t3.allclose(t3Correct, 0.0001))

    [<Test>]
    member this.TestTensorMatMulBroadcast1 () =
        let t1 = dsharp.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
        let t2 = dsharp.tensor([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])

        let t1Expanded = t1.expand([| 3;5 |])
        let t2Expanded = t2.expand([| 2;6;5;2 |])
        let t3Unexpanded = t1.matmul(t2)
        let t3 = t1Expanded.matmul(t2Expanded)
        let t3Correct = t3Unexpanded.expand([| 2;6;3;2 |])

        Assert.True(t3.allclose(t3Correct, 0.00001))

    [<Test>]
    member this.TestTensorMatMulBroadcast2 () =
        let t1 = dsharp.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
        let t2 = dsharp.tensor([[5.1067; 0.0681];
                                [7.4633; 3.6027];
                                [9.0070; 7.3012];
                                [2.6639; 2.8728];
                                [7.9229; 2.3695]])

        let t1Expanded = t1.expand([| 2;6;3;5 |])
        let t2Expanded = t2.expand([| 2;1;5;2 |])
        let t3Unexpanded = t1.matmul(t2)
        let t3 = t1Expanded.matmul(t2Expanded)
        let t3Correct = t3Unexpanded.expand([| 2;6;3;2 |])

        Assert.True(t3.allclose(t3Correct, 0.00001))

    member _.TestTensorMaxPool1D () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([[[-2.1704, -1.1558,  2.5995,  1.3858, -1.3157, -0.3179,  0.9593,  -2.1432,  0.7169, -1.7999],
                                     [ 0.4564, -0.2262,  0.3495,  0.4587, -0.3858,  0.2349,  0.2978,  0.6288,  1.1539,  0.2121]],

                                    [[ 0.6654,  0.7151,  0.9980,  0.1321, -2.0009, -1.1897,  1.0608,  -1.8059, -0.2344,  1.6387],
                                     [ 1.1872, -2.2679, -0.0297, -0.2067, -1.5622, -0.3916,  0.6039,  -1.1469,  0.4560,  1.2069]]])

            let tk3, tk3i = dsharp.maxpool1di(t, 3)
            let tk3Correct = combo.tensor([[[ 2.5995,  1.3858,  0.9593],
                                              [ 0.4564,  0.4587,  1.1539]],
                                     
                                             [[ 0.9980,  0.1321,  1.0608],
                                              [ 1.1872, -0.2067,  0.6039]]])
            let tk3iCorrect = combo.tensor([[[2, 3, 6],
                                              [0, 3, 8]],
                                     
                                             [[2, 3, 6],
                                              [0, 3, 6]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk3Correct, tk3)
            Assert.CheckEqual(tk3iCorrect, tk3i)

            let tk3p1, tk3p1i = dsharp.maxpool1di(t, 3, padding=1)
            let tk3p1Correct = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                                [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                       
                                               [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                                [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk3p1iCorrect = combo.tensor([[[1, 2, 6, 8],
                                                [0, 3, 7, 8]],
                                       
                                               [[1, 2, 6, 9],
                                                [0, 2, 6, 9]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk3p1iCorrect, tk3p1i)
            Assert.CheckEqual(tk3p1Correct, tk3p1)

            let tk3s2, tk3s2i = dsharp.maxpool1di(t, 3, stride=2)
            let tk3s2Correct = combo.tensor([[[ 2.5995,  2.5995,  0.9593,  0.9593],
                                              [ 0.4564,  0.4587,  0.2978,  1.1539]],
                                     
                                             [[ 0.9980,  0.9980,  1.0608,  1.0608],
                                              [ 1.1872, -0.0297,  0.6039,  0.6039]]])
            let tk3s2iCorrect = combo.tensor([[[2, 2, 6, 6],
                                                  [0, 3, 6, 8]],
                                         
                                                 [[2, 2, 6, 6],
                                                  [0, 2, 6, 6]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk3s2iCorrect, tk3s2i)
            Assert.CheckEqual(tk3s2Correct, tk3s2)

            let tk4s3p2, tk4s3p2i = dsharp.maxpool1di(t, 4, stride=3, padding=2)
            let tk4s3p2Correct = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                                  [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                         
                                                 [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                                  [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk4s3p2iCorrect = combo.tensor([[[1, 2, 6, 8],
                                                  [0, 3, 7, 8]],
                                         
                                                 [[1, 2, 6, 9],
                                                  [0, 2, 6, 9]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk4s3p2iCorrect, tk4s3p2i)
            Assert.CheckEqual(tk4s3p2Correct, tk4s3p2)

        for combo in Combos.IntegralAndBool do 
            let x = combo.zeros([1;4;4])
            isInvalidOp(fun () -> dsharp.maxpool1d(x,3))

    [<Test>]
    member _.TestTensorMaxPool2D () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([[[[ 0.7372,  0.7090,  0.9216,  0.3363,  1.0141, -0.7642,  0.3801, -0.9568],
                                      [-0.3520, -1.2336,  1.8489,  0.9929, -0.8138,  0.0978, -1.3206, -1.5434],
                                      [ 0.6883, -0.2346,  0.1735,  0.6695, -1.9122,  1.1338, -0.1248,  0.2164],
                                      [-1.1349,  0.3008, -0.1635, -1.0362, -0.6487, -0.8422, -0.4334,  1.0604],
                                      [-2.1562, -0.1079,  0.5744, -0.7275,  1.0254, -0.0508, -0.0525, -0.0746],
                                      [-0.7494,  0.6819, -1.7327, -0.4838, -0.6120,  1.6331,  0.1797, -0.6068],
                                      [ 0.6400,  0.1389,  0.3033,  0.3195,  0.9934,  1.2455, -1.0953,  0.9922],
                                      [ 0.2375,  0.6003, -1.1614,  1.0146,  0.2100, -1.0145, -0.1933,  1.1415]],

                                     [[-0.0819,  0.2091,  0.4351,  1.7527, -1.1970,  2.1048,  1.0200, -0.5153],
                                      [ 1.0867, -1.8738, -0.2754, -0.5089,  0.8850, -0.4751, -0.7820,  1.4476],
                                      [-0.9072,  0.9977, -0.9106, -0.3171, -1.2444,  0.7102,  0.5656,  1.2660],
                                      [ 0.1986, -0.4967,  0.2384, -0.6551,  1.0156,  0.0520, -0.1964,  1.1367],
                                      [ 0.8948,  2.2070,  0.9938,  0.5311, -1.0674,  0.3894,  0.4192, -0.6235],
                                      [ 2.7646, -0.6509,  0.4669, -1.8774, -0.6341,  0.5113,  1.2398,  2.5090],
                                      [ 1.0722,  0.8162, -2.3271,  1.3826,  1.3832,  0.6205, -0.9138, -0.8237],
                                      [-0.0688, -1.6786,  0.1672, -0.7255, -0.1228, -0.1603, -2.1906, -2.6372]]],


                                    [[[-1.0461,  0.4063,  0.2085, -0.7598, -1.3893, -0.8866,  1.0594, -0.6184],
                                      [ 2.1120, -0.6475, -0.3964,  0.0378,  0.0138, -0.1672,  0.9265, -1.7734],
                                      [-0.2313,  0.6284, -0.0508, -0.1014, -0.5059,  0.8666, -0.7010, -0.5073],
                                      [ 0.1709,  0.2466,  0.1781, -1.6740, -0.0251, -1.4144, -2.1012,  0.3922],
                                      [ 0.9141,  0.6582, -0.0826, -0.7104,  1.7133,  1.2406,  1.1415, -0.6222],
                                      [-2.1525, -0.2996, -1.3787,  0.0336, -1.4643,  0.6534,  0.3996,  0.3145],
                                      [-0.3298,  0.3855, -0.5100,  1.2770,  0.5306, -0.6604, -0.0489,  0.0609],
                                      [-0.1552, -1.1218, -0.8435,  0.2365,  1.4428,  0.4234, -1.1083, -1.3874]],

                                     [[ 0.0511,  0.1216, -1.0103, -1.2529,  1.7200, -0.0225,  0.7446, -0.8076],
                                      [ 0.2543,  1.4250,  0.7869,  0.0526, -2.1598,  1.8228, -0.4628,  1.4234],
                                      [ 0.5492,  0.8668,  0.2120,  0.6599, -1.0934, -1.3726,  0.4788, -0.1171],
                                      [ 0.5121,  1.2607, -0.4565,  0.5448, -2.5025, -0.5503, -1.3373,  0.1711],
                                      [-0.3939, -0.6382, -0.0899, -1.4706,  0.4580,  0.3304,  1.8958,  0.1178],
                                      [ 0.1109,  0.2468,  0.3485, -0.0960, -0.0432, -0.3026, -1.9750,  0.4057],
                                      [-1.1117, -0.3422,  1.2130, -1.1206,  0.9506, -0.7723,  0.3162, -0.5487],
                                      [ 0.6304, -0.9149,  0.6075, -0.5371,  1.5875, -0.2979, -0.5832, -3.0311]]]])

            let tk3, tk3i = dsharp.maxpool2di(t, 3)
            let tk3Correct = combo.tensor([[[[1.8489, 1.1338],
                                              [0.6819, 1.6331]],

                                             [[1.0867, 2.1048],
                                              [2.7646, 1.0156]]],


                                            [[[2.1120, 0.8666],
                                              [0.9141, 1.7133]],

                                             [[1.4250, 1.8228],
                                              [1.2607, 0.5448]]]])
            let tk3iCorrect = combo.tensor([[[[10, 21],
                                                  [41, 45]],

                                                 [[ 8,  5],
                                                  [40, 28]]],


                                                [[[ 8, 21],
                                                  [32, 36]],

                                                 [[ 9, 13],
                                                  [25, 27]]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk3Correct, tk3)
            Assert.CheckEqual(tk3iCorrect, tk3i)

            let tk3p1, tk3p1i = dsharp.maxpool2di(t, 3, padding=1)
            let tk3p1Correct = combo.tensor([[[[0.7372, 1.8489, 0.3801],
                                                  [0.6883, 1.0254, 1.1338],
                                                  [0.6819, 1.0146, 1.6331]],

                                                 [[1.0867, 1.7527, 2.1048],
                                                  [2.2070, 1.0156, 1.2660],
                                                  [2.7646, 1.3832, 2.5090]]],


                                                [[[2.1120, 0.2085, 1.0594],
                                                  [0.9141, 1.7133, 1.2406],
                                                  [0.3855, 1.4428, 0.6534]],

                                                 [[1.4250, 1.7200, 1.8228],
                                                  [1.2607, 0.6599, 1.8958],
                                                  [0.6304, 1.5875, 0.4057]]]])
            let tk3p1iCorrect = combo.tensor([[[[ 0, 10,  6],
                                                  [16, 36, 21],
                                                  [41, 59, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 23],
                                                  [40, 52, 47]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 37],
                                                  [49, 60, 45]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [56, 60, 47]]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk3p1iCorrect, tk3p1i)
            Assert.CheckEqual(tk3p1Correct, tk3p1)

            let tk3s2, tk3s2i = dsharp.maxpool2di(t, 3, stride=2)
            let tk3s2Correct = combo.tensor([[[[1.8489, 1.8489, 1.1338],
                                                  [0.6883, 1.0254, 1.1338],
                                                  [0.6819, 1.0254, 1.6331]],

                                                 [[1.0867, 1.7527, 2.1048],
                                                  [2.2070, 1.0156, 1.0156],
                                                  [2.7646, 1.3832, 1.3832]]],


                                                [[[2.1120, 0.2085, 1.0594],
                                                  [0.9141, 1.7133, 1.7133],
                                                  [0.9141, 1.7133, 1.7133]],

                                                 [[1.4250, 1.7200, 1.8228],
                                                  [1.2607, 0.6599, 1.8958],
                                                  [1.2130, 1.2130, 1.8958]]]])
            let tk3s2iCorrect = combo.tensor([[[[10, 10, 21],
                                                  [16, 36, 21],
                                                  [41, 36, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 28],
                                                  [40, 52, 52]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 36],
                                                  [32, 36, 36]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [50, 50, 38]]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk3s2iCorrect, tk3s2i)
            Assert.CheckEqual(tk3s2Correct, tk3s2)

            let tk4s3p2, tk4s3p2i = dsharp.maxpool2di(t, 4, stride=3, padding=2)
            let tk4s3p2Correct = combo.tensor([[[[0.7372, 1.8489, 1.0141],
                                                  [0.6883, 1.8489, 1.1338],
                                                  [0.6819, 1.0254, 1.6331]],

                                                 [[1.0867, 1.7527, 2.1048],
                                                  [2.2070, 2.2070, 1.4476],
                                                  [2.7646, 2.2070, 2.5090]]],


                                                [[[2.1120, 0.4063, 1.0594],
                                                  [2.1120, 1.7133, 1.7133],
                                                  [0.9141, 1.7133, 1.7133]],

                                                 [[1.4250, 1.7200, 1.8228],
                                                  [1.4250, 1.4250, 1.8958],
                                                  [0.6304, 1.5875, 1.8958]]]])
            let tk4s3p2iCorrect = combo.tensor([[[[ 0, 10,  4],
                                                      [16, 10, 21],
                                                      [41, 36, 45]],

                                                     [[ 8,  3,  5],
                                                      [33, 33, 15],
                                                      [40, 33, 47]]],


                                                    [[[ 8,  1,  6],
                                                      [ 8, 36, 36],
                                                      [32, 36, 36]],

                                                     [[ 9,  4, 13],
                                                      [ 9,  9, 38],
                                                      [56, 60, 38]]]], dtype=Dtype.Int32)
            Assert.CheckEqual(tk4s3p2iCorrect, tk4s3p2i)
            Assert.CheckEqual(tk4s3p2Correct, tk4s3p2)

        for combo in Combos.IntegralAndBool do 
            let x = combo.zeros([4;4;4;4])
            isInvalidOp(fun () -> dsharp.maxpool2d(x,3))

    [<Test>]
    member _.TestTensorMaxPool3D () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([[[[ 0.4633,  0.9173,  0.4568, -1.7660, -0.1077],
                                       [-2.1112,  1.5542,  0.5720, -1.0952, -1.8144],
                                       [ 0.3505, -0.9843, -2.5655, -0.9835,  1.2303],
                                       [ 0.8156,  1.5415,  1.3066, -1.1820,  0.2060],
                                       [ 0.0684,  1.5936,  0.2956, -0.5176, -1.6960]],

                                      [[-1.7281, -0.7697, -2.2310,  0.3580,  0.6299],
                                       [ 0.8558, -0.6180, -1.6077, -0.6779,  1.2910],
                                       [ 0.1885, -0.7006, -0.1863, -1.6729, -0.5761],
                                       [ 0.1940, -0.0399,  0.9329,  1.0687,  0.0955],
                                       [-1.0189,  0.4046,  1.1762,  0.3842,  0.6831]],

                                      [[ 0.2996,  0.5738,  0.0369,  0.2835, -0.2363],
                                       [ 0.6847, -0.4949, -0.3974,  0.6808, -1.2942],
                                       [ 1.0910, -0.0594, -0.0037, -0.3355, -1.5056],
                                       [-0.0965,  1.1358,  1.2851, -1.7333, -1.1705],
                                       [ 0.0966, -1.2780,  1.2939,  1.3469, -0.2603]],

                                      [[-0.5270,  1.1442,  0.1259, -1.2813,  0.3536],
                                       [ 0.1579,  0.0828,  1.3531, -0.9110, -0.8747],
                                       [ 0.2473, -0.1507, -0.4880,  0.4575,  1.1186],
                                       [ 2.0900,  1.0479, -0.7209, -1.6928,  1.8761],
                                       [ 2.2015, -0.5097,  0.7364, -1.5177,  0.9212]],

                                      [[ 1.0358,  1.6584, -1.9654, -1.3971,  1.5641],
                                       [ 0.4032,  0.7737,  0.9351, -0.5245,  0.0783],
                                       [-1.2932, -0.9885, -1.1850, -0.7403,  0.1739],
                                       [-0.5471,  0.5017, -1.0571,  1.7574, -0.0911],
                                       [ 0.6944, -1.2772,  0.7473, -1.0983,  1.1462]]],


                                     [[[-1.2563,  0.0688,  1.0405, -0.2582,  0.7333],
                                       [ 2.0711, -0.1815,  0.8876, -0.2907,  1.1195],
                                       [-0.3912,  0.3624,  1.0576, -0.4748, -1.4021],
                                       [ 1.2176, -0.6160, -0.3471,  1.1689,  0.5677],
                                       [-0.0639,  0.3765, -0.2614,  1.8267,  0.0315]],

                                      [[ 1.2927,  1.0709, -0.8808,  0.8106, -0.5315],
                                       [ 0.7614, -0.3935,  1.2451, -0.0598, -0.5887],
                                       [-0.4089, -0.8598,  0.2478,  0.1282, -0.2745],
                                       [-0.4139, -1.2905, -0.2625, -2.0453,  1.8941],
                                       [-0.2400, -1.2830, -0.3503, -0.8536, -0.5927]],

                                      [[ 0.8200,  1.8860, -0.5216, -0.9590, -0.9760],
                                       [-1.5796,  2.2379, -0.5714, -1.5612,  1.4035],
                                       [-0.6434, -1.2257,  0.1408,  0.3781, -2.2344],
                                       [ 0.4963,  0.2431,  0.6835,  0.0047,  1.3374],
                                       [-1.5899,  2.5382,  0.9503,  1.9080,  1.8315]],

                                      [[ 0.5853,  1.9343, -0.7472,  2.1774, -2.1895],
                                       [-0.6187, -0.2870,  1.2485,  2.4069, -0.2632],
                                       [-1.6047, -0.3379,  0.5372,  1.7098,  1.6220],
                                       [ 0.5255,  0.2564, -1.8615,  1.5519, -0.5655],
                                       [-0.9452, -1.1828, -1.8192,  1.1349,  0.9806]],

                                      [[-1.8198,  0.5455,  1.1761,  1.3070, -0.4654],
                                       [ 1.2673,  0.2608,  0.8385, -1.0407, -0.6288],
                                       [-0.3860,  1.3343,  1.3084,  0.5794,  0.4639],
                                       [ 0.4750, -0.9006, -1.5002,  0.8689, -0.0379],
                                       [ 0.2891,  0.0195, -0.0503, -0.3235,  1.5407]]]]).unsqueeze(0)

            let tk2, tk2i = dsharp.maxpool3di(t, 2)
            let tk2Correct = combo.tensor([[[[1.5542, 0.5720],
                                                [1.5415, 1.3066]],
                                     
                                               [[1.1442, 1.3531],
                                                [2.0900, 1.2851]]],
                                     
                                     
                                              [[[2.0711, 1.2451],
                                                [1.2176, 1.1689]],
                                     
                                               [[2.2379, 2.4069],
                                                [0.5255, 1.7098]]]]).unsqueeze(0)
            let tk2iCorrect = combo.tensor([[[[ 6,  7],
                                                [16, 17]],
                                     
                                               [[76, 82],
                                                [90, 67]]],
                                     
                                     
                                              [[[ 5, 32],
                                                [15, 18]],
                                     
                                               [[56, 83],
                                                [90, 88]]]], dtype=Dtype.Int32).unsqueeze(0)
            Assert.CheckEqual(tk2Correct, tk2)
            Assert.CheckEqual(tk2iCorrect, tk2i)

            let tk2p1, tk2p1i = dsharp.maxpool3di(t, 2, padding=1)
            let tk2p1Correct = combo.tensor([[[[ 0.4633,  0.9173, -0.1077],
                                                [ 0.3505,  1.5542,  1.2303],
                                                [ 0.8156,  1.5936,  0.2060]],
                                     
                                               [[ 0.2996,  0.5738,  0.6299],
                                                [ 1.0910, -0.0037,  1.2910],
                                                [ 0.1940,  1.2939,  1.3469]],
                                     
                                               [[ 1.0358,  1.6584,  1.5641],
                                                [ 0.4032,  1.3531,  1.1186],
                                                [ 2.2015,  1.0479,  1.8761]]],
                                     
                                     
                                              [[[-1.2563,  1.0405,  0.7333],
                                                [ 2.0711,  1.0576,  1.1195],
                                                [ 1.2176,  0.3765,  1.8267]],
                                     
                                               [[ 1.2927,  1.8860,  0.8106],
                                                [ 0.7614,  2.2379,  1.4035],
                                                [ 0.4963,  2.5382,  1.9080]],
                                     
                                               [[ 0.5853,  1.9343,  2.1774],
                                                [ 1.2673,  1.3343,  2.4069],
                                                [ 0.5255,  0.2564,  1.5519]]]]).unsqueeze(0)
            let tk2p1iCorrect = combo.tensor([[[[  0,   1,   4],
                                                    [ 10,   6,  14],
                                                    [ 15,  21,  19]],
                                         
                                                   [[ 50,  51,  29],
                                                    [ 60,  62,  34],
                                                    [ 40,  72,  73]],
                                         
                                                   [[100, 101, 104],
                                                    [105,  82,  89],
                                                    [ 95,  91,  94]]],
                                         
                                         
                                                  [[[  0,   2,   4],
                                                    [  5,  12,   9],
                                                    [ 15,  21,  23]],
                                         
                                                   [[ 25,  51,  28],
                                                    [ 30,  56,  59],
                                                    [ 65,  71,  73]],
                                         
                                                   [[ 75,  76,  78],
                                                    [105, 111,  83],
                                                    [ 90,  91,  93]]]], dtype=Dtype.Int32).unsqueeze(0)
            Assert.CheckEqual(tk2p1iCorrect, tk2p1i)
            Assert.CheckEqual(tk2p1Correct, tk2p1)

            let tk2s3, tk2s3i = dsharp.maxpool3di(t, 2, stride=3)
            let tk2s3Correct = combo.tensor([[[[1.5542, 1.2910],
                                                [1.5936, 1.0687]],
                                     
                                               [[1.6584, 1.5641],
                                                [2.2015, 1.8761]]],
                                     
                                     
                                              [[[2.0711, 1.1195],
                                                [1.2176, 1.8941]],
                                     
                                               [[1.9343, 2.4069],
                                                [0.5255, 1.5519]]]]).unsqueeze(0)
            let tk2s3iCorrect = combo.tensor([[[[  6,  34],
                                                    [ 21,  43]],
                                         
                                                   [[101, 104],
                                                    [ 95,  94]]],
                                         
                                         
                                                  [[[  5,   9],
                                                    [ 15,  44]],
                                         
                                                   [[ 76,  83],
                                                    [ 90,  93]]]], dtype=Dtype.Int32).unsqueeze(0)
            Assert.CheckEqual(tk2s3iCorrect, tk2s3i)
            Assert.CheckEqual(tk2s3Correct, tk2s3)

            let tk2s3p1, tk2s3p1i = dsharp.maxpool3di(t, 2, stride=3, padding=1)
            let tk2s3p1Correct = combo.tensor([[[[ 0.4633,  0.4568],
                                                    [ 0.8156,  1.3066]],
                                         
                                                   [[ 0.2996,  0.2835],
                                                    [ 2.0900,  1.2851]]],
                                         
                                         
                                                  [[[-1.2563,  1.0405],
                                                    [ 1.2176,  1.1689]],
                                         
                                                   [[ 0.8200,  2.1774],
                                                    [ 0.5255,  1.7098]]]]).unsqueeze(0)
            let tk2s3p1iCorrect = combo.tensor([[[[ 0,  2],
                                                    [15, 17]],
                                         
                                                   [[50, 53],
                                                    [90, 67]]],
                                         
                                         
                                                  [[[ 0,  2],
                                                    [15, 18]],
                                         
                                                   [[50, 78],
                                                    [90, 88]]]], dtype=Dtype.Int32).unsqueeze(0)
            Assert.CheckEqual(tk2s3p1iCorrect, tk2s3p1i)
            Assert.CheckEqual(tk2s3p1Correct, tk2s3p1)

        for combo in Combos.IntegralAndBool do 
            let x = combo.zeros([4;4;4;4;4])
            isInvalidOp(fun () -> dsharp.maxpool3d(x,3))

    [<Test>]
    member _.TestTensorMaxUnpool1D () =
        for combo in Combos.FloatingPoint do
            let tk3 = combo.tensor([[[ 2.5995,  1.3858,  0.9593],
                                      [ 0.4564,  0.4587,  1.1539]],
                             
                                     [[ 0.9980,  0.1321,  1.0608],
                                      [ 1.1872, -0.2067,  0.6039]]])
            let tk3i = combo.tensor([[[2, 3, 6],
                                          [0, 3, 8]],
                                 
                                         [[2, 3, 6],
                                          [0, 3, 6]]], dtype=Dtype.Int32)
            let tk3u = dsharp.maxunpool1d(tk3, tk3i, 3)
            let tk3uCorrect = combo.tensor([[[ 0.0000,  0.0000,  2.5995,  1.3858,  0.0000,  0.0000,  0.9593,  0.0000,  0.0000],
                                             [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.0000,  0.0000,  1.1539]],

                                            [[ 0.0000,  0.0000,  0.9980,  0.1321,  0.0000,  0.0000,  1.0608,  0.0000,  0.0000],
                                             [ 1.1872,  0.0000,  0.0000, -0.2067,  0.0000,  0.0000,  0.6039,  0.0000,  0.0000]]])
            Assert.CheckEqual(tk3uCorrect, tk3u)

            let tk3p1 = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                            [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                   
                                           [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                            [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk3p1i = combo.tensor([[[1, 2, 6, 8],
                                                [0, 3, 7, 8]],
                                       
                                               [[1, 2, 6, 9],
                                                [0, 2, 6, 9]]], dtype=Dtype.Int32)
            let tk3p1u = dsharp.maxunpool1d(tk3p1, tk3p1i, 3, padding=1)
            let tk3p1uCorrect = combo.tensor([[[ 0.0000, -1.1558,  2.5995,  0.0000,  0.0000,  0.0000,  0.9593,
                                                   0.0000,  0.7169,  0.0000],
                                                 [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.0000,
                                                   0.6288,  1.1539,  0.0000]],

                                                [[ 0.0000,  0.7151,  0.9980,  0.0000,  0.0000,  0.0000,  1.0608,
                                                   0.0000,  0.0000,  1.6387],
                                                 [ 1.1872,  0.0000, -0.0297,  0.0000,  0.0000,  0.0000,  0.6039,
                                                   0.0000,  0.0000,  1.2069]]])
            Assert.CheckEqual(tk3p1uCorrect, tk3p1u)

            let tk3s2 = combo.tensor([[[ 2.5995,  2.5995,  0.9593,  0.9593],
                                              [ 0.4564,  0.4587,  0.2978,  1.1539]],
                                     
                                             [[ 0.9980,  0.9980,  1.0608,  1.0608],
                                              [ 1.1872, -0.0297,  0.6039,  0.6039]]])
            let tk3s2i = combo.tensor([[[2, 2, 6, 6],
                                                  [0, 3, 6, 8]],
                                         
                                                 [[2, 2, 6, 6],
                                                  [0, 2, 6, 6]]], dtype=Dtype.Int32)
            let tk3s2u = dsharp.maxunpool1d(tk3s2, tk3s2i, 3, stride=2)
            let tk3s2uCorrect = combo.tensor([[[ 0.0000,  0.0000,  2.5995,  0.0000,  0.0000,  0.0000,  0.9593,
                                                   0.0000,  0.0000],
                                                 [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.2978,
                                                   0.0000,  1.1539]],

                                                [[ 0.0000,  0.0000,  0.9980,  0.0000,  0.0000,  0.0000,  1.0608,
                                                   0.0000,  0.0000],
                                                 [ 1.1872,  0.0000, -0.0297,  0.0000,  0.0000,  0.0000,  0.6039,
                                                   0.0000,  0.0000]]])
            Assert.CheckEqual(tk3s2uCorrect, tk3s2u)

            let tk4s3p2 = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                              [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                     
                                             [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                              [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk4s3p2i = combo.tensor([[[1, 2, 6, 8],
                                                  [0, 3, 7, 8]],
                                         
                                                 [[1, 2, 6, 9],
                                                  [0, 2, 6, 9]]], dtype=Dtype.Int32)
            let tk4s3p2u = dsharp.maxunpool1d(tk4s3p2, tk4s3p2i, 4, stride=3, padding=2, outputSize=[2;2;10])
            let tk4s3p2uCorrect = combo.tensor([[[ 0.0000, -1.1558,  2.5995,  0.0000,  0.0000,  0.0000,  0.9593,
                                                   0.0000,  0.7169,  0.0000],
                                                 [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.0000,
                                                   0.6288,  1.1539,  0.0000]],

                                                [[ 0.0000,  0.7151,  0.9980,  0.0000,  0.0000,  0.0000,  1.0608,
                                                   0.0000,  0.0000,  1.6387],
                                                 [ 1.1872,  0.0000, -0.0297,  0.0000,  0.0000,  0.0000,  0.6039,
                                                   0.0000,  0.0000,  1.2069]]])
            Assert.CheckEqual(tk4s3p2uCorrect, tk4s3p2u)

    [<Test>]
    member _.TestTensorMaxUnpool2D () =
        for combo in Combos.FloatingPoint do
            let tk3 = combo.tensor([[[[1.8489, 1.1338],
                                              [0.6819, 1.6331]],

                                             [[1.0867, 2.1048],
                                              [2.7646, 1.0156]]],


                                            [[[2.1120, 0.8666],
                                              [0.9141, 1.7133]],

                                             [[1.4250, 1.8228],
                                              [1.2607, 0.5448]]]])
            let tk3i = combo.tensor([[[[10, 21],
                                                  [41, 45]],

                                                 [[ 8,  5],
                                                  [40, 28]]],


                                                [[[ 8, 21],
                                                  [32, 36]],

                                                 [[ 9, 13],
                                                  [25, 27]]]], dtype=Dtype.Int32)
            let tk3u = dsharp.maxunpool2d(tk3, tk3i, 3, outputSize=[2;2;8;8])
            let tk3uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                             [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.1048, 0.0000, 0.0000],
                                              [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 1.0156, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                            [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8666, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                             [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 1.2607, 0.0000, 0.5448, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
            Assert.CheckEqual(tk3uCorrect, tk3u)

            let tk3p1 = combo.tensor([[[[0.7372, 1.8489, 0.3801],
                                              [0.6883, 1.0254, 1.1338],
                                              [0.6819, 1.0146, 1.6331]],

                                             [[1.0867, 1.7527, 2.1048],
                                              [2.2070, 1.0156, 1.2660],
                                              [2.7646, 1.3832, 2.5090]]],


                                            [[[2.1120, 0.2085, 1.0594],
                                              [0.9141, 1.7133, 1.2406],
                                              [0.3855, 1.4428, 0.6534]],

                                             [[1.4250, 1.7200, 1.8228],
                                              [1.2607, 0.6599, 1.8958],
                                              [0.6304, 1.5875, 0.4057]]]])
            let tk3p1i = combo.tensor([[[[ 0, 10,  6],
                                                  [16, 36, 21],
                                                  [41, 59, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 23],
                                                  [40, 52, 47]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 37],
                                                  [49, 60, 45]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [56, 60, 47]]]], dtype=Dtype.Int32)
            let tk3p1u = dsharp.maxunpool2d(tk3p1, tk3p1i, 3, padding=1, outputSize=[2;2;8;8])
            let tk3p1uCorrect = combo.tensor([[[[0.7372, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3801, 0.0000],
                                                  [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6883, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0254, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 1.0146, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 1.7527, 0.0000, 2.1048, 0.0000, 0.0000],
                                                  [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.2660],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0156, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 2.2070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.5090],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.3832, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                [[[0.0000, 0.0000, 0.2085, 0.0000, 0.0000, 0.0000, 1.0594, 0.0000],
                                                  [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 1.2406, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6534, 0.0000, 0.0000],
                                                  [0.0000, 0.3855, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.4428, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 0.0000, 1.7200, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.6599, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.2607, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.8958, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4057],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6304, 0.0000, 0.0000, 0.0000, 1.5875, 0.0000, 0.0000, 0.0000]]]])
            Assert.CheckEqual(tk3p1uCorrect, tk3p1u)

            let tk3s2 = combo.tensor([[[[1.8489, 1.8489, 1.1338],
                                              [0.6883, 1.0254, 1.1338],
                                              [0.6819, 1.0254, 1.6331]],

                                             [[1.0867, 1.7527, 2.1048],
                                              [2.2070, 1.0156, 1.0156],
                                              [2.7646, 1.3832, 1.3832]]],


                                            [[[2.1120, 0.2085, 1.0594],
                                              [0.9141, 1.7133, 1.7133],
                                              [0.9141, 1.7133, 1.7133]],

                                             [[1.4250, 1.7200, 1.8228],
                                              [1.2607, 0.6599, 1.8958],
                                              [1.2130, 1.2130, 1.8958]]]])
            let tk3s2i = combo.tensor([[[[10, 10, 21],
                                                  [16, 36, 21],
                                                  [41, 36, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 28],
                                                  [40, 52, 52]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 36],
                                                  [32, 36, 36]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [50, 50, 38]]]], dtype=Dtype.Int32)
            let tk3s2u = dsharp.maxunpool2d(tk3s2, tk3s2i, 3, stride=2, outputSize=[2;2;8;8])
            let tk3s2uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6883, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0254, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 1.7527, 0.0000, 2.1048, 0.0000, 0.0000],
                                                  [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0156, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 2.2070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.3832, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                [[[0.0000, 0.0000, 0.2085, 0.0000, 0.0000, 0.0000, 1.0594, 0.0000],
                                                  [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 0.0000, 1.7200, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.6599, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.2607, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.8958, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 1.2130, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
            Assert.CheckEqual(tk3s2uCorrect, tk3s2u)

            let tk4s3p2 = combo.tensor([[[[0.7372, 1.8489, 1.0141],
                                              [0.6883, 1.8489, 1.1338],
                                              [0.6819, 1.0254, 1.6331]],

                                             [[1.0867, 1.7527, 2.1048],
                                              [2.2070, 2.2070, 1.4476],
                                              [2.7646, 2.2070, 2.5090]]],


                                            [[[2.1120, 0.4063, 1.0594],
                                              [2.1120, 1.7133, 1.7133],
                                              [0.9141, 1.7133, 1.7133]],

                                             [[1.4250, 1.7200, 1.8228],
                                              [1.4250, 1.4250, 1.8958],
                                              [0.6304, 1.5875, 1.8958]]]])
            let tk4s3p2i = combo.tensor([[[[ 0, 10,  4],
                                                      [16, 10, 21],
                                                      [41, 36, 45]],

                                                     [[ 8,  3,  5],
                                                      [33, 33, 15],
                                                      [40, 33, 47]]],


                                                    [[[ 8,  1,  6],
                                                      [ 8, 36, 36],
                                                      [32, 36, 36]],

                                                     [[ 9,  4, 13],
                                                      [ 9,  9, 38],
                                                      [56, 60, 38]]]], dtype=Dtype.Int32)
            let tk4s3p2u = dsharp.maxunpool2d(tk4s3p2, tk4s3p2i, 4, stride=3, padding=2, outputSize=[2;2;8;8])
            let tk4s3p2uCorrect = combo.tensor([[[[0.7372, 0.0000, 0.0000, 0.0000, 1.0141, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6883, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0254, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 1.7527, 0.0000, 2.1048, 0.0000, 0.0000],
                                                  [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.4476],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 2.2070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.5090],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                [[[0.0000, 0.4063, 0.0000, 0.0000, 0.0000, 0.0000, 1.0594, 0.0000],
                                                  [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 0.0000, 1.7200, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.8958, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6304, 0.0000, 0.0000, 0.0000, 1.5875, 0.0000, 0.0000, 0.0000]]]])
            Assert.CheckEqual(tk4s3p2uCorrect, tk4s3p2u)


    [<Test>]
    member _.TestTensorMaxUnpool3D () =
        for combo in Combos.FloatingPoint do
            let tk2 = combo.tensor([[[[1.5542, 0.5720],
                                        [1.5415, 1.3066]],
                             
                                       [[1.1442, 1.3531],
                                        [2.0900, 1.2851]]],
                             
                             
                                      [[[2.0711, 1.2451],
                                        [1.2176, 1.1689]],
                             
                                       [[2.2379, 2.4069],
                                        [0.5255, 1.7098]]]]).unsqueeze(0)
            let tk2i = combo.tensor([[[[ 6,  7],
                                        [16, 17]],
                             
                                       [[76, 82],
                                        [90, 67]]],
                             
                             
                                      [[[ 5, 32],
                                        [15, 18]],
                             
                                       [[56, 83],
                                        [90, 88]]]], dtype=Dtype.Int32).unsqueeze(0)
            let tk2u = dsharp.maxunpool3d(tk2, tk2i, 2, outputSize=[1;2;5;5;5])
            let tk2uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 1.5542, 0.5720, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 1.5415, 1.3066, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 1.2851, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 1.1442, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 1.3531, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [2.0900, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                             [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [2.0711, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [1.2176, 0.0000, 0.0000, 1.1689, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 1.2451, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 2.2379, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 2.4069, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 1.7098, 0.0000],
                                               [0.5255, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]]).unsqueeze(0)
            Assert.CheckEqual(tk2uCorrect, tk2u)

            let tk2p1 = combo.tensor([[[[ 0.4633,  0.9173, -0.1077],
                                                [ 0.3505,  1.5542,  1.2303],
                                                [ 0.8156,  1.5936,  0.2060]],
                                     
                                               [[ 0.2996,  0.5738,  0.6299],
                                                [ 1.0910, -0.0037,  1.2910],
                                                [ 0.1940,  1.2939,  1.3469]],
                                     
                                               [[ 1.0358,  1.6584,  1.5641],
                                                [ 0.4032,  1.3531,  1.1186],
                                                [ 2.2015,  1.0479,  1.8761]]],
                                     
                                     
                                              [[[-1.2563,  1.0405,  0.7333],
                                                [ 2.0711,  1.0576,  1.1195],
                                                [ 1.2176,  0.3765,  1.8267]],
                                     
                                               [[ 1.2927,  1.8860,  0.8106],
                                                [ 0.7614,  2.2379,  1.4035],
                                                [ 0.4963,  2.5382,  1.9080]],
                                     
                                               [[ 0.5853,  1.9343,  2.1774],
                                                [ 1.2673,  1.3343,  2.4069],
                                                [ 0.5255,  0.2564,  1.5519]]]]).unsqueeze(0)
            let tk2p1i = combo.tensor([[[[  0,   1,   4],
                                                    [ 10,   6,  14],
                                                    [ 15,  21,  19]],
                                         
                                                   [[ 50,  51,  29],
                                                    [ 60,  62,  34],
                                                    [ 40,  72,  73]],
                                         
                                                   [[100, 101, 104],
                                                    [105,  82,  89],
                                                    [ 95,  91,  94]]],
                                         
                                         
                                                  [[[  0,   2,   4],
                                                    [  5,  12,   9],
                                                    [ 15,  21,  23]],
                                         
                                                   [[ 25,  51,  28],
                                                    [ 30,  56,  59],
                                                    [ 65,  71,  73]],
                                         
                                                   [[ 75,  76,  78],
                                                    [105, 111,  83],
                                                    [ 90,  91,  93]]]], dtype=Dtype.Int32).unsqueeze(0)
            let tk2p1u = dsharp.maxunpool3d(tk2p1, tk2p1i, 2, padding=1, outputSize=[1;2;5;5;5])
            let tk2p1uCorrect = combo.tensor([[[[ 0.4633,  0.9173,  0.0000,  0.0000, -0.1077],
                                                   [ 0.0000,  1.5542,  0.0000,  0.0000,  0.0000],
                                                   [ 0.3505,  0.0000,  0.0000,  0.0000,  1.2303],
                                                   [ 0.8156,  0.0000,  0.0000,  0.0000,  0.2060],
                                                   [ 0.0000,  1.5936,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.6299],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  1.2910],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.1940,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.2996,  0.5738,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 1.0910,  0.0000, -0.0037,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  1.2939,  1.3469,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  1.3531,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  1.1186],
                                                   [ 0.0000,  1.0479,  0.0000,  0.0000,  1.8761],
                                                   [ 2.2015,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 1.0358,  1.6584,  0.0000,  0.0000,  1.5641],
                                                   [ 0.4032,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],


                                                 [[[-1.2563,  0.0000,  1.0405,  0.0000,  0.7333],
                                                   [ 2.0711,  0.0000,  0.0000,  0.0000,  1.1195],
                                                   [ 0.0000,  0.0000,  1.0576,  0.0000,  0.0000],
                                                   [ 1.2176,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.3765,  0.0000,  1.8267,  0.0000]],

                                                  [[ 1.2927,  0.0000,  0.0000,  0.8106,  0.0000],
                                                   [ 0.7614,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  1.8860,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  2.2379,  0.0000,  0.0000,  1.4035],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.4963,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  2.5382,  0.0000,  1.9080,  0.0000]],

                                                  [[ 0.5853,  1.9343,  0.0000,  2.1774,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  2.4069,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.5255,  0.2564,  0.0000,  1.5519,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 1.2673,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  1.3343,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]]).unsqueeze(0)
            Assert.CheckEqual(tk2p1uCorrect, tk2p1u)

            let tk2s3 = combo.tensor([[[[1.5542, 1.2910],
                                            [1.5936, 1.0687]],
                                 
                                           [[1.6584, 1.5641],
                                            [2.2015, 1.8761]]],
                                 
                                 
                                          [[[2.0711, 1.1195],
                                            [1.2176, 1.8941]],
                                 
                                           [[1.9343, 2.4069],
                                            [0.5255, 1.5519]]]]).unsqueeze(0)
            let tk2s3i = combo.tensor([[[[  6,  34],
                                                    [ 21,  43]],
                                         
                                                   [[101, 104],
                                                    [ 95,  94]]],
                                         
                                         
                                                  [[[  5,   9],
                                                    [ 15,  44]],
                                         
                                                   [[ 76,  83],
                                                    [ 90,  93]]]], dtype=Dtype.Int32).unsqueeze(0)
            let tk2s3u = dsharp.maxunpool3d(tk2s3, tk2s3i, 2, stride=3, outputSize=[1;2;5;5;5])
            let tk2s3uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 1.5542, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 1.5936, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 1.2910],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 1.0687, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 1.8761],
                                                   [2.2015, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 1.6584, 0.0000, 0.0000, 1.5641],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                 [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [2.0711, 0.0000, 0.0000, 0.0000, 1.1195],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [1.2176, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 1.8941],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 1.9343, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 2.4069, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.5255, 0.0000, 0.0000, 1.5519, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]]).unsqueeze(0)
            Assert.CheckEqual(tk2s3uCorrect, tk2s3u)

            let tk2s3p1 = combo.tensor([[[[ 0.4633,  0.4568],
                                                [ 0.8156,  1.3066]],
                                     
                                               [[ 0.2996,  0.2835],
                                                [ 2.0900,  1.2851]]],
                                     
                                     
                                              [[[-1.2563,  1.0405],
                                                [ 1.2176,  1.1689]],
                                     
                                               [[ 0.8200,  2.1774],
                                                [ 0.5255,  1.7098]]]]).unsqueeze(0)
            let tk2s3p1i = combo.tensor([[[[ 0,  2],
                                                    [15, 17]],
                                         
                                                   [[50, 53],
                                                    [90, 67]]],
                                         
                                         
                                                  [[[ 0,  2],
                                                    [15, 18]],
                                         
                                                   [[50, 78],
                                                    [90, 88]]]], dtype=Dtype.Int32).unsqueeze(0)
            let tk2s3p1u = dsharp.maxunpool3d(tk2s3p1, tk2s3p1i, 2, stride=3, padding=1, outputSize=[1;2;5;5;5])
            let tk2s3p1uCorrect = combo.tensor([[[[ 0.4633,  0.0000,  0.4568,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.8156,  0.0000,  1.3066,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.2996,  0.0000,  0.0000,  0.2835,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  1.2851,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 2.0900,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],


                                                 [[[-1.2563,  0.0000,  1.0405,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 1.2176,  0.0000,  0.0000,  1.1689,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.8200,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  2.1774,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  1.7098,  0.0000],
                                                   [ 0.5255,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]]).unsqueeze(0)
            Assert.CheckEqual(tk2s3p1uCorrect, tk2s3p1u)

    [<Test>]
    member _.TestTensorConv1D () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[[0.3460; 0.4414; 0.2384; 0.7905; 0.2267];
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
            let t2 = combo.tensor([[[0.4941; 0.8710; 0.0606];
                                    [0.2831; 0.7930; 0.5602];
                                    [0.0024; 0.1236; 0.4394];
                                    [0.9086; 0.1277; 0.2450]];

                                   [[0.5196; 0.1349; 0.0282];
                                    [0.1749; 0.6234; 0.5502];
                                    [0.7678; 0.0733; 0.3396];
                                    [0.6023; 0.6546; 0.3439]]])

            let t3 = dsharp.conv1d(t1, t2)
            let t3Correct = combo.tensor([[[2.8516; 2.0732; 2.6420];
                                           [2.3239; 1.7078; 2.7450]];

                                          [[3.0127; 2.9651; 2.5219];
                                           [3.0899; 3.1496; 2.4110]];

                                          [[3.4749; 2.9038; 2.7131];
                                           [2.7692; 2.9444; 3.2554]]])

            Assert.True(t3.allclose(t3Correct, 0.01))

            let t3p1 = dsharp.conv1d(t1, t2, padding=1)
            let t3p1Correct = combo.tensor([[[1.4392; 2.8516; 2.0732; 2.6420; 2.1177];
                                             [1.4345; 2.3239; 1.7078; 2.7450; 2.1474]];

                                            [[2.4208; 3.0127; 2.9651; 2.5219; 1.2960];
                                             [1.5544; 3.0899; 3.1496; 2.4110; 1.8567]];

                                            [[1.2965; 3.4749; 2.9038; 2.7131; 1.7408];
                                             [1.3549; 2.7692; 2.9444; 3.2554; 1.2120]]])

            Assert.True(t3p1.allclose(t3p1Correct, 0.01))

            let t3p2 = dsharp.conv1d(t1, t2, padding=2)
            let t3p2Correct = combo.tensor([[[0.6333; 1.4392; 2.8516; 2.0732; 2.6420; 2.1177; 1.0258];
                                             [0.6539; 1.4345; 2.3239; 1.7078; 2.7450; 2.1474; 1.2585]];

                                            [[0.5982; 2.4208; 3.0127; 2.9651; 2.5219; 1.2960; 1.0620];
                                             [0.5157; 1.5544; 3.0899; 3.1496; 2.4110; 1.8567; 1.3182]];

                                            [[0.3165; 1.2965; 3.4749; 2.9038; 2.7131; 1.7408; 0.5275];
                                             [0.3861; 1.3549; 2.7692; 2.9444; 3.2554; 1.2120; 0.7428]]])

            Assert.True(t3p2.allclose(t3p2Correct, 0.01))

            let t3s2 = dsharp.conv1d(t1, t2, stride=2)
            let t3s2Correct = combo.tensor([[[2.8516; 2.6420];
                                             [2.3239; 2.7450]];

                                            [[3.0127; 2.5219];
                                             [3.0899; 2.4110]];

                                            [[3.4749; 2.7131];
                                             [2.7692; 3.2554]]])

            Assert.True(t3s2.allclose(t3s2Correct, 0.01))

            let t3s3 = dsharp.conv1d(t1, t2, stride=3)
            let t3s3Correct = combo.tensor([[[2.8516];
                                             [2.3239]];

                                            [[3.0127];
                                             [3.0899]];

                                            [[3.4749];
                                             [2.7692]]])

            Assert.True(t3s3.allclose(t3s3Correct, 0.01))

            let t3s2p1 = dsharp.conv1d(t1, t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[1.4392; 2.0732; 2.1177];
                                                 [1.4345; 1.7078; 2.1474]];

                                                [[2.4208; 2.9651; 1.2960];
                                                 [1.5544; 3.1496; 1.8567]];

                                                [[1.2965; 2.9038; 1.7408];
                                                 [1.3549; 2.9444; 1.2120]]])

            Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01))

            let t3s3p2 = dsharp.conv1d(t1, t2, stride=3, padding=2)
            let t3s3p2Correct = combo.tensor([[[0.6333; 2.0732; 1.0258];
                                                 [0.6539; 1.7078; 1.2585]];

                                                [[0.5982; 2.9651; 1.0620];
                                                 [0.5157; 3.1496; 1.3182]];

                                                [[0.3165; 2.9038; 0.5275];
                                                 [0.3861; 2.9444; 0.7428]]])
        
            Assert.True(t3s3p2.allclose(t3s3p2Correct, 0.01))

            let t3d2 = dsharp.conv1d(t1, t2, dilation=2)
            let t3d2Correct = combo.tensor([[[2.8030];
                                             [2.4735]];

                                            [[2.9226];
                                             [3.1868]];

                                            [[2.8469];
                                             [2.4790]]])

            Assert.True(t3d2.allclose(t3d2Correct, 0.01))

            let t3p2d3 = dsharp.conv1d(t1, t2, padding=2, dilation=3)
            let t3p2d3Correct = combo.tensor([[[2.1121; 0.8484; 2.2709];
                                                 [1.6692; 0.5406; 1.8381]];

                                                [[2.5078; 1.2137; 0.9173];
                                                 [2.2395; 1.1805; 1.1954]];

                                                [[1.5215; 1.3946; 2.1327];
                                                 [1.0732; 1.3014; 2.0696]]])

            Assert.True(t3p2d3.allclose(t3p2d3Correct, 0.01))

            let t3s3p6d3 = dsharp.conv1d(t1, t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = combo.tensor([[[0.6333; 1.5018; 2.2709; 1.0580];
                                                 [0.6539; 1.5130; 1.8381; 1.0479]];

                                                [[0.5982; 1.7459; 0.9173; 0.2709];
                                                 [0.5157; 0.8537; 1.1954; 0.7027]];

                                                [[0.3165; 1.4118; 2.1327; 1.1949];
                                                 [0.3861; 1.5697; 2.0696; 0.8520]]])

            Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01))

            let t3b1 = t1.[0].unsqueeze(0).conv1d(t2)
            let t3b1Correct = t3Correct.[0].unsqueeze(0)
            Assert.True(t3b1.allclose(t3b1Correct, 0.01))

            let t3b1s2 = t1.[0].unsqueeze(0).conv1d(t2, stride = 2)
            let t3b1s2Correct = t3s2Correct.[0].unsqueeze(0)

            Assert.True(t3b1s2.allclose(t3b1s2Correct, 0.01))

        for combo in Combos.Integral do 
            let x = combo.ones([1;4;4])
            let y = combo.ones([1;4;4])
            let z = dsharp.conv1d(x,y)
            let zCorrect = combo.tensor([[[16]]])
            Assert.CheckEqual(z, zCorrect)
               

        // check types must always match
        for dtype1 in Dtypes.All do 
            for dtype2 in Dtypes.All do 
                if dtype1 <> dtype2 then 
                    let x = dsharp.zeros([1;4;4], dtype=dtype1)
                    let y = dsharp.zeros([1;4;4], dtype=dtype2)
                    isException(fun () -> dsharp.conv1d(x,y))

        for combo in Combos.Bool do 
            let x = combo.zeros([1;4;4])
            let y = combo.zeros([1;4;4])
            isInvalidOp(fun () -> dsharp.conv1d(x,y))

    [<Test>]
    member _.TestTensorConv2D () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[[[ 10.7072,  -5.0993,   3.6884,   2.0982],
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
            let t2 = combo.tensor([[[[-5.6745, -1.9422,  4.1369],
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

            let t3 = dsharp.conv2d(t1, t2)
            let t3Correct = combo.tensor([[[[  10.6089;   -1.4459];
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

            let t3p1 = dsharp.conv2d(t1, t2, padding=1)
            let t3p1Correct = combo.tensor([[[[  86.6988;    8.1164;  -85.8172;   69.5001];
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

            let t3p12 = dsharp.conv2d(t1, t2, paddings=[|1; 2|])
            let t3p12Correct = combo.tensor([[[[   7.5867;   86.6988;    8.1164;  -85.8172;   69.5001;  -35.4485];
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

            let t3s2 = dsharp.conv2d(t1, t2, stride=2)
            let t3s2Correct = combo.tensor([[[[  10.6089]];

                                             [[  97.8425]];

                                             [[ 427.2891]]];


                                            [[[-127.6157]];

                                             [[ 104.2333]];

                                             [[-106.0468]]]])

            let t3s13 = dsharp.conv2d(t1, t2, strides=[|1; 3|])
            let t3s13Correct = combo.tensor([[[[  10.6089];
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

            let t3s2p1 = dsharp.conv2d(t1, t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[[  86.6988;  -85.8172];
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

            let t3s23p32 = dsharp.conv2d(t1, t2, strides=[2; 3], paddings=[3; 2])
            let t3s23p32Correct = combo.tensor([[[[   0.0000,    0.0000],
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
        
            let t3p1d2 = dsharp.conv2d(t1, t2, padding=1, dilation=2)
            let t3p1d2Correct = combo.tensor([[[[ -72.7697,  -34.7305],
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

            let t3p22d23 = dsharp.conv2d(t1, t2, paddings=[2;2], dilations=[2;3])
            let t3p22d23Correct = combo.tensor([[[[-3.2693e+01, -4.3192e+01],
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

            let t3s3p6d3 = dsharp.conv2d(t1, t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = combo.tensor([[[[  78.0793,   88.7191,  -32.2774,   12.5512],
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

        // check intergral types
        for combo in Combos.Integral do 
            let x = combo.ones([1;1;4;4])
            let y = combo.ones([1;1;4;4])
            let z = dsharp.conv2d(x, y)
            let zCorrect = combo.tensor([[[[16]]]])
            Assert.CheckEqual(z, zCorrect)

        // check types must always match
        for dtype1 in Dtypes.All do 
            for dtype2 in Dtypes.All do 
                if dtype1 <> dtype2 then 
                    let x = dsharp.zeros([1;1;4;4], dtype=dtype1)
                    let y = dsharp.zeros([1;1;4;4], dtype=dtype2)
                    isException(fun () -> dsharp.conv2d(x,y, strides=[1;1]))

        for combo in Combos.Bool do 
            let x = combo.zeros([1;1;4;4])
            let y = combo.zeros([1;1;4;4])
            isInvalidOp(fun () -> dsharp.conv2d(x,y, strides=[1;1]))

    [<Test>]
    member _.TestTensorConv3D () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[[[ 2.0403e+00,  5.0188e-01,  4.6880e-01,  8.0736e-01],
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
            let t2 = combo.tensor([[[[-0.5868, -0.6268,  0.2067],
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

            let t3 = dsharp.conv3d(t1, t2)
            let t3Correct = combo.tensor([[[[ 3.1109,  6.7899],
                                               [ 4.3064,  4.1053]],

                                              [[ 5.0324, -8.8943],
                                               [-0.1298,  1.2862]]]]).unsqueeze(0)

            let t3p1 = dsharp.conv3d(t1, t2, padding=1)
            let t3p1Correct = combo.tensor([[[[  2.9555,  -2.2637,  -7.1829,   5.6339],
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

            let t3p123 = dsharp.conv3d(t1, t2, paddings=[|1; 2; 3|])
            let t3p123Correct = combo.tensor([[[[ 0.0000e+00, -2.9020e+00,  4.5825e+00, -3.1431e+00, -1.0803e+00,
                                                         8.2371e-01,  1.4897e-01,  0.0000e+00],
                                                   [ 0.0000e+00, -1.2234e+00,  2.9555e+00, -2.2637e+00, -7.1829e+00,
                                                         5.6339e+00,  5.1473e-01,  0.0000e+00],
                                                   [ 0.0000e+00, -6.8862e-01, -3.3115e+00,  1.1712e+01,  2.7917e+00,
                                                         2.6118e+00, -3.8470e-01,  0.0000e+00],
                                                   [ 0.0000e+00,  3.3201e+00,  5.5319e+00,  3.0030e+00,  3.2099e+00,
                                                        -2.7804e+00,  6.1979e-01,  0.0000e+00],
                                                   [ 0.0000e+00,  8.8853e-01, -1.4804e+00, -1.1566e-01, -6.4439e+00,
                                                        -7.1598e-02,  2.3270e-01,  0.0000e+00],
                                                   [ 0.0000e+00, -3.5118e+00,  2.0512e+00,  1.6275e+00,  1.7109e+00,
                                                         1.5145e-01, -1.7395e-01,  0.0000e+00]],

                                                  [[ 0.0000e+00,  7.1204e+00,  3.0177e-04, -6.9272e+00,  2.8760e+00,
                                                        -1.9002e-02, -2.4133e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  5.6420e+00,  2.4783e+00, -2.6479e+00,  5.6216e+00,
                                                        -1.2882e+00, -5.9195e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  7.1537e-02, -1.0339e+01,  3.1109e+00,  6.7899e+00,
                                                        -6.1003e+00,  1.2121e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  8.9927e-01, -1.3145e+00,  4.3064e+00,  4.1053e+00,
                                                         5.3012e+00, -4.4293e+00,  0.0000e+00],
                                                   [ 0.0000e+00, -5.7960e-01,  2.6878e+00, -4.5237e+00, -6.7276e-01,
                                                         6.7965e-01, -6.6988e-01,  0.0000e+00],
                                                   [ 0.0000e+00,  8.0942e-01,  6.4290e-01,  1.2871e+00,  5.3531e-01,
                                                        -1.0901e+00, -1.6275e+00,  0.0000e+00]],

                                                  [[ 0.0000e+00, -6.6101e-01, -4.8746e+00,  7.4949e+00,  3.0253e+00,
                                                        -1.3816e+00, -4.6669e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  4.2946e+00, -1.4721e+00, -4.1515e+00,  4.6180e+00,
                                                        -9.2384e+00,  3.2005e+00,  0.0000e+00],
                                                   [ 0.0000e+00, -2.9133e+00,  9.8664e+00,  5.0324e+00, -8.8943e+00,
                                                         5.2075e+00,  2.1560e+00,  0.0000e+00],
                                                   [ 0.0000e+00, -9.4993e+00, -1.5404e+00, -1.2982e-01,  1.2862e+00,
                                                        -3.2419e+00,  4.1770e-01,  0.0000e+00],
                                                   [ 0.0000e+00, -4.7673e+00,  8.5308e+00,  2.7561e+00, -6.2106e+00,
                                                         1.8973e+00,  2.6808e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  3.9791e+00,  5.8774e-01,  3.1007e-01, -4.0616e+00,
                                                        -8.0652e-01,  7.2560e-01,  0.0000e+00]],

                                                  [[ 0.0000e+00, -1.6718e+00,  2.1936e+00,  5.2331e-01, -2.4292e+00,
                                                        -2.0133e+00,  5.9281e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  3.6098e+00,  9.9384e-01, -2.9158e+00, -5.2227e+00,
                                                        -3.0340e+00,  1.4565e+00,  0.0000e+00],
                                                   [ 0.0000e+00,  2.3582e+00,  3.2490e+00,  2.0787e+00,  2.2262e+00,
                                                        -2.4861e+00,  3.0599e+00,  0.0000e+00],
                                                   [ 0.0000e+00, -6.6049e+00, -8.4240e-02,  3.4158e-01, -3.8301e+00,
                                                        -2.1084e+00,  2.8022e+00,  0.0000e+00],
                                                   [ 0.0000e+00, -1.1513e+00,  4.0825e+00, -1.9845e+00, -1.1269e+00,
                                                         2.3267e+00, -1.7839e-01,  0.0000e+00],
                                                   [ 0.0000e+00,  1.3527e+00, -3.7297e+00,  1.3533e+00,  1.6894e+00,
                                                        -3.2651e-01,  2.1566e-01,  0.0000e+00]]]]).unsqueeze(0)

            let t3s2 = dsharp.conv3d(t1, t2, stride=2)
            let t3s2Correct = combo.tensor([[[[3.1109]]]]).unsqueeze(0)

            let t3s132 = dsharp.conv3d(t1, t2, strides=[|1; 3; 2|])
            let t3s132Correct = combo.tensor([[[[3.1109]],
                                                  [[5.0324]]]]).unsqueeze(0)

            let t3s2p1 = dsharp.conv3d(t1, t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[[ 2.9555, -7.1829],
                                                   [ 5.5319,  3.2099]],

                                                  [[-1.4721,  4.6180],
                                                   [-1.5404,  1.2862]]]]).unsqueeze(0)

            let t3s231p321 = dsharp.conv3d(t1, t2, strides=[2; 3; 1], paddings=[3; 2; 1])
            let t3s231p321Correct = combo.tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
                                                       [ 0.0000,  0.0000,  0.0000,  0.0000]],

                                                      [[ 4.5825, -3.1431, -1.0803,  0.8237],
                                                       [ 5.5319,  3.0030,  3.2099, -2.7804]],

                                                      [[-4.8746,  7.4949,  3.0253, -1.3816],
                                                       [-1.5404, -0.1298,  1.2862, -3.2419]],

                                                      [[-0.1487, -1.5309,  1.1215,  3.0797],
                                                       [ 1.4189,  1.4221,  4.1597,  1.4329]]]]).unsqueeze(0)
            
            Assert.True(t3.allclose(t3Correct, 0.01, 0.01))
            Assert.True(t3p1.allclose(t3p1Correct, 0.01, 0.01))
            Assert.True(t3p123.allclose(t3p123Correct, 0.01, 0.01))
            Assert.True(t3s2.allclose(t3s2Correct, 0.01, 0.01))
            Assert.True(t3s132.allclose(t3s132Correct, 0.01, 0.01))
            Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01, 0.01))
            Assert.True(t3s231p321.allclose(t3s231p321Correct, 0.01, 0.01))

            let t3p1d2 = dsharp.conv3d(t1, t2, padding=1, dilation=2)
            let t3p1d2Correct = combo.tensor([[[[-0.2568,  0.7812],
                                                   [ 3.7157,  2.1968]],

                                                  [[ 7.7515,  1.1481],
                                                   [-1.2951, -2.1536]]]]).unsqueeze(0)
            Assert.True(t3p1d2.allclose(t3p1d2Correct, 0.01, 0.01))

            let t3p224d234 = dsharp.conv3d(t1, t2, paddings=[2;2;4], dilations=[2;3;4])
            let t3p224d234Correct = 
                                   combo.tensor([[[[ 0.5110,  0.8308,  0.8378,  2.1878],
                                                   [ 0.5542,  0.8628,  0.0433,  0.7889]],

                                                  [[ 0.7539,  0.8638,  2.9105, -0.6111],
                                                   [-2.2889,  2.2566, -0.4374, -1.2079]],

                                                  [[ 0.6620,  0.9611,  0.8799, -0.6184],
                                                   [-1.5508, -0.7252, -0.3192,  0.4482]],

                                                  [[-0.0271,  0.7710,  0.0897, -0.1711],
                                                   [-0.8259, -1.5293,  0.9234, -0.6048]]]]).unsqueeze(0)
            Assert.True(t3p224d234.allclose(t3p224d234Correct, 0.01, 0.01))

            let t3s3p6d3 = dsharp.conv3d(t1, t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = 
                                   combo.tensor([[[[-1.2082,  1.2172,  0.9059, -0.4916],
                                                   [ 2.1467, -3.7502,  5.0506,  0.3885],
                                                   [ 4.7375,  2.0637,  0.0984,  1.4406],
                                                   [-1.3617,  0.8104, -0.4940,  0.5110]],

                                                  [[-3.4229, -2.0909,  2.7974, -1.0638],
                                                   [-2.9979, -0.1444, -3.2004, -0.2850],
                                                   [ 1.0353, -1.1102,  0.8409, -0.3885],
                                                   [-1.3945,  2.0495,  1.7803, -0.3152]],

                                                  [[ 1.5129,  2.9412, -8.0788, -2.2397],
                                                   [ 0.6883, -1.7963,  0.6140, -2.7854],
                                                   [-1.1362,  1.5341, -3.5884, -1.6604],
                                                   [ 3.4384,  1.9425, -1.4670, -0.8295]],

                                                  [[-0.0370,  0.1560, -0.6491, -0.6168],
                                                   [ 2.4056,  0.5702, -3.0690, -0.5726],
                                                   [ 1.9479,  0.2854, -1.4980, -0.0100],
                                                   [-0.1114, -1.0524, -0.8736, -0.2113]]]]).unsqueeze(0)
            Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01, 0.01))

    [<Test>]
    member _.TestTensorNegT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.])
            let t1Neg = -t1
            let t1NegCorrect = combo.tensor([-1.; -2.; -3.])

            Assert.CheckEqual(t1NegCorrect, t1Neg)
            Assert.CheckEqual(t1Neg.dtype, combo.dtype)

        // Neg of Bool tensor not allowed
        //
        //    -torch.ones(10, dtype=torch.bool) 
        //
        // RuntimeError: Negation, the `-` operator, on a bool tensor is not supported. 

        for combo in Combos.Bool do 
            isInvalidOp(fun () -> -combo.tensor([1.0]))

    [<Test>]
    member _.TestTensorSumT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.])
            let t1Sum = t1.sum()
            let t1SumCorrect = combo.tensor(6., dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t1Sum.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(t1SumCorrect, t1Sum)

            // Now test cases where result type is set explicitly
            for dtype2 in Dtypes.IntegralAndFloatingPoint do
                let t1SumTyped = t1.sum(dtype=dtype2)
                let t1SumTypedCorrect = combo.tensor(6., dtype=dtype2)
                Assert.CheckEqual(t1SumTyped.dtype, dtype2)
                Assert.CheckEqual(t1SumTypedCorrect, t1SumTyped)

            let t2 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t2Sum = t2.sum()
            let t2SumCorrect = combo.tensor(10., dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t2Sum.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(t2SumCorrect, t2Sum)

        for combo in Combos.Bool do 
            // Sum of Bool tensor is Int64 tensor in pytorch
            let t3a = combo.tensor([true; true; false])
            let t3 = t3a.sum()
            let t3Correct = combo.tensor(2, dtype=Dtype.Int64)
            Assert.CheckEqual(t3, t3Correct)

    [<Test>]
    member _.TestTensorSumToSizeT () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.])
            let t1Sum = t1.sumToSize([| |])
            let t1SumCorrect = combo.tensor(6., dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t1SumCorrect, t1Sum)

            let t2 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t2Sum = t2.sumToSize([| |])
            let t2SumCorrect = combo.tensor(10., dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t2SumCorrect, t2Sum)

            let t3 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t3Sum = t3.sumToSize([| 2 |])
            let t3SumCorrect = combo.tensor( [4.; 6.], dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t3SumCorrect, t3Sum)

            let t4 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t4Sum = t4.sumToSize([| 1; 2 |])
            let t4SumCorrect = combo.tensor( [ [4.; 6.] ], dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t4SumCorrect, t4Sum)

            let t5 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t5Sum = t5.sumToSize([| 2; 1 |])
            let t5SumCorrect = combo.tensor( [ [3.]; [7.] ], dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t5SumCorrect, t5Sum)

    [<Test>]
    member _.TestTensorSumToSizeSystematic () =
        for combo in Combos.IntegralAndFloatingPoint do 
            // Systematically test all legitimate reductions of 2x2x2 to smaller sizes
            let t6 = combo.tensor([ [[1.; 2.]; [3.; 4.] ]; [[5.; 6.]; [7.; 8.] ] ])
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
                [|([||], combo.tensor (36., dtype=combo.dtype.SummationType));
                  ([|1|], combo.tensor ([36.], dtype=combo.dtype.SummationType));
                  ([|2|], combo.tensor ([16.; 20.], dtype=combo.dtype.SummationType));
                  ([|1; 1|], combo.tensor ([[36.]], dtype=combo.dtype.SummationType));
                  ([|1; 2|], combo.tensor ([[16.; 20.]], dtype=combo.dtype.SummationType));
                  ([|2; 1|], combo.tensor([[14.]; [22.]], dtype=combo.dtype.SummationType));
                  ([|2; 2|], combo.tensor([[6.; 8.]; [10.; 12.]], dtype=combo.dtype.SummationType));
                  ([|1; 1; 1|], combo.tensor([[[36.]]], dtype=combo.dtype.SummationType));
                  ([|1; 1; 2|], combo.tensor([[[16.; 20.]]], dtype=combo.dtype.SummationType));
                  ([|1; 2; 1|], combo.tensor([[[14.]; [22.]]], dtype=combo.dtype.SummationType));
                  ([|1; 2; 2|], combo.tensor([[[6.; 8.]; [10.; 12.]]], dtype=combo.dtype.SummationType));
                  ([|2; 1; 1|], combo.tensor([[[10.]]; [[26.]]], dtype=combo.dtype.SummationType));
                  ([|2; 1; 2|], combo.tensor([[[4.; 6.]]; [[12.; 14.]]], dtype=combo.dtype.SummationType));
                  ([|2; 2; 1|], combo.tensor([[[3.]; [7.]]; [[11.]; [15.]]], dtype=combo.dtype.SummationType));
                  ([|2; 2; 2|], combo.tensor([[[1.; 2.]; [3.; 4.]]; [[5.; 6.]; [7.; 8.]]], dtype=combo.dtype.SummationType))|]

            Assert.CheckEqual(systematicResults, expectedResults)

    [<Test>]
    member _.TestTensorSumT2Dim0 () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t1Sum = t1.sumT2Dim0()
            let t1SumCorrect = combo.tensor([4.; 6.])

            Assert.CheckEqual(t1SumCorrect, t1Sum)
            Assert.CheckEqual(t1Sum.dtype, combo.dtype)
    
    [<Test>]
    member _.TestTensorSumDim () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t = combo.tensor([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
            let tSum0 = t.sum(0)
            let tSum0Correct = combo.tensor([[14.0f, 16.0f, 18.0f, 20.0f], [22.0f, 24.0f, 26.0f, 28.0f], [30.0f, 32.0f, 34.0f, 36.0f]], dtype=combo.dtype.SummationType)
            let tSum1 = t.sum(1)
            let tSum1Correct = combo.tensor([[15.0f, 18.0f, 21.0f, 24.0f], [51.0f, 54.0f, 57.0f, 60.0f]], dtype=combo.dtype.SummationType)
            let tSum2 = t.sum(2)
            let tSum2Correct = combo.tensor([[10.0f, 26.0f, 42.0f], [58.0f, 74.0f, 90.0f]], dtype=combo.dtype.SummationType)

            Assert.CheckEqual(tSum0.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(tSum1.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(tSum2.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(tSum0Correct, tSum0)
            Assert.CheckEqual(tSum1Correct, tSum1)
            Assert.CheckEqual(tSum2Correct, tSum2)
    
    [<Test>]
    member _.TestTensorSumDimKeepDim () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t = combo.tensor([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
            let tSum0 = t.sum(0, keepDim=true)
            let tSum0Correct = combo.tensor([[[14.0f; 16.0f; 18.0f; 20.0f]; [22.0f; 24.0f; 26.0f; 28.0f]; [30.0f; 32.0f; 34.0f; 36.0f]]], dtype=combo.dtype.SummationType)
            let tSum1 = t.sum(1, keepDim=true)
            let tSum1Correct = combo.tensor([[[15.0f; 18.0f; 21.0f; 24.0f]]; [[51.0f; 54.0f; 57.0f; 60.0f]]], dtype=combo.dtype.SummationType)
            let tSum2 = t.sum(2, keepDim=true)
            let tSum2Correct = combo.tensor([[[10.0f]; [26.0f]; [42.0f]]; [[58.0f]; [74.0f]; [90.0f]]], dtype=combo.dtype.SummationType)

            Assert.CheckEqual(tSum0.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(tSum1.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(tSum2.dtype, combo.dtype.SummationType)
            Assert.CheckEqual(tSum0Correct, tSum0)
            Assert.CheckEqual(tSum1Correct, tSum1)
            Assert.CheckEqual(tSum2Correct, tSum2)

    [<Test>]
    member _.TestTensorSumDimBackwards () =
        for combo in Combos.FloatingPoint do 
            let t = combo.randn([2;2;2])
            let tsum_3 = t.sum(-3)
            let tsum_2 = t.sum(-2)
            let tsum_1 = t.sum(-1)
            let tsum0 = t.sum(0)
            let tsum1 = t.sum(1)
            let tsum2 = t.sum(2)

            Assert.CheckEqual(tsum_3, tsum0)
            Assert.CheckEqual(tsum_2, tsum1)
            Assert.CheckEqual(tsum_1, tsum2)

    [<Test>]
    member _.TestTensorMeanDimBackwards () =
        for combo in Combos.FloatingPoint do 
            let t = combo.randn([2;2;2])
            let tmean_3 = t.mean(-3)
            let tmean_2 = t.mean(-2)
            let tmean_1 = t.mean(-1)
            let tmean0 = t.mean(0)
            let tmean1 = t.mean(1)
            let tmean2 = t.mean(2)

            Assert.CheckEqual(tmean_3, tmean0)
            Assert.CheckEqual(tmean_2, tmean1)
            Assert.CheckEqual(tmean_1, tmean2)

    [<Test>]
    member _.TestTensorVarianceDimBackwards () =
        for combo in Combos.FloatingPoint do 
            let t = combo.randn([2;2;2])
            let tvariance_3 = t.variance(-3)
            let tvariance_2 = t.variance(-2)
            let tvariance_1 = t.variance(-1)
            let tvariance0 = t.variance(0)
            let tvariance1 = t.variance(1)
            let tvariance2 = t.variance(2)

            Assert.CheckEqual(tvariance_3, tvariance0)
            Assert.CheckEqual(tvariance_2, tvariance1)
            Assert.CheckEqual(tvariance_1, tvariance2)

    [<Test>]
    member _.TestTensorMean () =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
            let tMean = t.mean()
            let tMeanCorrect = combo.tensor(12.5)

            Assert.CheckEqual(tMeanCorrect, tMean)
            Assert.CheckEqual(tMean.dtype, combo.dtype)

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
            let tMean0Correct = combo.tensor([[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]])
            let tMean1 = t.mean(1)
            let tMean1Correct = combo.tensor([[5.; 6.; 7.; 8.]; [17.; 18.; 19.; 20.]])
            let tMean2 = t.mean(2)
            let tMean2Correct = combo.tensor([[2.5; 6.5; 10.5]; [14.5; 18.5; 22.5]])

            Assert.CheckEqual(tMean0Correct, tMean0)
            Assert.CheckEqual(tMean1Correct, tMean1)
            Assert.CheckEqual(tMean2Correct, tMean2)

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
            let tMeanKeepDim0Correct = combo.tensor([[[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]]])
            let tMeanKeepDim1 = t.mean(1, keepDim=true)
            let tMeanKeepDim1Correct = combo.tensor([[[5.; 6.; 7.; 8.]]; [[17.; 18.; 19.; 20.]]])
            let tMeanKeepDim2 = t.mean(2, keepDim=true)
            let tMeanKeepDim2Correct = combo.tensor([[[2.5]; [6.5]; [10.5]]; [[14.5]; [18.5]; [22.5]]])

            Assert.CheckEqual(tMeanKeepDim0, tMeanKeepDim0Correct)
            Assert.CheckEqual(tMeanKeepDim1, tMeanKeepDim1Correct)
            Assert.CheckEqual(tMeanKeepDim2, tMeanKeepDim2Correct)

    [<Test>]
    member _.TestTensorStddev () =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416];
                [0.6078;0.4742;0.7844;0.0967];
                [0.1416;0.1559;0.6452;0.1417]];
 
                [[0.0848;0.4156;0.5542;0.4166];
                [0.5187;0.0520;0.4763;0.1509];
                [0.4767;0.8096;0.1729;0.6671]]])
            let tStddev = t.stddev()
            let tStddevCorrect = combo.tensor(0.2398)

            Assert.True(tStddev.allclose(tStddevCorrect, 0.01))
            Assert.CheckEqual(tStddev.dtype, combo.dtype)

            // stddev, dim={0,1,2,3}, keepDim=true
            let tStddev0 = t.stddev(0)
            let tStddev0Correct = combo.tensor([[0.2078; 0.2375; 0.2326; 0.0530];
                [0.0630; 0.2985; 0.2179; 0.0383];
                [0.2370; 0.4623; 0.3339; 0.3715]])
            let tStddev1 = t.stddev(1)
            let tStddev1Correct = combo.tensor([[0.2331; 0.2981; 0.2911; 0.1304];
                [0.2393; 0.3789; 0.2014; 0.2581]])
            let tStddev2 = t.stddev(2)
            let tStddev2Correct = combo.tensor([[0.2277; 0.2918; 0.2495];[0.1996; 0.2328; 0.2753]])

            Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
            Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
            Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))
            Assert.CheckEqual(tStddev0.dtype, combo.dtype)
            Assert.CheckEqual(tStddev1.dtype, combo.dtype)
            Assert.CheckEqual(tStddev2.dtype, combo.dtype)

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
            let tStddev0Correct = combo.tensor([[[0.2078; 0.2375; 0.2326; 0.0530];[0.0630; 0.2985; 0.2179; 0.0383];[0.2370; 0.4623; 0.3339; 0.3715]]])
            let tStddev1 = t.stddev(1, keepDim=true)
            let tStddev1Correct = combo.tensor([[[0.2331; 0.2981; 0.2911; 0.1304]];[[0.2393; 0.3789; 0.2014; 0.2581]]])
            let tStddev2 = t.stddev(2, keepDim=true)
            let tStddev2Correct = combo.tensor([[[0.2277]; [0.2918]; [0.2495]];[[0.1996]; [0.2328]; [0.2753]]])

            Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
            Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
            Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))

    [<Test>]
    member _.TestTensorVariance () =
        for combo in Combos.FloatingPoint do 
            (* Python:
            import torch
            input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
            input.var()
            *)
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
            let tVariance = t.variance()
            let tVarianceCorrect = combo.tensor(0.0575)

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
            let tVariance0Correct = combo.tensor([[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]])
            let tVariance1 = t.variance(1)
            let tVariance1Correct = combo.tensor([[0.0543; 0.0888; 0.0847; 0.0170];[0.0573; 0.1436; 0.0406; 0.0666]])
            let tVariance2 = t.variance(2)
            let tVariance2Correct = combo.tensor([[0.0519; 0.0852; 0.0622];[0.0398; 0.0542; 0.0758]])

            Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
            Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
            Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))
            Assert.CheckEqual(tVariance0.dtype, combo.dtype)
            Assert.CheckEqual(tVariance1.dtype, combo.dtype)
            Assert.CheckEqual(tVariance2.dtype, combo.dtype)

            let tVarianceBiased = t.variance(unbiased=false)
            let tVarianceBiasedCorrect = combo.tensor(0.0551)

            Assert.True(tVarianceBiased.allclose(tVarianceBiasedCorrect, 0.01))

            let tVarianceBiased0 = t.variance(0, unbiased=false)
            let tVarianceBiased0Correct = combo.tensor([[0.0216, 0.0282, 0.0271, 0.0014],
                                                        [0.0020, 0.0446, 0.0237, 0.0007],
                                                        [0.0281, 0.1068, 0.0558, 0.0690]])
            let tVarianceBiased1 = t.variance(1, unbiased=false)
            let tVarianceBiased1Correct = combo.tensor([[0.0362, 0.0592, 0.0565, 0.0113],
                                                        [0.0382, 0.0957, 0.0271, 0.0444]])
            let tVarianceBiased2 = t.variance(2, unbiased=false)
            let tVarianceBiased2Correct = combo.tensor([[0.0389, 0.0639, 0.0467],
                                                        [0.0299, 0.0407, 0.0568]])

            Assert.True(tVarianceBiased0.allclose(tVarianceBiased0Correct, 0.01, 0.01))
            Assert.True(tVarianceBiased1.allclose(tVarianceBiased1Correct, 0.01, 0.01))
            Assert.True(tVarianceBiased2.allclose(tVarianceBiased2Correct, 0.01, 0.01))
            Assert.CheckEqual(tVarianceBiased0.dtype, combo.dtype)
            Assert.CheckEqual(tVarianceBiased1.dtype, combo.dtype)
            Assert.CheckEqual(tVarianceBiased2.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorVarianceKeepDim () =
        for combo in Combos.FloatingPoint do 
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
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
            let tVariance0 = t.variance(0, keepDim=true)
            let tVariance0Correct = combo.tensor([[[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]]])
            let tVariance1 = t.variance(1, keepDim=true)
            let tVariance1Correct = combo.tensor([[[0.0543; 0.0888; 0.0847; 0.0170]];[[0.0573; 0.1436; 0.0406; 0.0666]]])
            let tVariance2 = t.variance(2, keepDim=true)
            let tVariance2Correct = combo.tensor([[[0.0519];[0.0852];[0.0622]];[[0.0398];[0.0542];[0.0758]]])

            Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
            Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
            Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))
            Assert.CheckEqual(tVariance0.dtype, combo.dtype)
            Assert.CheckEqual(tVariance1.dtype, combo.dtype)
            Assert.CheckEqual(tVariance2.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorTransposeT () =
        for combo in Combos.All do 
            let t = combo.arange(24).view([2;4;3]).cast(combo.dtype)
            
            let t00 = t.transpose(0, 0)
            let t00Correct = t

            let t01 = t.transpose(0, 1)
            let t01Correct = combo.tensor([[[ 0,  1,  2],
                                             [12, 13, 14]],

                                            [[ 3,  4,  5],
                                             [15, 16, 17]],

                                            [[ 6,  7,  8],
                                             [18, 19, 20]],

                                            [[ 9, 10, 11],
                                             [21, 22, 23]]])
            let t02 = t.transpose(0, 2)
            let t02Correct = combo.tensor([[[ 0, 12],
                                             [ 3, 15],
                                             [ 6, 18],
                                             [ 9, 21]],

                                            [[ 1, 13],
                                             [ 4, 16],
                                             [ 7, 19],
                                             [10, 22]],

                                            [[ 2, 14],
                                             [ 5, 17],
                                             [ 8, 20],
                                             [11, 23]]])
            let t12 = t.transpose(1, 2)
            let t12Correct = combo.tensor([[[ 0,  3,  6,  9],
                                             [ 1,  4,  7, 10],
                                             [ 2,  5,  8, 11]],

                                            [[12, 15, 18, 21],
                                             [13, 16, 19, 22],
                                             [14, 17, 20, 23]]])

            Assert.CheckEqual(t00Correct, t00)
            Assert.CheckEqual(t01Correct, t01)
            Assert.CheckEqual(t02Correct, t02)
            Assert.CheckEqual(t12Correct, t12)
            Assert.CheckEqual(t00.dtype, combo.dtype)
            Assert.CheckEqual(t01.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorTransposeT2 () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]])
            let t1Transpose = t1.transpose()
            let t1TransposeCorrect = combo.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])

            let t2 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t2TransposeTranspose = t2.transpose().transpose()
            let t2TransposeTransposeCorrect = t2

            Assert.CheckEqual(t1TransposeCorrect, t1Transpose)
            Assert.CheckEqual(t2TransposeTransposeCorrect, t2TransposeTranspose)
            Assert.CheckEqual(t1Transpose.dtype, combo.dtype)
            Assert.CheckEqual(t2TransposeTranspose.dtype, combo.dtype)

    member _.TestTensorSignT () =
        // Test all signed types
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.])
            let t1Sign = t1.sign()
            let t1SignCorrect = combo.tensor([-1.; -1.; 0.; 1.])

            Assert.CheckEqual(t1SignCorrect, t1Sign)
            Assert.CheckEqual(t1Sign.dtype, combo.dtype)

        // Test all signed types
        for combo in Combos.UnsignedIntegral do 
            let t1 = combo.tensor([1; 1; 0; 3])
            let t1Sign = t1.sign()
            let t1SignCorrect = combo.tensor([1; 1; 0; 1])

            Assert.CheckEqual(t1SignCorrect, t1Sign)
            Assert.CheckEqual(t1Sign.dtype, combo.dtype)

        // Test bool type separately
        // Note, PyTorch 'torch.tensor([True, False]).sign()' gives 'tensor([ True, False])'
        for combo in Combos.AllDevicesAndBackends do
            let t1Bool = combo.tensor([true;false], dtype=Dtype.Bool)
            let t1BoolSignCorrect = combo.tensor([true; false], dtype=Dtype.Bool)

            Assert.CheckEqual(t1BoolSignCorrect, t1Bool.sign())

    [<Test>]
    member _.TestTensorFloorT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Floor = t1.floor()
            let t1FloorCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            Assert.True(t1Floor.allclose(t1FloorCorrect, 0.01))
            Assert.CheckEqual(t1Floor.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).floor())

    [<Test>]
    member _.TestTensorCeilT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Ceil = t1.ceil()
            let t1CeilCorrect = combo.tensor([1.; 1.; 1.; 1.; 1.])

            Assert.True(t1Ceil.allclose(t1CeilCorrect, 0.01))
            Assert.CheckEqual(t1Ceil.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).ceil())

    [<Test>]
    member _.TestTensorRoundT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Round = t1.round()
            let t1RoundCorrect = combo.tensor([1.; 0.; 0.; 1.; 1.])

            Assert.True(t1Round.allclose(t1RoundCorrect, 0.01))
            Assert.CheckEqual(t1Round.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).round())

    [<Test>]
    member _.TestTensorAbsT () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.])
            let t1Abs = t1.abs()
            let t1AbsCorrect = combo.tensor([1.; 2.; 0.; 3.])

            Assert.CheckEqual(t1AbsCorrect, t1Abs)
            Assert.CheckEqual(t1Abs.dtype, combo.dtype)

        for combo in Combos.UnsignedIntegral do 
            let t1 = combo.tensor([1.; 2.; 0.; 3.])
            let t1Abs = t1.abs()
            let t1AbsCorrect = combo.tensor([1.; 2.; 0.; 3.])

            Assert.CheckEqual(t1AbsCorrect, t1Abs)
            Assert.CheckEqual(t1Abs.dtype, combo.dtype)

        // Test bool separately
        // Note: PyTorch fails on 'torch.tensor([True, False]).abs()'
        for combo in Combos.AllDevicesAndBackends do
            let t1 = combo.tensor([true; false], dtype=Dtype.Bool)
            isInvalidOp (fun () -> t1.abs())

    [<Test>]
    member _.TestTensorReluT () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.; 10.])
            let t1Relu = t1.relu()
            let t1ReluCorrect = combo.tensor([0.; 0.; 0.; 3.; 10.])

            Assert.CheckEqual(t1ReluCorrect, t1Relu)
            Assert.CheckEqual(t1Relu.dtype, combo.dtype)

        // Test bool separately
        for combo in Combos.AllDevicesAndBackends do
            let t1 = combo.tensor([true; false], dtype=Dtype.Bool)
            isInvalidOp (fun () -> t1.relu())

    [<Test>]
    member _.TestTensorLeakyRelu () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.; 10.])
            let t1LeakyRelu = t1.leakyRelu()
            let t1LeakyReluCorrect = combo.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])

            Assert.CheckEqual(t1LeakyReluCorrect, t1LeakyRelu)
            Assert.CheckEqual(t1LeakyRelu.dtype, combo.dtype)
            Assert.CheckEqual(t1LeakyRelu.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorSigmoidT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Sigmoid = t1.sigmoid()
            let t1SigmoidCorrect = combo.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])

            Assert.True(t1Sigmoid.allclose(t1SigmoidCorrect, 0.01))
            Assert.CheckEqual(t1Sigmoid.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
          isInvalidOp(fun () -> combo.tensor([1.0]).sigmoid())

    [<Test>]
    member _.TestTensorSoftplusT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05])
            let t1Softplus = t1.softplus()
            let t1SoftplusCorrect = combo.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])

            Assert.True(t1Softplus.allclose(t1SoftplusCorrect, 0.01))
            Assert.CheckEqual(t1Softplus.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).softplus())

    [<Test>]
    member _.TestTensorExpT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9139; -0.5907;  1.9422; -0.7763; -0.3274])
            let t1Exp = t1.exp()
            let t1ExpCorrect = combo.tensor([2.4940; 0.5539; 6.9742; 0.4601; 0.7208])

            Assert.True(t1Exp.allclose(t1ExpCorrect, 0.01))
            Assert.CheckEqual(t1Exp.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).exp())

    [<Test>]
    member _.TestTensorLogT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
            let t1Log = t1.log()
            let t1LogCorrect = combo.tensor([-2.0516; -0.5426; -0.4301; -0.9727; -0.9100])

            Assert.True(t1Log.allclose(t1LogCorrect, 0.01))
            Assert.CheckEqual(t1Log.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).log())

    [<Test>]
    member _.TestTensorLog10T () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
            let t1Log10 = t1.log10()
            let t1Log10Correct = combo.tensor([-0.8911; -0.2357; -0.1868; -0.4224; -0.3952])

            Assert.True(t1Log10.allclose(t1Log10Correct, 0.01))
            Assert.CheckEqual(t1Log10.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).log10())

    [<Test>]
    member _.TestTensorSqrtT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Sqrt = t1.sqrt()
            let t1SqrtCorrect = combo.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])

            Assert.True(t1Sqrt.allclose(t1SqrtCorrect, 0.01))
            Assert.CheckEqual(t1Sqrt.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sqrt())

    [<Test>]
    member _.TestTensorSinT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Sin = t1.sin()
            let t1SinCorrect = combo.tensor([-0.9828;  0.9991; -0.3698; -0.7510;  0.9491])

            Assert.True(t1Sin.allclose(t1SinCorrect, 0.01))
            Assert.CheckEqual(t1Sin.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sin())

    [<Test>]
    member _.TestTensorCosT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Cos = t1.cos()
            let t1CosCorrect = combo.tensor([-0.1849;  0.0418; -0.9291;  0.6603;  0.3150])

            Assert.True(t1Cos.allclose(t1CosCorrect, 0.01))
            Assert.CheckEqual(t1Cos.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).cos())

    [<Test>]
    member _.TestTensorTanT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Tan = t1.tan()
            let t1TanCorrect = combo.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])

            Assert.True(t1Tan.allclose(t1TanCorrect, 0.01))
            Assert.CheckEqual(t1Tan.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).tan())

    [<Test>]
    member _.TestTensorSinhT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Sinh = t1.sinh()
            let t1SinhCorrect = combo.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])

            Assert.True(t1Sinh.allclose(t1SinhCorrect, 0.01))
            Assert.CheckEqual(t1Sinh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sinh())

    [<Test>]
    member _.TestTensorCoshT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Cosh = t1.cosh()
            let t1CoshCorrect = combo.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])

            Assert.True(t1Cosh.allclose(t1CoshCorrect, 0.01))
            Assert.CheckEqual(t1Cosh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).cosh())

    [<Test>]
    member _.TestTensorTanhT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Tanh = t1.tanh()
            let t1TanhCorrect = combo.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])

            Assert.True(t1Tanh.allclose(t1TanhCorrect, 0.01))
            Assert.CheckEqual(t1Tanh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).tanh())

    [<Test>]
    member _.TestTensorAsinT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Asin = t1.asin()
            let t1AsinCorrect = combo.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])

            Assert.True(t1Asin.allclose(t1AsinCorrect, 0.01))
            Assert.CheckEqual(t1Asin.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).asin())

    [<Test>]
    member _.TestTensorAcosT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Acos = t1.acos()
            let t1AcosCorrect = combo.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])

            Assert.True(t1Acos.allclose(t1AcosCorrect, 0.01))
            Assert.CheckEqual(t1Acos.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).acos())

    [<Test>]
    member _.TestTensorAtanT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Atan = t1.atan()
            let t1AtanCorrect = combo.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])

            Assert.True(t1Atan.allclose(t1AtanCorrect, 0.01))
            Assert.CheckEqual(t1Atan.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).atan())

    [<Test>]
    member _.TestTensorSlice () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1.;2.])
            let t1s1 = t1.[0]
            let t1s2 = t1.[*]
            let t1s1Correct = combo.tensor(1.)
            let t1s2Correct = combo.tensor([1.;2.])

            let t2 = combo.tensor([[1.;2.];[3.;4.]])
            let t2s1 = t2.[0]
            let t2s2 = t2.[*]
            let t2s3 = t2.[0,0]
            let t2s4 = t2.[0,*]
            let t2s5 = t2.[*,0]
            let t2s6 = t2.[*,*]
            let t2s1Correct = combo.tensor([1.;2.])
            let t2s2Correct = combo.tensor([[1.;2.];[3.;4.]])
            let t2s3Correct = combo.tensor(1.)
            let t2s4Correct = combo.tensor([1.;2.])
            let t2s5Correct = combo.tensor([1.;3.])
            let t2s6Correct = combo.tensor([[1.;2.];[3.;4.]])

            let t2b = combo.tensor([[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]])
            let t2bs1 = t2b.[1..,2..]
            let t2bs1Correct = combo.tensor([[7.;8.];[11.;12.]])
            let t2bs2 = t2b.[1..2,2..3]
            let t2bs2Correct = combo.tensor([[7.;8.];[11.;12.]])

            let t3 = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
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
            let t3s1Correct  = combo.tensor([[1.;2.];[3.;4.]])
            let t3s2Correct  = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
            let t3s3Correct  = combo.tensor([1.;2.])
            let t3s4Correct  = combo.tensor([[1.;2.];[3.;4.]])
            let t3s5Correct  = combo.tensor([[1.;2.];[5.;6.]])
            let t3s6Correct  = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
            let t3s7Correct  = combo.tensor(1.)
            let t3s8Correct  = combo.tensor([1.;2.])
            let t3s9Correct  = combo.tensor([1.;3.])
            let t3s10Correct = combo.tensor([[1.;2.];[3.;4.]])
            let t3s11Correct = combo.tensor([1.;5.])
            let t3s12Correct = combo.tensor([[1.;2.];[5.;6.]])
            let t3s13Correct = combo.tensor([[1.;3.];[5.;7.]])
            let t3s14Correct = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])

            let t4 = combo.tensor([[[[1.]]; 
                                     [[2.]]; 
                                     [[3.]]]; 
                                    [[[4.]]; 
                                     [[5.]]; 
                                     [[6.]]]])
            let t4s1 = t4.[0]
            let t4s2 = t4.[0,*,*,*]
            let t4s1Correct = combo.tensor([[[1]];
                                             [[2]];
                                             [[3]]])
            let t4s2Correct = t4s1Correct

            Assert.CheckEqual(t1s1Correct, t1s1)
            Assert.CheckEqual(t1s2Correct, t1s2)

            Assert.CheckEqual(t2s1Correct, t2s1)
            Assert.CheckEqual(t2s2Correct, t2s2)
            Assert.CheckEqual(t2s3Correct, t2s3)
            Assert.CheckEqual(t2s4Correct, t2s4)
            Assert.CheckEqual(t2s5Correct, t2s5)
            Assert.CheckEqual(t2s6Correct, t2s6)

            Assert.CheckEqual(t2bs1Correct, t2bs1)
            Assert.CheckEqual(t2bs2Correct, t2bs2)

            Assert.CheckEqual(t3s1Correct, t3s1)
            Assert.CheckEqual(t3s2Correct, t3s2)
            Assert.CheckEqual(t3s3Correct, t3s3)
            Assert.CheckEqual(t3s4Correct, t3s4)
            Assert.CheckEqual(t3s5Correct, t3s5)
            Assert.CheckEqual(t3s6Correct, t3s6)
            Assert.CheckEqual(t3s7Correct, t3s7)
            Assert.CheckEqual(t3s8Correct, t3s8)
            Assert.CheckEqual(t3s9Correct, t3s9)
            Assert.CheckEqual(t3s10Correct, t3s10)
            Assert.CheckEqual(t3s11Correct, t3s11)
            Assert.CheckEqual(t3s12Correct, t3s12)
            Assert.CheckEqual(t3s13Correct, t3s13)
            Assert.CheckEqual(t3s14Correct, t3s14)

            Assert.CheckEqual(t4s1Correct, t4s1)
            Assert.CheckEqual(t4s2Correct, t4s2)

            Assert.CheckEqual(t1s1.dtype, combo.dtype)
            Assert.CheckEqual(t1s2.dtype, combo.dtype)

            Assert.CheckEqual(t2s1.dtype, combo.dtype)
            Assert.CheckEqual(t2s2.dtype, combo.dtype)
            Assert.CheckEqual(t2s3.dtype, combo.dtype)
            Assert.CheckEqual(t2s4.dtype, combo.dtype)
            Assert.CheckEqual(t2s5.dtype, combo.dtype)
            Assert.CheckEqual(t2s6.dtype, combo.dtype)

            Assert.CheckEqual(t2bs1.dtype, combo.dtype)
            Assert.CheckEqual(t2bs2.dtype, combo.dtype)

            Assert.CheckEqual(t3s1.dtype, combo.dtype)
            Assert.CheckEqual(t3s2.dtype, combo.dtype)
            Assert.CheckEqual(t3s3.dtype, combo.dtype)
            Assert.CheckEqual(t3s4.dtype, combo.dtype)
            Assert.CheckEqual(t3s5.dtype, combo.dtype)
            Assert.CheckEqual(t3s6.dtype, combo.dtype)
            Assert.CheckEqual(t3s7.dtype, combo.dtype)
            Assert.CheckEqual(t3s8.dtype, combo.dtype)
            Assert.CheckEqual(t3s9.dtype, combo.dtype)
            Assert.CheckEqual(t3s10.dtype, combo.dtype)
            Assert.CheckEqual(t3s11.dtype, combo.dtype)
            Assert.CheckEqual(t3s12.dtype, combo.dtype)
            Assert.CheckEqual(t3s13.dtype, combo.dtype)
            Assert.CheckEqual(t3s14.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorAddTTSlice () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]])
            let t2 = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]])
            let t3 = t1.addSlice([0;1], t2)
            let t3Correct = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.CheckEqual(t3.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorPad () =
        for combo in Combos.All do
            let t1 = combo.tensor([1.,2.,3.])
            let t1p0 = dsharp.pad(t1, [0])
            let t1p0Correct = combo.tensor([1.,2.,3.])
            let t1p1 = dsharp.pad(t1, [1])
            let t1p1Correct = combo.tensor([0.,1.,2.,3.,0.])
            let t1p2 = dsharp.pad(t1, [2])
            let t1p2Correct = combo.tensor([0.,0.,1.,2.,3.,0.,0.])
            let t2 = combo.tensor([[1.,2.,3.], [4.,5.,6.]])
            let t2p00 = dsharp.pad(t2, [0;0])
            let t2p00Correct = combo.tensor([[1.,2.,3.], [4.,5.,6.]])
            let t2p12 = dsharp.pad(t2, [1;2])
            let t2p12Correct = combo.tensor([[0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 2, 3, 0, 0],
                                              [0, 0, 4, 5, 6, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0]])
            let t2p22 = dsharp.pad(t2, [2;2])
            let t2p22Correct = combo.tensor([[0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 2, 3, 0, 0],
                                                [0, 0, 4, 5, 6, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0]])
            Assert.CheckEqual(t1p0Correct, t1p0)
            Assert.CheckEqual(t1p1Correct, t1p1)
            Assert.CheckEqual(t1p2Correct, t1p2)
            Assert.CheckEqual(t2p00Correct, t2p00)
            Assert.CheckEqual(t2p12Correct, t2p12)
            Assert.CheckEqual(t2p22Correct, t2p22)


    [<Test>]
    member _.TestTensorExpandT () =
        for combo in Combos.All do 
            let t1 = combo.tensor(1.0)
            let t1Expand = t1.expand([2;3])
            let t1ExpandCorrect = combo.tensor([[1.;1.;1.];[1.;1.;1.]])
            Assert.CheckEqual(t1ExpandCorrect, t1Expand)

            let t2 = combo.tensor([1.0])
            let t2Expand = t2.expand([2;3])
            let t2ExpandCorrect = combo.tensor([[1.;1.;1.];[1.;1.;1.]])

            Assert.CheckEqual(t2ExpandCorrect, t2Expand)

            let t3 = combo.tensor([1.; 2.]) // 2
            let t3Expand = t3.expand([3;2]) // 3x2
            let t3ExpandCorrect = combo.tensor([[1.;2.];[1.;2.];[1.;2.]]) // 3x2

            Assert.CheckEqual(t3ExpandCorrect, t3Expand)

            let t4 = combo.tensor([[1.]; [2.]]) // 2x1
            let t4Expand = t4.expand([2;2]) // 2x2
            let t4ExpandCorrect = combo.tensor([[1.;1.];[2.;2.]])

            Assert.CheckEqual(t4ExpandCorrect, t4Expand)

            let t5 = combo.tensor([[1.]; [2.]]) // 2x1
            let t5Expand = t5.expand([2;2;2]) // 2x2x2
            let t5ExpandCorrect = combo.tensor([[[1.;1.];[2.;2.]];[[1.;1.];[2.;2.]]])

            Assert.CheckEqual(t5ExpandCorrect, t5Expand)

            let t6 = combo.tensor([[1.]; [2.]; [3.]]) // 3x1
            let t6Expand = t6.expand([-1;4]) // 3x4
            let t6ExpandCorrect = combo.tensor([[1.;1.;1.;1.];[2.;2.;2.;2.];[3.;3.;3.;3.]])

            Assert.CheckEqual(t6ExpandCorrect, t6Expand)

            isAnyException(fun () -> t6.expand([-1;3;4]))

            let t6Expand2 = t6.expand([2;-1;-1]) // 2x3x1
            let t6ExpandCorrect2 = combo.tensor([[[1.]; [2.]; [3.]] ; [[1.]; [2.]; [3.]]])
            Assert.CheckEqual(t6ExpandCorrect2, t6Expand2)

    [<Test>]
    member _.TestTensorExpandAs () =
        for combo in Combos.All do
            let t1 = combo.tensor([[1], [2], [3]])
            let t2 = combo.zeros([3;2])
            let t1Expand = t1.expandAs(t2)
            let t1ExpandCorrect = combo.tensor([[1, 1],
                                                [2, 2],
                                                [3, 3]])
            Assert.CheckEqual(t1ExpandCorrect, t1Expand)

    [<Test>]
    member _.TestTensorSqueezeT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[[1.; 2.]]; [[3.;4.]]])
            let t1Squeeze = t1.squeeze()
            let t1SqueezeCorrect = combo.tensor([[1.;2.];[3.;4.]])

            Assert.True(t1Squeeze.allclose(t1SqueezeCorrect, 0.01))
            Assert.CheckEqual(t1Squeeze.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorUnsqueezeT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.;2.];[3.;4.]])
            let t1Unsqueeze = t1.unsqueeze(1)
            let t1UnsqueezeCorrect = combo.tensor([[[1.; 2.]]; [[3.;4.]]])

            Assert.True(t1Unsqueeze.allclose(t1UnsqueezeCorrect, 0.01))
            Assert.CheckEqual(t1Unsqueeze.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorFlipT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.;2.];[3.;4.]])
            let t2 = t1.flip([|0|])
            let t2Correct = combo.tensor([[3.;4.]; [1.;2.]])
            let t3 = t1.flip([|1|])
            let t3Correct = combo.tensor([[2.;1.]; [4.;3.]])
            let t4 = t1.flip([|0; 1|])
            let t4Correct = combo.tensor([[4.;3.]; [2.;1.]])
            let t5 = t1.flip([|0; 1|]).flip([|0; 1|])
            let t5Correct = combo.tensor([[1.;2.]; [3.;4.]])

            Assert.CheckEqual(t2Correct, t2)
            Assert.CheckEqual(t3Correct, t3)
            Assert.CheckEqual(t4Correct, t4)
            Assert.CheckEqual(t5Correct, t5)

    [<Test>]
    member _.TestTensorDilateT () =
        for combo in Combos.FloatingPoint do 
            let tin1 = combo.tensor([1.;2.;3.])
            let t1 = tin1.dilate([|2|])
            let t1Correct = combo.tensor([1.;0.;2.;0.;3.])

            Assert.CheckEqual(t1Correct, t1)

            let tin2 = combo.tensor([[1.;2.]; [3.;4.]])
            let t2 = tin2.dilate([|1; 2|])
            let t2Correct = combo.tensor([[1.;0.;2.];[3.;0.;4.]])

            Assert.CheckEqual(t2Correct, t2)
            Assert.CheckEqual(combo.dtype, t2.dtype)

            let t3 = tin2.dilate([|2; 2|])
            let t3Correct = combo.tensor([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])

            Assert.CheckEqual(t3Correct, t3)
            Assert.CheckEqual(combo.dtype, t3.dtype)

            let tin5 = combo.tensor([1.;2.;3.;4.])
            let t5 = tin5.dilate([|3|])
            let t5Correct = combo.tensor([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])

            Assert.CheckEqual(t5Correct, t5)
            Assert.CheckEqual(combo.dtype, t5.dtype)

            // Dilate 3D 1; 1; 2
            let tin6 = combo.tensor([[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]])
            let t6 = tin6.dilate([|1; 1; 2|])
            let t6Correct = combo.tensor([[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]])

            Assert.CheckEqual(t6Correct, t6)
            Assert.CheckEqual(combo.dtype, t6.dtype)

            // Dilate 4D 1; 1; 1; 2
            let tin7 = combo.tensor([[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]];[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]]])
            let t7 = tin7.dilate([|1; 1; 1; 2|])
            let t7Correct = combo.tensor([[[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]; [[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]])

            Assert.CheckEqual(t7Correct, t7)
            Assert.CheckEqual(combo.dtype, t7.dtype)

            let tin8 = combo.tensor([[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]])
            let t8 = tin8.dilate([|2; 1; 2|])
            let t8Correct = combo.tensor([[[1.;0.;2.];[3.;0.;4.]]; [[0.;0.;0.];[0.;0.;0.]]; [[5.;0.;6.];[7.;0.;8.]]])

            Assert.CheckEqual(t8Correct, t8)
            Assert.CheckEqual(combo.dtype, t8.dtype)

            // Dilate 4D, 2; 1; 1; 2
            let tin9 = combo.tensor([[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]];[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]]])
            let t9 = tin9.dilate([|2; 1; 1; 2|])
            let t9Correct = combo.tensor([[[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]; 
                                          [[[0.;0.;0.];[0.;0.;0.]]; [[0.;0.;0.];[0.;0.;0.]]]; 
                                          [[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]])

            Assert.CheckEqual(t9Correct, t9)
            Assert.CheckEqual(combo.dtype, t9.dtype)

    [<Test>]
    member _.TestTensorUndilateT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.;0.;2.];[3.;0.;4.]])
            let t2 = t1.undilate([|1; 2|])
            let t2Correct = combo.tensor([[1.;2.]; [3.;4.]])
            let t3 = combo.tensor([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])
            let t4 = t3.undilate([|2; 2|])
            let t4Correct = combo.tensor([[1.;2.]; [3.;4.]])
            let t5 = combo.tensor([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])
            let t6 = t5.undilate([|3|])
            let t6Correct = combo.tensor([1.;2.;3.;4.])

            Assert.CheckEqual(t2Correct, t2)
            Assert.CheckEqual(t4Correct, t4)
            Assert.CheckEqual(t6Correct, t6)
            Assert.CheckEqual(combo.dtype, t2.dtype)
            Assert.CheckEqual(combo.dtype, t4.dtype)
            Assert.CheckEqual(combo.dtype, t6.dtype)

    [<Test>]
    member _.TestTensorClampT () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t = combo.tensor([-4,-3,-2,-1,0,1,2,3,4])
            let tClamped = dsharp.clamp(t, -2, 3)
            let tClampedCorrect = combo.tensor([-2, -2, -2, -1,  0,  1,  2,  3,  3])
            Assert.CheckEqual(tClampedCorrect, tClamped)

    [<Test>]
    member _.TestTensorView () =
        for combo in Combos.All do 
            let t = combo.randint(0, 2, [10;10])
            let t1 = t.view(-1)
            let t1Shape = t1.shape
            let t1ShapeCorrect = [|100|]
            let t2Shape = t.view([-1;50]).shape
            let t2ShapeCorrect = [|2;50|]
            let t3Shape = t.view([2;-1;50]).shape
            let t3ShapeCorrect = [|2;1;50|]
            let t4Shape = t.view([2;-1;10]).shape
            let t4ShapeCorrect = [|2;5;10|]
        
            Assert.CheckEqual(t1ShapeCorrect, t1Shape)
            Assert.CheckEqual(t2ShapeCorrect, t2Shape)
            Assert.CheckEqual(t3ShapeCorrect, t3Shape)
            Assert.CheckEqual(t4ShapeCorrect, t4Shape)
            Assert.CheckEqual(t1.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorViewAs () =
        for combo in Combos.All do
            let t1 = combo.tensor([1,2,3,4,5,6])
            let t2 = combo.zeros([3;2])
            let t1View = t1.viewAs(t2)
            let t1ViewCorrect = combo.tensor([[1, 2],
                                                [3, 4],
                                                [5, 6]])
            Assert.CheckEqual(t1ViewCorrect, t1View)

    [<Test>]
    member _.TestTensorFlatten () =
        for combo in Combos.All do 
            let t1 = combo.randint(0, 2, [5;5;5;5])
            let t1f1shape = dsharp.flatten(t1).shape
            let t1f1shapeCorrect = [|625|]
            let t1f2shape = dsharp.flatten(t1, startDim=1).shape
            let t1f2shapeCorrect = [|5; 125|]
            let t1f3shape = dsharp.flatten(t1, startDim=1, endDim=2).shape
            let t1f3shapeCorrect = [|5; 25; 5|]

            let t2 = combo.randint(0, 2, 5)
            let t2fshape = dsharp.flatten(t2).shape
            let t2fshapeCorrect = [|5|]

            let t3 = combo.tensor(2.5)
            let t3fshape = dsharp.flatten(t3).shape
            let t3fshapeCorrect = [||]

            Assert.CheckEqual(t1f1shapeCorrect, t1f1shape)
            Assert.CheckEqual(t1f2shapeCorrect, t1f2shape)
            Assert.CheckEqual(t1f3shapeCorrect, t1f3shape)
            Assert.CheckEqual(t2fshapeCorrect, t2fshape)
            Assert.CheckEqual(t3fshapeCorrect, t3fshape)

    [<Test>]
    member _.TestTensorGather () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1,2,3,4,5])
            let t1g = dsharp.gather(t1, 0, combo.tensor([0,2,3], dtype=Dtype.Int32))
            let t1gCorrect = combo.tensor([1, 3, 4])

            let t2 = combo.tensor([[1,2],[3,4]])
            let t2g0 = dsharp.gather(t2, 0, combo.tensor([[0,1],[1,0]], dtype=Dtype.Int32))
            let t2g0Correct = combo.tensor([[1, 4],
                                             [3, 2]])
            let t2g1 = dsharp.gather(t2, 1, combo.tensor([[0,0,1],[1,0,0]], dtype=Dtype.Int32))
            let t2g1Correct = combo.tensor([[1, 1, 2],
                                             [4, 3, 3]])

            Assert.CheckEqual(t1gCorrect, t1g)
            Assert.CheckEqual(combo.dtype, t1g.dtype)

            Assert.CheckEqual(t2g0Correct, t2g0)
            Assert.CheckEqual(combo.dtype, t2g0.dtype)

            Assert.CheckEqual(t2g1Correct, t2g1)
            Assert.CheckEqual(combo.dtype, t2g1.dtype)

    [<Test>]
    member _.TestTensorMax () =
        for combo in Combos.All do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t1Max = t1.max()
            let t1MaxCorrect = combo.tensor(20.)

            let t2 = combo.tensor([[1.;4.];[2.;3.]])
            let t2Max = t2.max()
            let t2MaxCorrect = combo.tensor(4.)

            let t3 = combo.tensor([[[ 7.6884; 65.9125;  4.0114];
                                 [46.7944; 61.5331; 40.1627];
                                 [48.3240;  4.9910; 50.1571]];

                                [[13.4777; 65.7656; 36.8161];
                                 [47.8268; 42.2229;  5.6115];
                                 [43.4779; 77.8675; 95.7660]];

                                [[59.8422; 47.1146; 36.7614];
                                 [71.6328; 18.5912; 27.7328];
                                 [49.9120; 60.3023; 53.0838]]])

            let t3Max = t3.max()
            let t3MaxCorrect = combo.tensor(95.7660)
        
            let t4 = combo.tensor([[[[8.8978; 8.0936];
                                  [4.8087; 1.0921];
                                  [8.5664; 3.7814]];

                                 [[2.3581; 3.7361];
                                  [1.0436; 6.0353];
                                  [7.7843; 8.7153]];

                                 [[3.9188; 6.7906];
                                  [9.1242; 4.8711];
                                  [1.7870; 9.7456]];
                                 [[5.0444; 0.5447];
                                  [6.2945; 5.9047];
                                  [8.0867; 3.1606]]]])

            let t4Max = t4.max()
            let t4MaxCorrect = combo.tensor(9.7456)

            Assert.CheckEqual(t1MaxCorrect, t1Max)
            Assert.CheckEqual(t2MaxCorrect, t2Max)
            Assert.CheckEqual(t3MaxCorrect, t3Max)
            Assert.CheckEqual(t4MaxCorrect, t4Max)
            Assert.CheckEqual(t1Max.dtype, combo.dtype)
            Assert.CheckEqual(t2Max.dtype, combo.dtype)
            Assert.CheckEqual(t3Max.dtype, combo.dtype)
            Assert.CheckEqual(t4Max.dtype, combo.dtype)


    [<Test>]
    member _.TestTensorMin () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t1Min = t1.min()
            let t1MinCorrect = combo.tensor(1.)

            let t2 = combo.tensor([[1.;4.];[2.;3.]])
            let t2Min = t2.min()
            let t2MinCorrect = combo.tensor(1.)

            let t3 = combo.tensor([[[ 7.6884; 65.9125;  4.0114];
                 [46.7944; 61.5331; 40.1627];
                 [48.3240;  4.9910; 50.1571]];

                [[13.4777; 65.7656; 36.8161];
                 [47.8268; 42.2229;  5.6115];
                 [43.4779; 77.8675; 95.7660]];

                [[59.8422; 47.1146; 36.7614];
                 [71.6328; 18.5912; 27.7328];
                 [49.9120; 60.3023; 53.0838]]])
            let t3Min = t3.min()
            let t3MinCorrect = combo.tensor(4.0114)
       
            let t4 = combo.tensor([[[[8.8978; 8.0936];
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
            let t4MinCorrect = combo.tensor(0.5370)

            Assert.CheckEqual(t1MinCorrect, t1Min)
            Assert.CheckEqual(t2MinCorrect, t2Min)
            Assert.CheckEqual(t3MinCorrect, t3Min)
            Assert.CheckEqual(t4MinCorrect, t4Min)
            Assert.CheckEqual(t1Min.dtype, combo.dtype)
            Assert.CheckEqual(t2Min.dtype, combo.dtype)
            Assert.CheckEqual(t3Min.dtype, combo.dtype)
            Assert.CheckEqual(t4Min.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorMaxBinary () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[-4.9385; 12.6206; 10.1783];
                [-2.9624; 17.6992;  2.2506];
                [-2.3536;  8.0772; 13.5639]])
            let t2 = combo.tensor([[  0.7027;  22.3251; -11.4533];
                [  3.6887;   4.3355;   3.3767];
                [  0.1203;  -5.4088;   1.5658]])
            let t3 = t1.max(t2)
            let t3Correct = combo.tensor([[ 0.7027; 22.3251; 10.1783];
                [ 3.6887; 17.6992;  3.3767];
                [ 0.1203;  8.0772; 13.5639]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.CheckEqual(t3.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorMinBinary () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[-4.9385; 12.6206; 10.1783];
                [-2.9624; 17.6992;  2.2506];
                [-2.3536;  8.0772; 13.5639]])
            let t2 = combo.tensor([[  0.7027;  22.3251; -11.4533];
                [  3.6887;   4.3355;   3.3767];
                [  0.1203;  -5.4088;   1.5658]])
            let t3 = t1.min(t2)
            let t3Correct = combo.tensor([[ -4.9385;  12.6206; -11.4533];
                [ -2.9624;   4.3355;   2.2506];
                [ -2.3536;  -5.4088;   1.5658]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.CheckEqual(t3.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorSoftmax () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([2.7291; 0.0607; 0.8290])
            let t1Softmax0 = t1.softmax(0)
            let t1Softmax0Correct = combo.tensor([0.8204; 0.0569; 0.1227])

            let t2 = combo.tensor([[1.3335; 1.6616; 2.4874; 6.1722];
                [3.3478; 9.3019; 1.0844; 8.9874];
                [8.6300; 1.8842; 9.1387; 9.1321]])
            let t2Softmax0 = t2.softmax(0)
            let t2Softmax0Correct = combo.tensor([[6.7403e-04; 4.8014e-04; 1.2904e-03; 2.7033e-02];
                [5.0519e-03; 9.9892e-01; 3.1723e-04; 4.5134e-01];
                [9.9427e-01; 5.9987e-04; 9.9839e-01; 5.2163e-01]])
            let t2Softmax1 = t2.softmax(1)
            let t2Softmax1Correct = combo.tensor([[7.5836e-03; 1.0528e-02; 2.4044e-02; 9.5784e-01];
                [1.4974e-03; 5.7703e-01; 1.5573e-04; 4.2131e-01];
                [2.3167e-01; 2.7240e-04; 3.8528e-01; 3.8277e-01]])

            let t3 = combo.tensor([[[3.0897; 2.0902];
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
            let t3Softmax0Correct = combo.tensor([[[2.4662e-03; 3.7486e-03];
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
            let t3Softmax1Correct = combo.tensor([[[1.8050e-01; 1.2351e-03];
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
            let t3Softmax2Correct = combo.tensor([[[7.3096e-01; 2.6904e-01];
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
            Assert.CheckEqual(t1Softmax0.dtype, combo.dtype)
            Assert.CheckEqual(t2Softmax0.dtype, combo.dtype)
            Assert.CheckEqual(t2Softmax1.dtype, combo.dtype)
            Assert.CheckEqual(t3Softmax0.dtype, combo.dtype)
            Assert.CheckEqual(t3Softmax1.dtype, combo.dtype)
            Assert.CheckEqual(t3Softmax2.dtype, combo.dtype)


    [<Test>]
    member _.TestTensorLogsoftmax () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([2.7291, 0.0607, 0.8290])
            let t1Logsoftmax0 = t1.logsoftmax(0)
            let t1Logsoftmax0Correct = combo.tensor([-0.1980, -2.8664, -2.0981])

            let t2 = combo.tensor([[1.3335, 1.6616, 2.4874, 6.1722],
                                    [3.3478, 9.3019, 1.0844, 8.9874],
                                    [8.6300, 1.8842, 9.1387, 9.1321]])
            let t2Logsoftmax0 = t2.logsoftmax(0)
            let t2Logsoftmax0Correct = combo.tensor([[-7.3022e+00, -7.6414e+00, -6.6529e+00, -3.6107e+00],
                                                        [-5.2879e+00, -1.0806e-03, -8.0559e+00, -7.9552e-01],
                                                        [-5.7426e-03, -7.4188e+00, -1.6088e-03, -6.5082e-01]])
            let t2Logsoftmax1 = t2.logsoftmax(1)
            let t2Logsoftmax1Correct = combo.tensor([[-4.8818, -4.5537, -3.7279, -0.0431],
                                                        [-6.5040, -0.5499, -8.7674, -0.8644],
                                                        [-1.4624, -8.2082, -0.9537, -0.9603]])

            let t3 = combo.tensor([[[3.0897, 2.0902],
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
            let t3Logsoftmax0Correct = combo.tensor([[[-6.0050e+00, -5.5864e+00],
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
            let t3Logsoftmax1Correct = combo.tensor([[[-1.7120e+00, -6.6966e+00],
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
            let t3Logsoftmax2Correct = combo.tensor([[[-3.1340e-01, -1.3129e+00],
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
    member _.TestTensorLogsumexp () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([2.7291, 0.0607, 0.8290])
            let t1Logsumexp0 = t1.logsumexp(0)
            let t1Logsumexp0Correct = combo.tensor(2.9271)
            let t1Logsumexp0keepdim = t1.logsumexp(0, keepDim=true)
            let t1Logsumexp0keepdimCorrect = combo.tensor([2.9271])

            let t2 = combo.tensor([[1.3335, 1.6616, 2.4874, 6.1722],
                                    [3.3478, 9.3019, 1.0844, 8.9874],
                                    [8.6300, 1.8842, 9.1387, 9.1321]])
            let t2Logsumexp0 = t2.logsumexp(0)
            let t2Logsumexp0Correct = combo.tensor([8.6357, 9.3030, 9.1403, 9.7829])
            let t2Logsumexp0keepdim = t2.logsumexp(0, keepDim=true)
            let t2Logsumexp0keepdimCorrect = combo.tensor([[8.6357, 9.3030, 9.1403, 9.7829]])
            let t2Logsumexp1 = t2.logsumexp(1)
            let t2Logsumexp1Correct = combo.tensor([ 6.2153,  9.8518, 10.0924])
            let t2Logsumexp1keepdim = t2.logsumexp(1, keepDim=true)
            let t2Logsumexp1keepdimCorrect = combo.tensor([[ 6.2153],
                                                            [ 9.8518],
                                                            [10.0924]])

            let t3 = combo.tensor([[[3.0897, 2.0902],
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
            let t3Logsumexp0Correct = combo.tensor([[9.0947, 7.6766],
                                                        [8.1668, 9.9756],
                                                        [3.1949, 9.4756],
                                                        [9.3685, 3.7257]])
            let t3Logsumexp0keepdim = t3.logsumexp(0, keepDim=true)
            let t3Logsumexp0keepdimCorrect = combo.tensor([[[9.0947, 7.6766],
                                                             [8.1668, 9.9756],
                                                             [3.1949, 9.4756],
                                                             [9.3685, 3.7257]]])                                                    
            let t3Logsumexp1 = t3.logsumexp(1)
            let t3Logsumexp1Correct = combo.tensor([[ 4.8017,  8.7868],
                                                        [ 8.6526, 10.0003],
                                                        [ 9.8158,  9.0061]])
            let t3Logsumexp1keepdim = t3.logsumexp(1, keepDim=true)
            let t3Logsumexp1keepdimCorrect = combo.tensor([[[ 4.8017,  8.7868]],

                                                            [[ 8.6526, 10.0003]],

                                                            [[ 9.8158,  9.0061]]])
            let t3Logsumexp2 = t3.logsumexp(2)
            let t3Logsumexp2Correct = combo.tensor([[3.4031, 2.6778, 8.7815, 4.7154],
                                                        [8.7999, 9.9565, 2.9858, 2.8058],
                                                        [8.3898, 8.2726, 8.7842, 9.3611]])
            let t3Logsumexp2keepdim = t3.logsumexp(2, keepDim=true)
            let t3Logsumexp2keepdimCorrect = combo.tensor([[[3.4031],
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

            let t4 = combo.tensor([[167.385696, -146.549866, 168.850235, -41.856903, -56.691696, -78.774994, 42.035625, 97.490936, -42.763878, -2.130855], 
                                     [-62.961613, -497.529846, 371.218231, -30.224543, 368.146393, -325.945068, -292.102631, -24.760872, 130.348282, -193.775909]])
            let t4Logsumexp1 = t4.logsumexp(dim=1)
            let t4Logsumexp1Correct = combo.tensor([169.0582, 371.2635])
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
    member _.TestTensorNllLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([[0.15,0.85],[0.5,0.5],[0.8,0.2]]).log()
            let t1b = combo.tensor([0,1,1])
            let t1w = combo.tensor([-1.2,0.6])
            let l1 = dsharp.nllLoss(t1a, t1b)
            let l1Correct = combo.tensor(1.3999)
            // Note, test disabled - this is not the correct answer, even on the backend
            // it was coming out as -Infinity
            //let l2 = dsharp.nllLoss(t1a, t1b, weight=t1w)
            //let l2Correct = combo.tensor(-0.8950)
            let l3 = dsharp.nllLoss(t1a, t1b, reduction="none")
            let l3Correct = combo.tensor([1.8971, 0.6931, 1.6094])
            let l4 = dsharp.nllLoss(t1a, t1b, reduction="none", weight=t1w)
            let l4Correct = combo.tensor([-2.2765,  0.4159,  0.9657])
            let l5 = dsharp.nllLoss(t1a, t1b, reduction="sum")
            let l5Correct = combo.tensor(4.1997)
            let l6 = dsharp.nllLoss(t1a, t1b, reduction="sum", weight=t1w)
            let l6Correct = combo.tensor(-0.8950)

            let t2a = combo.tensor([[[[-1.9318, -1.9386, -0.9488, -0.8787],
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
            let t2b = combo.tensor([[[2, 0, 1, 2],
                                         [2, 0, 1, 0],
                                         [2, 1, 0, 1],
                                         [1, 2, 1, 1]],

                                        [[2, 0, 2, 0],
                                         [0, 1, 0, 2],
                                         [2, 0, 2, 1],
                                         [1, 1, 1, 2]]])
            let t2w = combo.tensor([ 1.1983, -0.2633, -0.3064])
            let l7 = dsharp.nllLoss(t2a, t2b)
            let l7Correct = combo.tensor(1.3095)
            let l8 = dsharp.nllLoss(t2a, t2b, weight=t2w)
            let l8Correct = combo.tensor(2.4610)
            let l9 = dsharp.nllLoss(t2a, t2b, reduction="none")
            let l9Correct = combo.tensor([[[1.2868, 1.9386, 1.2375, 1.8975],
                                             [0.5013, 2.4614, 0.9717, 1.1577],
                                             [1.2271, 1.3655, 0.8123, 1.0334],
                                             [0.8829, 0.4410, 1.5420, 1.9021]],

                                            [[1.2316, 2.4012, 1.2460, 1.4381],
                                             [1.5336, 0.5392, 2.1201, 1.5724],
                                             [0.8335, 1.2666, 1.9886, 0.5593],
                                             [0.6594, 0.9271, 1.0346, 1.8940]]])
            let l10 = dsharp.nllLoss(t2a, t2b, reduction="none", weight=t2w)
            let l10Correct = combo.tensor([[[-0.3943,  2.3231, -0.3258, -0.5814],
                                             [-0.1536,  2.9496, -0.2558,  1.3872],
                                             [-0.3760, -0.3595,  0.9734, -0.2721],
                                             [-0.2324, -0.1351, -0.4059, -0.5007]],

                                            [[-0.3774,  2.8775, -0.3818,  1.7233],
                                             [ 1.8378, -0.1419,  2.5406, -0.4818],
                                             [-0.2554,  1.5179, -0.6093, -0.1472],
                                             [-0.1736, -0.2440, -0.2724, -0.5804]]])
            let l11 = dsharp.nllLoss(t2a, t2b, reduction="sum")
            let l11Correct = combo.tensor(41.9042)
            let l12 = dsharp.nllLoss(t2a, t2b, reduction="sum", weight=t2w)
            let l12Correct = combo.tensor(10.4726)

            Assert.True(l1Correct.allclose(l1, 0.001))
            //Assert.True(l2Correct.allclose(l2, 0.001))
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
    member _.TestTensorCrossEntropyLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([[-0.6596,  0.3078, -0.2525, -0.2593, -0.2354],
                                        [ 0.4708,  0.6073,  1.5621, -1.4636,  0.9769],
                                        [ 0.5078,  0.0579,  1.0054,  0.3532,  1.1819],
                                        [ 1.5425, -0.2887,  1.0716, -1.3946,  0.8806]])
            let t1b = combo.tensor([3, 1, 0, 4])
            let t1w = combo.tensor([-1.4905,  0.5929,  1.0018, -1.0858, -0.5993])
            let l1 = dsharp.crossEntropyLoss(t1a, t1b)
            let l1Correct = combo.tensor(1.7059)
            let l2 = dsharp.crossEntropyLoss(t1a, t1b, weight=t1w)
            let l2Correct = combo.tensor(1.6969)
            let l3 = dsharp.crossEntropyLoss(t1a, t1b, reduction="none")
            let l3Correct = combo.tensor([1.6983, 1.7991, 1.8085, 1.5178])
            let l4 = dsharp.crossEntropyLoss(t1a, t1b, reduction="none", weight=t1w)
            let l4Correct = combo.tensor([-1.8439,  1.0666, -2.6956, -0.9096])
            let l5 = dsharp.crossEntropyLoss(t1a, t1b, reduction="sum")
            let l5Correct = combo.tensor(6.8237)
            let l6 = dsharp.crossEntropyLoss(t1a, t1b, reduction="sum", weight=t1w)
            let l6Correct = combo.tensor(-4.3825)

            Assert.True(l1Correct.allclose(l1, 0.001))
            Assert.True(l2Correct.allclose(l2, 0.001))
            Assert.True(l3Correct.allclose(l3, 0.001))
            Assert.True(l4Correct.allclose(l4, 0.001))
            Assert.True(l5Correct.allclose(l5, 0.001))
            Assert.True(l6Correct.allclose(l6, 0.001))

    [<Test>]
    member _.TestTensorMseLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([-0.2425,  0.2643,  0.7070,  1.2049,  1.6245])
            let t1b = combo.tensor([-1.0742,  1.5874,  0.6509,  0.8715,  0.0692])
            let l1 = dsharp.mseLoss(t1a, t1b)
            let l1Correct = combo.tensor(0.9951)
            let l2 = dsharp.mseLoss(t1a, t1b, reduction="none")
            let l2Correct = combo.tensor([0.6917, 1.7507, 0.0031, 0.1112, 2.4190])
            let l3 = dsharp.mseLoss(t1a, t1b, reduction="sum")
            let l3Correct = combo.tensor(4.9756)

            let t2a = combo.tensor([[ 0.6650,  0.5049, -0.7356,  0.5312, -0.6574],
                                     [ 1.0133,  0.9106,  0.1523,  0.2662,  1.1438],
                                     [ 0.3641, -1.8525, -0.0822, -1.0361,  0.2723]])
            let t2b = combo.tensor([[-1.0001, -1.4867, -0.3340, -0.2590,  0.1395],
                                     [-2.0158,  0.8281,  1.1726, -0.2359,  0.5007],
                                     [ 1.3242,  0.5215,  1.4293, -1.4235,  0.2473]])
            let l4 = dsharp.mseLoss(t2a, t2b)
            let l4Correct = combo.tensor(1.8694)
            let l5 = dsharp.mseLoss(t2a, t2b, reduction="none")
            let l5Correct = combo.tensor([[2.7726e+00, 3.9663e+00, 1.6130e-01, 6.2438e-01, 6.3511e-01],
                                            [9.1753e+00, 6.8075e-03, 1.0409e+00, 2.5207e-01, 4.1352e-01],
                                            [9.2194e-01, 5.6358e+00, 2.2848e+00, 1.5011e-01, 6.2556e-04]])
            let l6 = dsharp.mseLoss(t2a, t2b, reduction="sum")
            let l6Correct = combo.tensor(28.0416)

            Assert.True(l1Correct.allclose(l1, 0.01, 0.01))
            Assert.True(l2Correct.allclose(l2, 0.01, 0.01))
            Assert.True(l3Correct.allclose(l3, 0.01, 0.01))
            Assert.True(l4Correct.allclose(l4, 0.01, 0.01))
            Assert.True(l5Correct.allclose(l5, 0.01, 0.01))
            Assert.True(l6Correct.allclose(l6, 0.01, 0.01))

    [<Test>]
    member _.TestTensorBceLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([[0.6732, 0.3984, 0.1378, 0.4564, 0.0396],
                                    [0.7311, 0.6797, 0.8294, 0.8716, 0.5781],
                                    [0.6032, 0.0346, 0.3714, 0.7304, 0.0434]])
            let t1b = combo.tensor([[0.1272, 0.8250, 0.5473, 0.2635, 0.2387],
                                    [0.9150, 0.9273, 0.3127, 0.7458, 0.5805],
                                    [0.2771, 0.3095, 0.8710, 0.0176, 0.7242]])
            let t1w = combo.tensor([0.9270, 0.4912, 0.7324])
            let l1 = dsharp.bceLoss(t1a, t1b)
            let l1Correct = combo.tensor(0.9516)
            let l2 = dsharp.bceLoss(t1a, t1b, reduction="none")
            let l2Correct = combo.tensor([[1.0264, 0.8481, 1.1520, 0.6556, 0.8016],
                                            [0.3982, 0.4408, 1.2739, 0.6242, 0.6801],
                                            [0.8083, 1.0655, 0.9226, 1.2933, 2.2837]])
            let l3 = dsharp.bceLoss(t1a, t1b, reduction="sum")
            let l3Correct = combo.tensor(14.2745)
            let l4 = dsharp.bceLoss(t1a, t1b, weight=t1w)
            let l4Correct = combo.tensor(0.7002)
            let l5 = dsharp.bceLoss(t1a, t1b, reduction="none", weight=t1w)
            let l5Correct = combo.tensor([[0.9515, 0.7862, 1.0679, 0.6078, 0.7431],
                                            [0.1956, 0.2165, 0.6258, 0.3066, 0.3341],
                                            [0.5920, 0.7804, 0.6757, 0.9472, 1.6726]])
            let l6 = dsharp.bceLoss(t1a, t1b, reduction="sum", weight=t1w)
            let l6Correct = combo.tensor(10.5032)

            Assert.True(l1Correct.allclose(l1, 0.01, 0.01))
            Assert.True(l2Correct.allclose(l2, 0.01, 0.01))
            Assert.True(l3Correct.allclose(l3, 0.01, 0.01))
            Assert.True(l4Correct.allclose(l4, 0.01, 0.01))
            Assert.True(l5Correct.allclose(l5, 0.01, 0.01))
            Assert.True(l6Correct.allclose(l6, 0.01, 0.01))

    [<Test>]
    member _.TestTensorNormalize () =
        for combo in Combos.FloatingPoint do
            let t0 = combo.tensor(0.5)
            let t0n = t0.normalize()
            let t0nCorrect = combo.tensor(0.)

            let t1 = combo.tensor([-2,-2])
            let t1n = t1.normalize()
            let t1nCorrect = combo.tensor([0.,0.])

            let t2 = combo.tensor([[-2.,-1.,0.,1.,2.,3.],[0.5, 0.7, -5.2, 2.3, 1., 2.]])
            let t2n = t2.normalize()
            let t2nCorrect = combo.tensor([[0.3902, 0.5122, 0.6341, 0.7561, 0.8780, 1.0000],
                                            [0.6951, 0.7195, 0.0000, 0.9146, 0.7561, 0.8780]])

            Assert.True(t0nCorrect.allclose(t0n, 0.01, 0.01))
            Assert.True(t1nCorrect.allclose(t1n, 0.01, 0.01))
            Assert.True(t2nCorrect.allclose(t2n, 0.01, 0.01))

    [<Test>]
    member _.TestTensorStandardize () =
        for combo in Combos.FloatingPoint do
            let t0 = combo.tensor(0.5)
            let t0s = t0.standardize()
            let t0sCorrect = combo.tensor(0.)

            let t1 = combo.tensor([-2,-2])
            let t1s = t1.standardize()
            let t1sCorrect = combo.tensor([0.,0.])

            let t2 = combo.tensor([[-2.,-1.,0.,1.,2.,3.],[0.5, 0.7, -5.2, 2.3, 1., 2.]])
            let t2s = t2.standardize()
            let t2sCorrect = combo.tensor([[-1.0496, -0.6046, -0.1595,  0.2856,  0.7307,  1.1757],
                                            [ 0.0631,  0.1521, -2.4739,  0.8642,  0.2856,  0.7307]])

            Assert.True(t0sCorrect.allclose(t0s, 0.01, 0.01))
            Assert.True(t1sCorrect.allclose(t1s, 0.01, 0.01))
            Assert.True(t2sCorrect.allclose(t2s, 0.01, 0.01))

    [<Test>]
    member _.TestTensorSaveImageLoadImage () =
        let fileName = System.IO.Path.GetTempFileName() + ".png"
        let t0 = dsharp.rand([3; 16; 16])
        t0.saveImage(fileName)
        let t1 = dsharp.loadImage(fileName)

        Assert.True(t0.allclose(t1, 0.01, 0.01))

    [<Test>]
    member _.TestTensorDepth () =
        for combo in Combos.All do 
            let t0 = combo.tensor([1.;2.])
            let t0Depth = t0.depth
            let t0DepthCorrect = 0
            let t1 = combo.tensor([1.;2.]).reverseDiff()
            let t1Depth = t1.depth
            let t1DepthCorrect = 1
            let t2 = combo.tensor([1.;2.]).reverseDiff().reverseDiff()
            let t2Depth = t2.depth
            let t2DepthCorrect = 2
            let t3 = combo.tensor([1.;2.]).reverseDiff().reverseDiff().forwardDiff(combo.tensor([1.; 1.]))
            let t3Depth = t3.depth
            let t3DepthCorrect = 3

            Assert.CheckEqual(t0DepthCorrect, t0Depth)
            Assert.CheckEqual(t1DepthCorrect, t1Depth)
            Assert.CheckEqual(t2DepthCorrect, t2Depth)
            Assert.CheckEqual(t3DepthCorrect, t3Depth)

    [<Test>]
    member _.TestTensorIEnumerable () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1,2,3])
            t1.unstack() |> Seq.iteri (fun i v -> Assert.CheckEqual(t1.[i], v))
            let t2 = combo.tensor([[1,2,3], [4,5,6]])
            t2.unstack() |> Seq.iteri (fun i v -> Assert.CheckEqual(t2.[i], v))

    [<Test>]
    member _.TestTensorFSharpCoreOps () =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([0.1; 0.2; 0.3])
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
        
            Assert.CheckEqual(addCorrect, add)
            Assert.CheckEqual(subCorrect, sub)
            Assert.CheckEqual(mulCorrect, mul)
            Assert.CheckEqual(divCorrect, div)
            Assert.CheckEqual(powCorrect, pow)
            Assert.CheckEqual(negCorrect, neg)
            Assert.CheckEqual(floorCorrect, floor)
            Assert.CheckEqual(ceilCorrect, ceil)
            Assert.CheckEqual(roundCorrect, round)
            Assert.CheckEqual(absCorrect, abs)
            Assert.CheckEqual(expCorrect, exp)
            Assert.CheckEqual(logCorrect, log)
            Assert.CheckEqual(log10Correct, log10)
            Assert.CheckEqual(sqrtCorrect, sqrt)
            Assert.CheckEqual(sinCorrect, sin)
            Assert.CheckEqual(cosCorrect, cos)
            Assert.CheckEqual(tanCorrect, tan)
            Assert.CheckEqual(sinhCorrect, sinh)
            Assert.CheckEqual(coshCorrect, cosh)
            Assert.CheckEqual(tanhCorrect, tanh)
            Assert.CheckEqual(asinCorrect, asin)
            Assert.CheckEqual(acosCorrect, acos)
            Assert.CheckEqual(atanCorrect, atan)

            Assert.CheckEqual(combo.dtype, add.dtype)
            Assert.CheckEqual(combo.dtype, sub.dtype)
            Assert.CheckEqual(combo.dtype, mul.dtype)
            Assert.CheckEqual(combo.dtype, div.dtype)
            Assert.CheckEqual(combo.dtype, pow.dtype)
            Assert.CheckEqual(combo.dtype, neg.dtype)
            Assert.CheckEqual(combo.dtype, floor.dtype)
            Assert.CheckEqual(combo.dtype, ceil.dtype)
            Assert.CheckEqual(combo.dtype, round.dtype)
            Assert.CheckEqual(combo.dtype, abs.dtype)
            Assert.CheckEqual(combo.dtype, exp.dtype)
            Assert.CheckEqual(combo.dtype, log.dtype)
            Assert.CheckEqual(combo.dtype, log10.dtype)
            Assert.CheckEqual(combo.dtype, sqrt.dtype)
            Assert.CheckEqual(combo.dtype, sin.dtype)
            Assert.CheckEqual(combo.dtype, cos.dtype)
            Assert.CheckEqual(combo.dtype, tan.dtype)
            Assert.CheckEqual(combo.dtype, sinh.dtype)
            Assert.CheckEqual(combo.dtype, cosh.dtype)
            Assert.CheckEqual(combo.dtype, tanh.dtype)
            Assert.CheckEqual(combo.dtype, asin.dtype)
            Assert.CheckEqual(combo.dtype, acos.dtype)
            Assert.CheckEqual(combo.dtype, atan.dtype)

    [<Test>]
    member _.TestTensorConvTranspose1D () =
        for combo in Combos.FloatingPoint do
            let t1 = combo.tensor([[[-1.2531,  0.9667,  0.2120, -1.2948,  0.4470,  1.3539],
                                    [-0.3736,  0.8294, -0.8978,  0.1512, -1.9213, -0.0488],
                                    [-0.6830,  0.0080, -0.1773, -1.7092, -0.0818, -0.2670]]])
            let t2 = combo.tensor([[[ 0.1036,  0.4791, -1.3667],
                                    [ 1.8627, -1.0295, -0.9342]],
                           
                                   [[-0.1559,  0.4204, -1.0169],
                                    [ 1.0772,  0.9606,  0.4394]],
                           
                                   [[-0.0849,  0.5367, -1.4039],
                                    [-0.1863,  0.8559,  0.1834]]])

            let t3 = dsharp.convTranspose1d(t1, t2)
            let t3Correct = combo.tensor([[[-0.0135, -1.1538,  4.0443, -2.5593, -0.2493,  3.5484,  1.9425,
                                            -1.4259],
                                           [-2.6092,  3.0392,  0.1504, -3.7002, -1.8314,  1.1058, -2.9461,
                                            -1.3352]]])

            let t3p1 = dsharp.convTranspose1d(t1, t2, padding=1)
            let t3p1Correct = combo.tensor([[[-1.1538,  4.0443, -2.5593, -0.2493,  3.5484,  1.9425],
                                              [ 3.0392,  0.1504, -3.7002, -1.8314,  1.1058, -2.9461]]])

            let t3p2 = dsharp.convTranspose1d(t1, t2, padding=2)
            let t3p2Correct = combo.tensor([[[ 4.0443, -2.5593, -0.2493,  3.5484],
                                             [ 0.1504, -3.7002, -1.8314,  1.1058]]])

            let t3s2 = dsharp.convTranspose1d(t1, t2, stride=2)
            let t3s2Correct = combo.tensor([[[-0.0135, -1.1240,  3.0214,  0.8161, -1.9989, -0.3710,  0.8596,
                                              -1.4742,  4.3680, -0.6374,  1.6282,  0.4848, -1.4259],
                                             [-2.6092,  0.3466,  3.5738, -0.1917, -1.0763, -1.2325, -2.5556,
                                               0.0154, -0.2591, -2.3758,  1.2422, -1.6693, -1.3352]]])

            let t3s3 = dsharp.convTranspose1d(t1, t2, stride=3)
            let t3s3Correct = combo.tensor([[[-0.0135, -1.1240,  3.0512, -0.0298,  0.8161, -2.1758,  0.1770,
                                              -0.3710,  0.8721, -0.0125, -1.4742,  4.0153,  0.3527, -0.6374,
                                               1.4576,  0.1705,  0.4848, -1.4259],
                                             [-2.6092,  0.3466,  0.8812,  2.6926, -0.1917, -0.5372, -0.5391,
                                              -1.2325, -0.6251, -1.9305,  0.0154,  0.9626, -1.2217, -2.3758,
                                              -1.2768,  2.5191, -1.6693, -1.3352]]])

            let t3s2p1 = dsharp.convTranspose1d(t1, t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[-1.1240,  3.0214,  0.8161, -1.9989, -0.3710,  0.8596, -1.4742,
                                                 4.3680, -0.6374,  1.6282,  0.4848],
                                               [ 0.3466,  3.5738, -0.1917, -1.0763, -1.2325, -2.5556,  0.0154,
                                                 -0.2591, -2.3758,  1.2422, -1.6693]]])

            let t3s3p2 = dsharp.convTranspose1d(t1, t2, stride=3, padding=2)
            let t3s3p2Correct = combo.tensor([[[ 3.0512, -0.0298,  0.8161, -2.1758,  0.1770, -0.3710,  0.8721,
                                                  -0.0125, -1.4742,  4.0153,  0.3527, -0.6374,  1.4576,  0.1705],
                                                 [ 0.8812,  2.6926, -0.1917, -0.5372, -0.5391, -1.2325, -0.6251,
                                                   -1.9305,  0.0154,  0.9626, -1.2217, -2.3758, -1.2768,  2.5191]]])

            let t3d2 = dsharp.convTranspose1d(t1, t2, dilation=2)
            let t3d2Correct = combo.tensor([[[-0.0135, -0.0298, -0.9470,  0.8036,  3.0329, -3.4795,  0.2347,
                                                 4.5001,  1.4576, -1.4259],
                                               [-2.6092,  2.6926, -0.1925, -2.1222, -1.5730,  1.9973, -3.0009,
                                                -0.7067, -1.2768, -1.3352]]])

            let t3p2d3 = dsharp.convTranspose1d(t1, t2, padding=2, dilation=3)
            let t3p2d3Correct = combo.tensor([[[ 0.1770, -1.1365,  1.1688, -0.2005,  1.5770, -2.8133,  1.3570,
                                                     4.0153],
                                                   [-0.5391, -1.5840, -1.4133,  1.2866,  0.8965, -2.9130, -2.2944,
                                                     0.9626]]])

            let t3s3p6d3 = dsharp.convTranspose1d(t1, t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = combo.tensor([[[ 4.0443,  0.0000,  0.0000, -2.5593,  0.0000,  0.0000, -0.2493,
                                                   0.0000,  0.0000,  3.5484],
                                                 [ 0.1504,  0.0000,  0.0000, -3.7002,  0.0000,  0.0000, -1.8314,
                                                   0.0000,  0.0000,  1.1058]]])

            Assert.True(t3Correct.allclose(t3, 0.01))
            Assert.True(t3p1Correct.allclose(t3p1, 0.01))
            Assert.True(t3p2Correct.allclose(t3p2, 0.01))
            Assert.True(t3s2Correct.allclose(t3s2, 0.01))
            Assert.True(t3s3Correct.allclose(t3s3, 0.01))
            Assert.True(t3s2p1Correct.allclose(t3s2p1, 0.01))
            Assert.True(t3s3p2Correct.allclose(t3s3p2, 0.01))
            Assert.True(t3d2Correct.allclose(t3d2, 0.01))
            Assert.True(t3p2d3Correct.allclose(t3p2d3, 0.01))
            Assert.True(t3s3p6d3Correct.allclose(t3s3p6d3, 0.01, 0.01))
            printfn "done"

    [<Test>]
    member _.TestTensorConvTranspose2D () =
        for combo in Combos.FloatingPoint do
            let t1 = combo.tensor([[[[-2.0280, -7.4258, -1.1627, -3.6714],
                                       [ 3.1646, -2.0775,  1.1166, -3.1054],
                                       [-2.9795,  6.3719,  6.7753, -0.2423],
                                       [-5.1595, -1.5602, -1.5165, -4.1525]],
                             
                                      [[-4.4974, -1.6737,  0.2967, -1.3116],
                                       [ 3.7593, -1.4428, -2.1954, -3.8098],
                                       [-0.2220,  4.3347,  2.6288,  4.9739],
                                       [-2.8094, -3.4588, -1.3126, -2.8789]],
                             
                                      [[ 1.8656,  3.6751,  3.6202,  0.7065],
                                       [ 2.9986, -2.5643, -3.2444, -0.0339],
                                       [-1.0250,  3.4748, -0.9057,  0.6292],
                                       [ 0.1423,  2.9450,  4.5264, -1.4891]]],
                             
                             
                                     [[[-0.5852, -1.6015, -0.2604,  6.8539],
                                       [-1.6572,  0.3233,  2.4716,  0.8160],
                                       [-7.9254,  0.5539, -0.4043,  0.7395],
                                       [ 2.3128,  1.5731,  2.1585,  0.2829]],
                             
                                      [[ 2.0864, -4.2912,  0.8241,  3.3248],
                                       [ 2.4391,  5.8813,  1.0969, -0.4856],
                                       [ 2.2431, -3.8626, -0.0758,  0.7386],
                                       [-1.3231,  2.5438, -3.1992,  2.7404]],
                             
                                      [[ 2.1057,  2.1381,  4.3754, -4.7032],
                                       [-0.0310,  1.5864, -4.6051, -3.2207],
                                       [-8.3767,  1.9677, -2.5842,  0.6181],
                                       [-5.3311,  3.3852, -0.9679, 10.0806]]]])
            let t2 = combo.tensor([[[[-0.6207,  0.9829,  1.9519],
                                       [-1.3195, -1.0054, -0.0422],
                                       [-0.7566, -0.5450,  0.0660]],
                             
                                      [[ 1.2635, -0.5134, -1.5355],
                                       [ 0.0294, -0.7468,  1.5766],
                                       [-0.6810,  0.0306,  0.7619]]],
                             
                             
                                     [[[ 1.8189,  0.0156,  1.2304],
                                       [-0.6246, -0.5269, -0.6632],
                                       [ 1.0706,  0.0366,  0.4163]],
                             
                                      [[ 1.1352,  0.1125, -1.1641],
                                       [-0.4009,  0.2187,  0.6077],
                                       [ 0.0796, -1.0126, -0.2706]]],
                             
                             
                                     [[[-0.1754,  0.1714, -0.4221],
                                       [ 0.3765, -2.9587,  1.4150],
                                       [ 0.3446, -0.8976,  2.2664]],
                             
                                      [[-0.4247, -0.3800,  1.0981],
                                       [-1.4746,  0.9296,  0.3400],
                                       [ 0.1843,  1.0527,  0.3531]]]])

            let t3 = dsharp.convTranspose2d(t1, t2)
            let t3Correct = combo.tensor([[[[ -7.2488,  -0.8234, -16.3482, -18.8535,  -6.9405,  -9.0783],
                                               [ 10.5350,  13.9143,   8.8096,  -6.3821,   4.9971,  -8.7103],
                                               [ -6.4071,  -6.7860,  18.0874,  51.6045,  28.4071,   8.8040],
                                               [  4.4154, -19.7267, -40.3442, -24.0385, -25.0267, -15.2847],
                                               [ 10.2796,  15.1273, -10.1839,  15.1099,  16.6752,   3.4580],
                                               [  0.9450,   1.0737,  -1.8054,   1.3937,  13.1064,  -4.8475]],
                                     
                                              [[ -8.4601, -13.0171,   9.9559,  10.2132,   6.8841,   7.9401],
                                               [  5.9849,  -8.2165,  -5.9590, -13.5684,   1.1469,   2.8209],
                                               [ -8.0502,  30.9921,  31.5255, -14.5540, -10.9641, -14.1428],
                                               [ -9.5590, -10.0463,  -0.3495,   4.8396,  29.0524,   9.5997],
                                               [  2.5872,  -3.9301, -20.5199,   6.8982,  -0.8133, -10.1110],
                                               [  3.3162,   4.1667,   5.1967,   9.1590,   2.0185,  -2.9105]]],
                                     
                                     
                                             [[[  3.7889,  -7.3680,   0.1545,  -6.1833,   4.6413,  19.4543],
                                               [  5.7324,   7.4798,   2.3715, -12.0400,  19.2886,  -6.7945],
                                               [ 14.5222, -24.0757, -16.1896,  -1.7090,  13.5193, -11.0012],
                                               [  6.8502,  46.0075, -20.2598,  28.5731, -11.2059,  -7.4251],
                                               [  1.2786,  19.7991, -42.3652,  12.9959, -37.7386,  14.1919],
                                               [ -5.0035,   6.1760, -21.6751,  14.6040, -12.4852,  24.0061]],
                                     
                                              [[  0.7348,  -8.0679,  -0.9427,  22.7978,   2.8876, -19.5593],
                                               [ -3.2706,   8.9194,   2.4042,   2.6916, -16.5644,   7.0029],
                                               [ -3.9380,   1.2676,  14.5253,  11.3920, -10.3566,   1.2414],
                                               [ 16.2215,  -0.6001, -28.4006, -15.5361,  -8.6123,   8.8859],
                                               [ 12.4917, -24.5919,   2.5210, -14.8144,   9.6141,   6.1206],
                                               [ -2.6629,  -4.4465,  -0.6293,   5.8754,  10.0140,   3.0334]]]])

            let t1p1 = combo.tensor([[[[ 1.0744,  7.9558,  0.4934,  2.3298,  2.3925, -1.2102],
                                       [-2.1089,  4.0487,  0.9586,  4.5810,  1.0251,  5.6604],
                                       [-3.4522, -4.8244,  0.5531, -6.3983, -5.8461,  3.7263],
                                       [ 7.5891,  4.5920,  1.9801, -5.1166, -3.8933,  2.1153],
                                       [ 0.6262,  2.5322, -6.0730, -3.4204,  2.3583,  0.4224],
                                       [ 0.6814, -0.9715, -1.2208,  9.5117, -1.2802,  2.0687]],
                             
                                      [[ 3.3219, -0.4099, -0.3930, -1.8511, -2.0642, -1.9206],
                                       [ 2.6994,  1.6932,  1.3649,  3.2362,  2.3385, -0.2216],
                                       [-4.3740, -8.2227, -2.9300, -8.7945, -2.0426, -1.1449],
                                       [ 3.6044, -0.5552,  0.0607,  3.7366,  0.1317,  0.3760],
                                       [ 0.7646, -3.2802, -0.7214, -5.0273,  0.0336, -3.9015],
                                       [-1.3125,  1.8098, -1.9835,  7.9206, -0.8541,  3.2770]],
                             
                                      [[ 3.0539, -3.7408,  1.0175, -3.9080, -1.6320, -0.7949],
                                       [ 0.6580,  3.8309,  5.3238, -6.3294,  5.0700,  4.4601],
                                       [ 4.7103, -1.8846,  3.8602, -3.9283,  4.4575,  1.5436],
                                       [-2.9477,  4.4539,  0.6466,  3.8747, -1.8396,  0.4202],
                                       [ 2.0424,  4.7229, -2.0569, -0.7198, -7.7648,  3.7662],
                                       [ 6.3408, -1.8474, -2.4028, -1.1776,  6.5768, -2.5502]]],
                             
                             
                                     [[[ 0.2038, -1.9139, -1.0913,  1.7458,  1.3187, -0.7424],
                                       [-0.6190, -1.4809, -4.1774,  4.1776, -1.6485, -2.8223],
                                       [ 1.3574, -0.9936,  0.4081, -1.2595, -3.1222, -0.1957],
                                       [ 3.2237, -3.5044, -2.2179,  1.1732,  2.7336, -1.0194],
                                       [ 2.8080, -0.6129,  2.4027, -0.8684, -5.8353,  0.5219],
                                       [-5.1602,  0.4612, -1.8575, -1.8444,  1.2732,  5.0051]],
                             
                                      [[ 0.4338, -0.3004,  3.5642,  0.7867, -0.3105,  0.5667],
                                       [ 0.0962, -0.1167, -1.1296,  1.1743, -0.3805,  0.3942],
                                       [ 3.1247, -0.7838,  7.1643, -3.3606, -2.5899,  0.4827],
                                       [-0.7164, -0.9592, -1.6169,  2.0705,  1.3104,  2.9180],
                                       [ 0.9406,  6.0178,  7.0580, -1.1603, -4.9145, -3.0228],
                                       [-1.2659, -4.5113, -0.4634,  2.0256,  3.4598,  1.6469]],
                             
                                      [[ 6.1612, -7.6000,  1.1598,  2.3335, -6.1723,  5.6237],
                                       [ 3.0543, -5.6086,  2.6119, -0.5712, -0.5620,  3.4211],
                                       [-0.8446, -1.7392, -4.8108, -0.0792, -4.0653,  2.2177],
                                       [ 0.2648, -1.0341, -3.0084,  0.6107,  3.5405,  3.5716],
                                       [ 8.1575, -5.9643, -5.5036, -1.8790, -2.2454, -1.4370],
                                       [-1.7650, -5.9335,  3.4498,  0.8872, -1.0203,  3.9062]]]])

            let t3p1 = dsharp.convTranspose2d(t1p1, t2, padding=1)
            let t3p1Correct = combo.tensor([[[[-2.5539e+01,  9.8793e+00,  2.3522e+00,  1.6893e+01,  1.4417e+01,
                                                 1.2602e+01],
                                               [-3.1444e+01, -3.5893e+01, -6.7083e+01,  5.7391e+00, -7.6345e+01,
                                                -2.8184e+01],
                                               [ 5.7274e+00,  5.1016e+01,  2.4985e+01,  6.2553e+01, -4.1881e+01,
                                                 1.1302e+00],
                                               [-2.0541e+01, -9.0034e+00, -2.3712e+01, -5.8394e-01, -2.7339e+01,
                                                 8.2359e+00],
                                               [-4.5268e+00, -2.3789e+01,  5.4599e+01,  7.0560e+00,  7.2854e+01,
                                                -3.1187e+01],
                                               [-2.5245e+01,  1.9611e+01, -7.4000e-01, -1.5047e+01, -3.2242e+01,
                                                -7.5167e+00]],
                                     
                                              [[ 1.5225e+01, -8.1117e+00,  2.3894e+01,  3.4110e+00, -1.5564e+01,
                                                 -1.6471e+00],
                                               [-2.4827e+01,  2.1827e+00,  2.1729e+01, -1.7261e+01,  2.6620e+01,
                                                 2.1128e+01],
                                               [ 8.8218e+00, -3.1076e+01, -4.0147e+00, -1.8270e+01,  3.3127e+00,
                                                 -9.0832e-01],
                                               [-5.8058e+00,  8.6448e+00,  7.2669e+00,  4.1642e+01,  2.9576e+00,
                                                -2.1493e+01],
                                               [-1.4143e+01,  2.1080e+01,  3.9626e+01, -1.6192e+00, -4.8345e+01,
                                                 1.0363e+01],
                                               [ 7.3484e+00,  1.9200e+01,  6.8525e-01, -2.3770e+01,  1.9260e+01,
                                                 3.4740e+00]]],
                                     
                                     
                                             [[[-1.7204e+01,  2.7923e+01, -2.1302e+01, -1.9810e+01,  3.2535e+01,
                                                -3.0405e+01],
                                               [-1.4934e+01,  7.5868e+01, -4.2455e+01,  9.5382e+00,  1.0415e+01,
                                                -3.3989e+01],
                                               [ 1.1103e+00,  1.2644e+01, -1.0561e+01, -3.6917e+00,  3.7296e+01,
                                                 -8.4717e+00],
                                               [ 1.7178e+01,  3.1835e+01,  1.5056e+01, -9.7515e+00, -2.2679e+01,
                                                 -3.5688e+01],
                                               [-4.6248e+01,  6.6559e+00,  2.6084e+00, -5.3964e+00,  1.5587e+01,
                                                 2.4642e+01],
                                               [ 7.0854e+00,  5.0883e+01, -2.0413e+01, -9.6752e+00, -1.0166e+01,
                                                 -2.7874e+01]],
                                     
                                              [[ 1.6488e+01, -6.7920e+00, -2.6865e+00,  1.7668e+01, -2.3272e+01,
                                                 8.2484e+00],
                                               [ 1.6097e+01, -6.7093e+00, -8.1159e+00, -2.9688e+01,  7.8054e+00,
                                                 9.2428e+00],
                                               [-2.1152e+00, -1.7606e-01,  4.4501e+00,  1.2627e+01, -1.0182e+01,
                                                -6.1416e+00],
                                               [-7.5072e-02,  2.9625e+01, -3.5118e+01, -3.8816e+01, -1.1095e+00,
                                                 2.1909e+01],
                                               [ 1.6438e+01,  1.7336e+01, -9.1775e+00,  1.9114e+01,  1.4552e+01,
                                                 -2.2556e+01],
                                               [ 2.0026e+01, -3.1022e+01, -1.8629e+01,  1.0793e+00, -8.2290e+00,
                                                 1.6719e+00]]]])

            let t1p12 = combo.tensor([[[[ 2.0904e-01, -3.0708e+00, -5.8043e-01,  5.2003e-01,  3.1007e+00,
                                            -3.0689e+00,  1.9686e+00,  2.2593e+00],
                                           [-1.5114e+00, -3.5046e+00, -7.6147e+00, -7.7734e-01, -7.9658e-01,
                                            -2.7403e+00,  3.2388e+00,  4.1308e-01],
                                           [ 7.0679e+00,  2.5667e+00, -3.3968e+00, -2.1805e+00, -4.6535e+00,
                                             -6.6126e+00, -4.5696e+00, -2.7322e+00],
                                           [ 4.8306e+00,  1.1991e+00,  1.6866e-01,  4.3821e-01, -4.4588e-01,
                                             2.7424e+00,  3.8553e+00, -1.8864e-01],
                                           [ 3.8901e-01, -5.3517e+00, -2.3543e+00,  3.5484e+00,  3.9898e-01,
                                             -4.1207e+00, -1.5045e+00,  1.9773e+00],
                                           [ 4.3314e+00, -3.5333e+00,  2.1335e+00,  5.1173e+00,  5.2105e+00,
                                             -5.9196e+00, -2.3715e+00,  8.5792e-02]],
                                 
                                          [[ 1.4584e+00,  1.0401e+00,  4.0129e+00,  1.2725e+00, -4.3258e-01,
                                             -3.3049e-01, -1.2140e+00, -1.6860e+00],
                                           [-1.7470e+00, -1.6925e+00, -7.9839e-02,  5.8790e-01, -1.4510e+00,
                                             4.8597e+00,  4.4617e+00,  3.7802e+00],
                                           [ 2.7816e+00, -1.4593e-01,  7.2832e-01,  1.8055e-01, -2.4145e+00,
                                             -3.6923e+00, -2.9494e+00, -6.4016e+00],
                                           [ 3.0402e+00,  6.5263e-01,  7.9575e+00, -2.5088e+00,  4.5268e+00,
                                             6.6195e+00,  1.6011e+00,  4.3730e+00],
                                           [-1.4767e+00, -2.0553e+00,  1.7944e+00, -6.4128e-02, -3.9420e-01,
                                            -9.2923e-01,  3.8154e+00, -9.5326e-01],
                                           [ 2.3029e+00, -1.4282e+00,  4.1835e+00, -7.0811e-01,  4.0882e+00,
                                             -1.2903e+00, -3.6706e-01, -2.4274e+00]],
                                 
                                          [[ 5.5625e+00, -3.0755e-01, -5.8200e+00,  8.1142e+00, -5.4013e+00,
                                             -3.2303e+00, -5.2555e-01, -7.5444e-01],
                                           [ 8.5872e+00, -1.0552e+01,  1.7941e+00,  4.1905e+00, -7.0491e-02,
                                             6.0357e+00,  8.2003e-01,  1.2992e+00],
                                           [ 2.3029e+00, -7.7644e+00,  5.2392e+00,  3.0534e+00, -1.3255e+00,
                                             1.9722e+00, -8.8349e+00,  1.8596e+00],
                                           [ 4.1077e+00, -6.4727e+00,  5.4707e+00, -8.1994e-01, -2.2840e+00,
                                             -7.3100e+00,  1.6094e+00, -1.8923e-01],
                                           [ 3.6762e+00, -8.5700e+00,  7.6150e+00, -7.6913e+00,  4.0187e+00,
                                             -3.4347e+00,  3.4880e+00,  4.4458e-01],
                                           [-9.3896e-02, -1.3383e+00, -1.4096e-02, -2.6950e+00,  4.1328e+00,
                                            -3.0136e-02, -4.4437e+00,  2.1302e+00]]],
                                 
                                 
                                         [[[-2.3001e+00, -6.1048e-01, -5.0777e-01,  6.5870e+00, -6.9482e-01,
                                             1.8224e+00, -1.5978e+00, -9.7060e-01],
                                           [ 1.0086e+00, -2.5574e+00, -3.5676e+00, -1.6516e+00, -3.0457e-02,
                                             2.0455e+00, -2.9152e+00,  4.7178e-01],
                                           [ 1.8213e+00,  2.9062e+00, -1.6246e+00, -3.0354e+00,  1.9258e+00,
                                             1.2320e+00,  1.7550e-01,  1.5679e+00],
                                           [-3.1161e-01,  2.1187e-01,  7.7038e-01, -8.0618e+00, -3.1787e+00,
                                            -7.8896e-01,  2.8006e+00,  1.1497e+00],
                                           [-4.3055e+00,  2.3031e+00,  6.8383e+00,  3.3530e+00, -2.6364e+00,
                                            -2.0941e+00,  2.3572e-01, -1.9117e-02],
                                           [-5.0467e-01, -4.9318e+00, -1.4161e+00, -4.3488e+00, -2.0141e+00,
                                            -9.1710e-01, -1.2912e+00, -4.6389e-01]],
                                 
                                          [[-1.5890e-02,  2.9213e-01,  2.8771e+00,  3.3473e+00, -1.5947e+00,
                                            -8.5990e-02,  5.4676e-01, -5.4066e-01],
                                           [ 4.7811e-01, -9.0797e-01, -3.5322e+00, -5.5444e+00, -9.3019e-01,
                                             -2.8029e+00, -2.3730e+00,  1.3185e+00],
                                           [ 1.1198e+00,  1.3149e+00,  3.9382e+00,  9.9105e-01,  2.3394e+00,
                                             1.5633e+00,  2.0929e+00, -3.1767e-02],
                                           [ 1.2592e+00,  8.2358e-01, -3.0186e+00, -4.4605e+00, -2.6266e+00,
                                             -7.4811e-01,  1.6034e+00,  5.7280e-01],
                                           [-2.5910e-01,  4.7576e+00,  6.1715e+00,  5.1725e+00,  4.2087e+00,
                                             1.5060e-02,  2.3843e+00, -1.1196e+00],
                                           [ 3.4526e-02, -3.3107e+00, -2.9831e+00, -4.9749e+00,  1.3786e+00,
                                             -2.0894e+00, -3.8216e-04, -4.6755e-01]],
                                 
                                          [[ 1.9114e+00, -2.1377e+00,  1.1277e+00,  1.8681e+00,  7.3343e+00,
                                             -1.7746e+00,  1.5446e+00, -6.5108e-01],
                                           [ 9.0044e-01, -3.1848e+00, -3.0141e+00,  8.5436e+00,  4.0129e-01,
                                             -9.0136e-01,  4.6455e-01, -1.2833e+00],
                                           [-9.6027e-01,  1.5802e+00,  1.1102e+00,  8.0889e-01,  2.0755e+00,
                                             2.4087e-01, -2.8644e+00, -8.1120e-01],
                                           [-2.9667e+00, -1.1450e+00,  2.8817e+00, -7.4703e+00,  4.4933e+00,
                                             1.5010e+00, -1.4258e+00,  3.4844e-01],
                                           [ 6.3914e-02, -6.0574e+00,  3.2300e+00, -6.4394e+00,  7.5388e+00,
                                             -3.8723e+00, -1.0272e+00,  5.6870e-01],
                                           [-1.0461e+00, -4.1427e+00,  1.4182e-01,  5.8372e+00, -3.7351e-01,
                                            -2.1219e+00, -5.3250e-01,  7.3212e-01]]]])

            let t3p12 = dsharp.convTranspose2d(t1p12, t2, paddings=[1;2])
            let t3p12Correct = combo.tensor([[[[-3.3253e+00,  5.4371e+00, -5.9816e+01,  3.2848e+01,  4.3607e+00,
                                                 1.6492e+00],
                                               [ 9.7679e+01,  6.7747e+00, -4.1665e+01,  2.0032e+01, -5.2838e+01,
                                                 -5.1707e+01],
                                               [ 8.8758e+01, -3.6796e+01,  2.3499e+01,  5.4406e+01,  1.2719e+01,
                                                 9.2254e+01],
                                               [ 2.9644e+01, -5.8237e+01,  8.4483e+00,  1.9658e+01, -4.7019e-01,
                                                 -2.9818e+01],
                                               [ 8.3079e+01, -7.1435e+01,  6.8016e+01,  1.2094e+00,  3.2177e+01,
                                                 -5.0162e+01],
                                               [ 2.5591e+01, -4.3952e+01,  1.5923e+01, -3.4699e+01,  3.2012e+01,
                                                 6.0155e+00]],
                                     
                                              [[ 2.1227e+01, -2.3985e+01,  2.5212e+01,  6.6007e+00,  1.4629e+01,
                                                 1.4605e+00],
                                               [-2.6661e+01, -3.7247e+01, -3.5895e+00, -1.5200e+01,  4.8951e+00,
                                                 2.6003e+00],
                                               [-2.9117e+00, -1.1533e+01,  5.0551e+00,  2.0305e+01,  1.1223e+01,
                                                -3.9817e+01],
                                               [ 4.3272e-01,  2.2397e+01,  2.0958e+01, -9.2573e+00,  2.9995e+00,
                                                 -1.3124e-02],
                                               [-1.6896e+01,  1.5415e+01, -1.4137e+01, -2.0068e+01, -3.1009e+01,
                                                 4.9770e+00],
                                               [ 5.7245e+00, -7.4357e+00, -1.5237e+01,  1.4700e+01,  2.3845e+01,
                                                 -2.0412e+01]]],
                                     
                                     
                                             [[[ 4.3162e+00, -3.7015e+01, -2.1557e+01, -4.2872e+01,  1.5128e+01,
                                                 -3.9774e+00],
                                               [ 4.5918e+01,  2.2101e+01, -2.0702e+01,  1.0873e+01,  3.8851e+01,
                                                 -7.5143e-02],
                                               [-1.0218e+01, -2.4655e+00, -3.6693e+01, -2.1096e+01, -1.4586e+01,
                                                 2.4224e+00],
                                               [-7.7054e-02,  3.7449e+01,  8.2727e+01,  2.5300e+00,  4.3413e+00,
                                                -6.6815e-01],
                                               [-1.6899e+01, -6.8931e+01,  2.8213e+01, -7.6618e+01,  2.3916e+01,
                                                -5.7188e+00],
                                               [ 2.7919e+01, -7.0535e+00,  1.7199e+01, -1.2670e+00,  3.3749e+01,
                                                 -6.5397e+00]],
                                     
                                              [[-1.3218e+01, -1.1301e+01, -9.3226e+00,  4.0663e+01, -1.1171e+01,
                                                 8.9378e+00],
                                               [-1.3149e+00, -3.0373e+01,  3.3557e+00,  1.8259e+01, -1.2272e+00,
                                                 7.0654e+00],
                                               [-7.9169e+00, -9.3778e+00,  1.4320e+01,  5.5024e+00,  3.0991e+01,
                                                 1.1212e+01],
                                               [ 2.0909e+01,  9.3709e+00, -2.4690e+01, -3.7275e+01,  1.1494e+01,
                                                 -1.2765e+01],
                                               [-2.6079e+01,  1.4229e+01,  1.4370e+00,  3.9834e+01,  3.5829e-01,
                                                -8.2415e+00],
                                               [-1.8959e+01, -2.4770e+01, -1.8573e-01,  6.9171e-01, -8.2630e+00,
                                                -1.0300e+01]]]])

            let t1s2 = combo.tensor([[[[-4.2302, -2.7939],
                                       [ 4.5052,  3.8188]],
                             
                                      [[ 5.7356,  8.4538],
                                       [ 3.7488,  6.3469]],
                             
                                      [[ 8.4869, 10.8920],
                                       [ 6.1609, -5.2561]]],
                             
                             
                                     [[[ 4.4355, -3.7308],
                                       [-1.7996,  2.1566]],
                             
                                      [[ 4.5993, -2.7351],
                                       [ 4.9944,  1.7658]],
                             
                                      [[-3.0583, -7.1895],
                                       [ 9.4745,  6.8794]]]])
            let t3s2 = dsharp.convTranspose2d(t1s2, t2, stride=2)
            let t3s2Correct = combo.tensor([[[[ 11.5695,  -2.6138,  10.4181,  -0.7474,   0.3506],
                                               [  5.1947, -23.8791,  10.8908, -33.8715,   9.9235],
                                               [ 15.2073,   0.4402,  57.1628,  -4.9930,  45.5023],
                                               [ -5.9665, -24.7330,  -4.9407,   8.3677, -11.8079],
                                               [  2.7278,  -7.8481,  17.9155,   2.8690,  -9.0182]],
                                     
                                              [[ -2.4383,  -0.4080,  10.5790,  -1.7535,   6.4095],
                                               [-14.9385,  12.3029, -19.8310,  14.0606,   4.4357],
                                               [ 12.2329,  -1.2355,  12.5506,   3.5710, -19.5942],
                                               [-10.4553,   3.1825,  16.7942,  -6.3499,   8.0906],
                                               [ -1.6342,   2.8274,   1.5294, -11.8432,  -0.6639]]],
                                     
                                     
                                             [[[  6.1489,   3.9072,  14.2093,  -4.9419,  -7.6127],
                                               [ -9.8767,   2.1657,  -3.6406,  26.4635,  -8.2018],
                                               [  9.0536,   0.4292,  -8.0068,  11.7128, -14.2008],
                                               [  2.8222, -28.8545,   8.8117, -23.4528,   8.4723],
                                               [  9.9734,  -7.3408,  26.0629,  -7.2857,  16.4690]],
                                     
                                              [[ 12.1241,  -0.5976, -20.2883,   4.3397,   1.0178],
                                               [  2.7963,  -5.1495,  20.3365,  -4.4954,  -9.9885],
                                               [ -3.8460,  -9.8555,  11.2138,  -8.4357,  -2.4537],
                                               [-16.0263,  11.2437,  -7.3697,   5.1708,   6.8122],
                                               [  3.3692,   4.8615,   0.5627,   5.5199,   3.5944]]]])

            let t1s13 = combo.tensor([[[[-9.8044, -2.9782],
                                           [-2.7887,  4.5641],
                                           [ 0.5278,  4.7393],
                                           [-4.0212, -5.5322]],
                                 
                                          [[ 0.7842, -1.7191],
                                           [-0.1781, -0.0738],
                                           [ 7.6769, -0.2776],
                                           [-5.3948, -1.7661]],
                                 
                                          [[ 6.1815, -2.2200],
                                           [-9.2024, -5.4785],
                                           [-6.2536,  0.4347],
                                           [-2.3570,  4.6716]]],
                                 
                                 
                                         [[[-3.0220,  2.2930],
                                           [-3.3329,  1.0919],
                                           [ 0.4386, -5.8802],
                                           [-3.3151,  0.9038]],
                                 
                                          [[ 2.5312,  4.7056],
                                           [ 0.3190,  0.0251],
                                           [-2.4100, -0.1728],
                                           [ 1.5978, -2.1062]],
                                 
                                          [[-1.8104, -8.8542],
                                           [-2.7608,  3.7158],
                                           [ 1.1023,  0.6211],
                                           [ 0.3481, -3.1282]]]])
            let t3s13 = dsharp.convTranspose2d(t1s13, t2, strides=[1;3])
            let t3s13Correct = combo.tensor([[[[  6.4277,  -8.5650, -20.7816,  -0.8889,  -3.3346,  -6.9913],
                                               [ 17.7955, -13.1661,   6.8624,   2.1615,  14.0141,   9.2549],
                                               [ 25.4467,  29.5152,  14.0191, -11.9140,  19.9407,  -5.1138],
                                               [-16.0010,  19.2597, -48.5687, -11.9344,  -8.1418, -26.4900],
                                               [ 13.4527,  19.4654, -10.5299,   6.4285, -10.3124,   9.1974],
                                               [ -3.5454,   4.1097,  -7.8530,   3.9047,  -1.2428,   9.4873]],
                                     
                                              [[-14.1229,   2.7728,  20.9297,  -4.7716,   2.1792,   4.1365],
                                               [ -9.5353,  18.1484, -18.4952,  11.8849,  -0.4852, -19.4332],
                                               [ 33.4754,   1.8713, -29.7474,  15.2129, -11.8344,  -3.7762],
                                               [ -3.8556, -11.7677,   7.9117, -15.4929,  -7.8825,  24.6944],
                                               [  4.6192, -14.7085, -14.3029,  -9.5128,   8.9717,  -4.3675],
                                               [  1.8746,   2.8585,  -2.4361,   4.4878,   6.5368,  -2.0876]]],
                                     
                                     
                                             [[[  6.7974,  -3.2412,  -2.0201,   8.6888,   0.8095,  14.0027],
                                               [  4.8581,   3.3171,  -9.0605, -10.5821,  23.1227, -15.1524],
                                               [  2.6824,  15.2982,  -9.8007,   3.4209, -10.9109, -24.7141],
                                               [  8.1563,  -1.3002,  -7.8564,   4.7387,   0.5551,  10.2388],
                                               [  0.9752,   0.1444,   1.0968,   3.4232,  12.0973,  -2.1200],
                                               [  4.3388,   1.5527,   1.2354,  -4.0167,   2.2382,  -7.9069]],
                                     
                                              [[ -0.1760,   2.5242,  -0.2943,  11.9993,   2.7168, -18.7214],
                                               [ -1.1104,   3.9235,  -2.1272,  11.0673, -10.8839,   5.8387],
                                               [  3.1212,  -5.4843,  -6.2836, -16.1657,  -8.6078,  10.2598],
                                               [ -1.3828,  -1.4117,  -0.3860,  -0.9968,   9.3384,  -9.3984],
                                               [ -1.5388,   6.7630,  -2.7617,   9.5889,  -3.3946,  -5.1327],
                                               [  2.4490,  -1.3529,  -2.8353,  -1.3597,  -1.1327,   0.1540]]]])

            let t1s2p1 = combo.tensor([[[[ -3.4577,   3.2779,   2.9547],
                                           [  2.2602,  -3.8841,   1.4702],
                                           [  0.2794,  -2.2958,  -3.5196]],
                                 
                                          [[  0.1823,  -0.9480,  -0.3327],
                                           [  0.7481,  -2.4933,  -3.9782],
                                           [  3.2706,   2.8311,  -4.2914]],
                                 
                                          [[-12.7793,  -1.5203,   8.0372],
                                           [  5.0149,  -9.2994,  -1.8505],
                                           [  6.6970,  -0.4846,   4.1385]]],
                                 
                                 
                                         [[[  1.8252,  -2.0286,   4.0794],
                                           [  0.4706,   2.6324,  -0.3310],
                                           [  0.9786,  -0.9518,  -5.4449]],
                                 
                                          [[  3.1169,   0.4747,  -1.1639],
                                           [ -0.0482,   0.6452,  -1.3964],
                                           [  1.8278,   0.1934,  -2.0665]],
                                 
                                          [[ -7.7843,  -7.3282,   1.5546],
                                           [ -3.3539,  -1.5674,   0.0477],
                                           [  2.6323,   6.4161,   6.6779]]]])
            let t3s2p1 = dsharp.convTranspose2d(t1s2p1, t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[[ 4.1190e+01, -2.2363e+01,  1.7022e+00, -2.3258e+00, -2.6575e+01],
                                                   [ 1.6455e+01, -3.0412e+01, -5.9070e+00, -1.7994e+01, -7.7708e+00],
                                                   [-1.7504e+01,  9.6857e+00,  3.2733e+01, -1.1493e+01,  6.0931e+00],
                                                   [-4.2322e+00,  1.7293e+01,  8.0773e+00, -3.5520e+01, -2.1028e+00],
                                                   [-2.1819e+01,  8.3739e+00,  2.2502e+00,  6.4162e+00, -6.4449e+00]],
                                         
                                                  [[-9.2576e+00, -6.9672e+00, -4.0686e+00, -7.5565e+00,  5.1920e+00],
                                                   [-1.6725e+01, -1.2407e+01,  4.7072e+00, -1.5580e+00,  8.3889e+00],
                                                   [ 3.1375e+00,  2.0321e+01, -6.2893e+00, -6.4338e+00, -3.6882e+00],
                                                   [ 2.2704e+00,  7.6596e+00, -5.7023e+00, -1.8606e+01,  1.8768e+00],
                                                   [ 6.7321e+00,  4.2171e+00,  1.8832e+00, -6.5496e+00,  5.5370e+00]]],
                                         
                                         
                                                 [[[ 1.9554e+01, -1.3538e+01,  2.3471e+01, -1.4669e+01, -8.0876e+00],
                                                   [ 5.9935e+00, -1.4617e+01,  1.0030e+01, -1.6091e+01, -4.0002e+00],
                                                   [ 9.4754e+00, -9.2002e+00,  1.6508e+00, -1.4299e+00,  9.2735e-01],
                                                   [ 4.1938e+00, -6.5662e+00,  1.6308e-01, -1.0217e+01, -4.1529e+00],
                                                   [-9.7351e+00,  6.0219e+00, -1.8128e+01,  1.9980e+01, -1.3195e+01]],
                                         
                                                  [[-7.9177e+00,  1.2681e+01, -5.1935e+00, -7.1072e+00, -1.8560e+00],
                                                   [-1.0267e+01, -1.7580e+00, -8.9404e+00, -1.5384e+01,  2.9346e+00],
                                                   [-3.4798e+00,  1.7023e+00, -3.2818e+00,  4.4891e+00, -1.3794e-02],
                                                   [-4.7645e+00, -7.2907e+00, -4.1505e+00, -2.3790e+00,  1.4794e+00],
                                                   [ 2.1159e+00, -6.0181e+00,  6.7175e+00, -8.3804e+00,  9.8221e+00]]]])

            let t1s23p32 = combo.tensor([[[[ 0.0000,  0.0000,  0.0000],
                                           [-3.2326, -1.2749, -3.3366],
                                           [-1.7567, -0.9686, -2.1970],
                                           [-1.4939,  2.3154, -0.4978],
                                           [ 5.1554, -0.8580, -1.6888]],
                                 
                                          [[ 0.0000,  0.0000,  0.0000],
                                           [-1.6604, -0.3488,  1.1702],
                                           [-2.1695, -0.4674,  4.5114],
                                           [ 0.6170,  0.3235,  4.8016],
                                           [ 3.4517,  0.1421,  1.8764]],
                                 
                                          [[ 0.0000,  0.0000,  0.0000],
                                           [-2.1929, -4.4554,  2.9319],
                                           [ 3.2436,  8.7959,  1.2112],
                                           [ 3.8262,  3.5775,  5.6113],
                                           [-1.9036, -1.5468,  0.0142]]],
                                 
                                 
                                         [[[ 0.0000,  0.0000,  0.0000],
                                           [-1.5589, -0.6350,  0.7208],
                                           [ 4.4022,  0.2401,  4.6891],
                                           [-1.1714, -9.2079, -4.1885],
                                           [ 1.9395,  5.5157,  3.1695]],
                                 
                                          [[ 0.0000,  0.0000,  0.0000],
                                           [ 0.8601,  0.7594,  2.9743],
                                           [ 3.9042, -0.1467,  1.2048],
                                           [ 0.3783, -3.1536,  5.7121],
                                           [ 1.4443,  1.0067,  4.0964]],
                                 
                                          [[ 0.0000,  0.0000,  0.0000],
                                           [ 0.8332, -8.5063, -0.7146],
                                           [-3.2521, -4.7905, -0.4381],
                                           [-0.6507,  4.6023, -2.5422],
                                           [-1.2853, -0.8996,  0.0497]]]])
            let t3s23p32 = dsharp.convTranspose2d(t1s23p32, t2, strides=[2;3], paddings=[3;2])
            let t3s23p32Correct = combo.tensor([[[[ -1.8653,   0.2227,  14.6477,  -6.0192,   4.7756],
                                                   [-13.3418,  -2.7359,   5.2295, -16.5054,  14.1446],
                                                   [  6.1026,   4.8817, -24.8045,  12.7971,   0.5372],
                                                   [  2.5602,   1.7874,  -4.4904,  23.0840,  14.9680],
                                                   [  5.0679,  -1.9103, -13.0832,   4.7500,  -0.2295]],
                                         
                                                  [[ -6.8511,   6.6723,  -3.2659,  -3.7369,  -4.8906],
                                                   [  5.9968,  -5.4707,  -7.2737,   9.2400,   4.7367],
                                                   [ -2.9852, -12.8116,   8.7978,   1.1795,  -3.6593],
                                                   [  6.1712,   4.0169,   7.1913,   2.4910,   4.5172],
                                                   [ -0.6795,  -5.3371,   1.6673,   5.0634, -10.2140]]],
                                         
                                         
                                                 [[[  0.6744,  -2.8392,  25.4059, -12.5133,  -3.0779],
                                                   [ 16.9127,  -1.2134,   7.4218, -16.6941,   1.7505],
                                                   [ -7.3768,  -2.0289,  14.0095,  -6.6914,  -7.1047],
                                                   [ -7.0010,  -2.8175,  -4.1471, -34.6982,  11.0266],
                                                   [ -1.1221,  15.8524,  -2.6974,   8.9922,   1.0017]],
                                         
                                                  [[ -1.6517,  12.2203,  -7.2672,  -3.4317,  -0.1174],
                                                   [-16.0018,   1.0966,  -8.0625,  -9.1513,   7.0926],
                                                   [  8.2073,   7.1299,  -4.6647,  -1.3393,   0.3008],
                                                   [  1.7930, -18.2269,  -2.2634,  21.3948,  -0.9061],
                                                   [ -1.8381,  -5.7929,  10.4651, -14.8689,   1.3356]]]])

            let t1p1d2 = combo.tensor([[[[ -1.4935,  -0.9143,   1.9049,  -3.4720],
                                           [ -0.0765,  -6.4800,  -5.8089,   1.8598],
                                           [ -4.9432,   0.7761,   4.2125,  -2.6577],
                                           [  3.2395,  -1.6309,   3.0082,   5.5846]],
                                 
                                          [[  0.8980,  -2.8900,   0.8966,  -1.4387],
                                           [ -1.3534,   3.0437,   1.8584,   2.4703],
                                           [  1.6080,   2.3951,   0.9763,   4.3595],
                                           [  2.8455,   4.4696,  -0.3192,  -0.7607]],
                                 
                                          [[  1.8914,  -2.6172,  -0.7348,   1.3387],
                                           [  1.5050,   6.0453,  -5.7601,  -5.8269],
                                           [ -1.9717,   3.9505,  -0.5285,  -4.7867],
                                           [ -1.6577,  -3.5756,  -2.8567,   1.3185]]],
                                 
                                 
                                         [[[  2.0819,   0.7653,  -1.9882,   1.9447],
                                           [ -1.2180,   0.8260,  -3.9099,   4.3648],
                                           [  1.3846,   1.3559,  -1.9401,   4.3954],
                                           [ -2.5044,   2.0114,   5.6507,   6.7569]],
                                 
                                          [[ -0.6521,  -2.0061,  -0.0293,   0.6525],
                                           [ -1.3767,  -2.5563,  -1.3317,  -0.2047],
                                           [ -1.4225,   2.7875,   0.7057,  -4.1782],
                                           [ -2.0456,   1.1288,   3.3816,  -3.9975]],
                                 
                                          [[-10.3514,   6.6914,   7.5311,  -4.3119],
                                           [  5.0292,  12.8169,  -0.9108,  -7.8711],
                                           [  2.2663,  -4.1982,   0.8442,   5.2652],
                                           [  2.8034,  -1.7984,  -8.3519,   4.9279]]]])
            let t3p1d2 = dsharp.convTranspose2d(t1p1d2, t2, padding=1, dilation=2)
            let t3p1d2Correct = combo.tensor([[[[  8.4979,   8.1578,  -0.9246,  -9.1176, -10.5874,  -6.6205],
                                                   [  5.2079, -13.8357,  28.0654,  -0.8426,  -2.0303,   7.9319],
                                                   [ 18.6946,   1.6702, -26.3933,  37.9199,  30.4063,  -2.4531],
                                                   [ -4.3368,   2.0038, -10.4593,  -2.6176,  11.9491,  -2.7395],
                                                   [  8.2586,  -1.6561,   0.9127,  12.4730,   1.7779, -16.6218],
                                                   [  3.3383,   2.1986,   1.1473,  -5.9113,  15.9062,  -0.5134]],
                                         
                                                  [[ -7.2998,  -3.4685,   9.0009,   8.7258,  14.5827,   0.4311],
                                                   [  7.0128,  13.9733,  -1.8878,  -1.3741,   3.4671,  -4.8869],
                                                   [ -5.7934,  12.6768,  27.1129, -11.6433, -22.6556, -17.3718],
                                                   [ -6.8525,   2.3969,  11.4891, -13.2466,   4.4319,   8.0043],
                                                   [  9.2020,   7.0856,  -1.6636,  -5.9827, -16.3868,  -3.3851],
                                                   [  0.3902,  -6.7436,   3.0319,  -6.3136,  -8.1966,   2.7587]]],
                                         
                                         
                                                 [[[ -7.4104,  -0.1922,   1.2678, -10.2139,  -4.0052,  -8.8859],
                                                   [  7.7276,  38.4212, -34.7024, -36.3262,  34.2309,   7.4849],
                                                   [  6.4525,  -5.1866, -56.6429,  10.9949,  52.3020,  18.4749],
                                                   [ -5.5316,   7.2902,  -0.3738, -26.0990,  -8.4270,  17.7335],
                                                   [ -2.9813, -20.0934, -20.1420,  36.3213,  10.0719, -17.1758],
                                                   [  0.5118,  -0.3265,  -2.8530,   4.9610, -15.5394,   2.0790]],
                                         
                                                  [[ -7.3016,  -7.5058,   3.0433,  11.1989,  16.5089,   6.5538],
                                                   [ -2.3799, -26.2130,  11.1512,  10.6192, -17.7198,   2.4928],
                                                   [-13.2641,  21.5614,  24.2401,  10.6934, -18.6836, -29.0676],
                                                   [  5.6654,  -8.2564,  -3.2337,  10.4815,   1.4508,  -1.1905],
                                                   [  3.8546,  24.1948,   3.2743, -13.9923,  -0.1919,   5.1839],
                                                   [ -1.4752,   5.4015,  -9.5561,   2.3548,   8.7044,  -1.3711]]]])

            let t1p22d23 = combo.tensor([[[[-1.2098e+00, -2.4110e+00, -1.0279e+00, -3.9876e+00],
                                           [ 5.6019e-01, -1.5290e+00,  1.2401e+00,  1.2266e-01],
                                           [ 1.9778e+00, -1.5180e+00, -1.3277e+00,  1.1161e+00],
                                           [ 7.8095e-01,  6.0152e+00, -1.1348e+00, -1.9066e+00],
                                           [ 2.4955e+00,  3.9095e+00,  1.1106e+00,  1.6221e+00],
                                           [-4.0381e+00, -3.6661e+00, -1.3509e+00, -4.5592e+00]],
                                 
                                          [[-1.0937e+00, -2.0893e-01,  1.9642e+00, -6.0165e-01],
                                           [-7.3989e-01, -5.2584e+00, -1.4846e+00,  2.1132e-01],
                                           [ 1.6045e+00, -3.0431e+00,  1.5164e+00,  2.7907e+00],
                                           [ 3.3791e+00,  5.5568e+00,  1.0130e+00,  4.2790e-01],
                                           [ 1.2678e+00,  3.2593e+00,  2.7528e+00, -1.6473e+00],
                                           [-4.9622e+00, -1.8143e+00, -2.2499e+00,  6.0567e-01]],
                                 
                                          [[-3.4310e+00, -2.9905e+00,  6.9098e-01, -3.8573e+00],
                                           [-1.5282e+00,  2.4647e-01,  2.8520e+00,  1.1805e+00],
                                           [ 4.1877e+00, -1.6244e+00, -3.7407e+00, -4.6168e+00],
                                           [-1.7002e+00,  1.5955e+00,  6.4699e+00,  2.2116e+00],
                                           [-5.5796e-01,  1.9423e+00, -1.5028e+00, -1.4009e+00],
                                           [ 2.4800e+00,  6.2988e-01,  1.3072e+00, -6.6665e+00]]],
                                 
                                 
                                         [[[-3.1804e+00, -5.1830e-01, -1.1245e+00, -2.0020e+00],
                                           [ 5.1911e-01, -1.7104e+00,  2.2359e+00,  4.3109e-02],
                                           [-4.8944e+00,  4.8992e+00,  1.6799e+00, -3.3535e+00],
                                           [ 1.4257e+00,  3.6713e+00, -4.5776e-01,  1.3292e+00],
                                           [ 2.8698e+00, -1.7510e+00,  5.5438e-01,  5.5704e-01],
                                           [-1.1954e+00,  6.5019e-01,  1.9188e+00,  8.1933e-02]],
                                 
                                          [[-4.6997e-01, -1.3293e+00, -6.7385e-01,  4.6287e+00],
                                           [-1.6234e+00, -1.0411e+00,  1.0147e+00,  1.0878e-01],
                                           [-5.8939e-01,  1.6040e+00, -7.2406e-01, -1.0665e+00],
                                           [ 1.8123e+00,  1.9490e+00, -4.8444e+00, -1.4087e+00],
                                           [ 5.6853e-01, -2.5669e-01,  3.1855e-01,  3.0923e+00],
                                           [-9.9076e-01,  4.7172e-03,  2.6959e+00, -1.8670e-01]],
                                 
                                          [[-6.2282e+00,  8.8515e-01, -2.2936e+00,  9.4559e-01],
                                           [-2.9560e+00,  9.6039e-01,  5.5681e+00,  1.3379e+00],
                                           [-4.0362e+00,  9.9716e+00,  1.6734e+00, -4.0311e+00],
                                           [ 3.0872e+00, -1.5992e+00, -7.6902e-01,  1.6764e+00],
                                           [ 4.4828e-01,  2.8493e+00,  5.6855e-01, -5.2895e+00],
                                           [ 2.8623e+00,  3.1194e+00, -3.9290e+00, -2.4554e+00]]]])
            let t3p22d23 = dsharp.convTranspose2d(t1p22d23, t2, paddings=[2;2], dilations=[2;3])
            let t3p22d23Correct = combo.tensor([[[[ 4.6280e+00,  2.4009e+01,  9.5642e+00, -3.9685e+00,  1.6077e+01,
                                                    -1.0013e+01],
                                                   [ 1.7769e+00,  6.6014e+00,  9.8511e+00, -8.8935e+00, -5.1148e-01,
                                                     2.1805e+01],
                                                   [ 7.0965e+00, -1.6816e+01,  1.6153e+01,  1.2492e+01,  2.1143e+01,
                                                     3.5813e+00],
                                                   [-1.7276e+00,  8.6048e+00, -1.6800e+01, -2.2964e+01, -3.5034e+01,
                                                    -1.3067e+01],
                                                   [-2.4116e+00, -7.3908e+00, -9.2208e+00,  6.0162e+00,  1.5574e+01,
                                                    -4.6266e+00],
                                                   [ 7.8526e+00,  6.3514e+00, -1.7287e+00, -6.4759e+00,  2.7634e+01,
                                                     8.5755e+00]],
                                         
                                                  [[-2.0408e-01,  7.3995e+00,  2.9141e-02,  4.1133e+00, -3.2888e+00,
                                                    -8.5523e-01],
                                                   [-6.6055e+00, -6.0599e+00, -2.8484e+00, -3.6159e-01, -5.8471e+00,
                                                    -1.9475e+01],
                                                   [ 1.1019e+01,  7.7522e+00, -6.4308e+00, -3.1367e+00, -1.0815e+01,
                                                     -1.5296e+01],
                                                   [-1.5233e+01, -7.2742e+00,  5.1823e+00,  1.1571e+01,  2.7042e+01,
                                                     2.2181e+01],
                                                   [ 1.4806e+00,  2.1242e+00,  9.2378e-01, -7.1384e+00, -3.4593e+00,
                                                     7.8981e+00],
                                                   [ 9.8048e-01,  1.0242e+01, -8.3664e-01,  7.4823e+00, -1.0282e+01,
                                                     -3.0258e+00]]],
                                         
                                         
                                                 [[[-1.6121e+00,  1.7316e+01,  5.1522e+00,  1.0198e+01, -2.4169e+01,
                                                     9.4833e+00],
                                                   [-9.8801e+00,  7.7356e+00,  2.7915e+00, -1.9914e+01, -1.9059e+00,
                                                     1.2361e+01],
                                                   [-1.6598e+00,  4.3962e+01, -3.7071e+01, -2.9644e+00,  2.4383e+00,
                                                     9.3214e+00],
                                                   [ 9.0554e+00, -9.5691e+00,  1.2193e+00,  3.6328e-01, -1.6113e+01,
                                                     -2.1219e+00],
                                                   [-2.1859e+00, -2.8941e+00, -1.8096e+01, -4.8515e+00,  9.2880e+00,
                                                     2.7867e+01],
                                                   [-1.0800e+01, -1.3079e+01, -1.0380e+01,  9.0377e+00,  1.7603e+01,
                                                     1.8126e+00]],
                                         
                                                  [[ 4.2091e+00, -6.5818e+00, -5.2048e+00, -3.0196e+00,  2.8719e+00,
                                                     2.3581e-01],
                                                   [-1.4303e+01, -7.8386e+00,  8.8451e-01,  3.7105e+00, -2.3244e+00,
                                                    -1.2665e+01],
                                                   [-1.0179e+00,  6.6568e+00,  8.0111e+00, -2.0887e+00, -2.1675e+01,
                                                     1.8483e+01],
                                                   [ 9.8004e+00, -5.4378e-01, -3.3078e+00,  4.2812e+00,  1.2749e+01,
                                                     8.1681e+00],
                                                   [-1.8430e+00,  2.6286e+00,  1.2923e+01,  2.7304e+00, -7.8955e+00,
                                                     4.8717e+00],
                                                   [ 4.5538e+00,  7.7849e+00, -1.1294e+00, -4.1387e-01,  1.0196e+00,
                                                     3.7936e+00]]]])

            let t1s3p6d3 = combo.tensor([[[[-0.4797,  1.2067,  0.8487, -0.9267],
                                           [ 0.0488,  2.9384, -2.8182, -2.7154],
                                           [ 0.9480, -2.3075, -4.5708, -2.2337],
                                           [ 0.1669,  4.3160,  2.9409, -0.7828]],
                                 
                                          [[-0.1887,  0.4049, -1.9126,  0.4331],
                                           [ 0.2998,  0.4966,  1.3509,  2.1225],
                                           [-0.3169, -2.3733, -4.2170, -0.0781],
                                           [-0.1093,  2.5067,  3.0689,  5.2431]],
                                 
                                          [[-2.0482,  1.2449,  0.3645,  0.2970],
                                           [ 1.1837,  8.8906, -0.6150, -0.3658],
                                           [ 1.4408,  2.9900, -8.0328, -0.4368],
                                           [ 0.8015, -0.6401, -0.4330, -0.6978]]],
                                 
                                 
                                         [[[-0.6182, -0.5837,  0.7181,  0.6395],
                                           [-1.5513, -1.4997,  0.8532,  0.0916],
                                           [ 0.0921, -0.2811,  0.0137, -2.8628],
                                           [-0.1444,  4.6484,  1.7724, -2.7309]],
                                 
                                          [[ 0.1016,  1.0336,  0.5866, -0.0869],
                                           [-0.3539,  0.7336,  1.4618,  1.5993],
                                           [ 0.6032, -0.6872, -2.0944, -1.2374],
                                           [-0.0151,  3.2930, -1.2824,  0.5289]],
                                 
                                          [[-0.8863, -0.9437, -1.2007, -0.1748],
                                           [-1.6423,  1.9599, -2.7169,  1.5076],
                                           [-1.4196,  1.2534, -3.9894,  3.1457],
                                           [-0.2654, -2.1439,  1.0330,  0.4360]]]])
            let t3s3p6d3 = dsharp.convTranspose2d(t1s3p6d3, t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = combo.tensor([[[[-38.8444,   0.0000,   0.0000,   8.3644],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [  7.0444,   0.0000,   0.0000,  90.9947]],
                                         
                                                  [[ -0.5142,   0.0000,   0.0000,  25.4986],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [ 36.8548,   0.0000,   0.0000,  -9.3262]]],
                                         
                                         
                                                 [[[-12.5651,   0.0000,   0.0000,   2.8805],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [-10.1415,   0.0000,   0.0000,  48.2164]],
                                         
                                                  [[ -3.6824,   0.0000,   0.0000, -12.8018],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [  0.0000,   0.0000,   0.0000,   0.0000],
                                                   [  5.7111,   0.0000,   0.0000, -31.2658]]]])

            Assert.True(t3Correct.allclose(t3, 0.01))
            Assert.True(t3p1Correct.allclose(t3p1, 0.01))
            Assert.True(t3p12Correct.allclose(t3p12, 0.01))
            Assert.True(t3s2Correct.allclose(t3s2, 0.01))
            Assert.True(t3s13Correct.allclose(t3s13, 0.01))
            Assert.True(t3s2p1Correct.allclose(t3s2p1, 0.01))
            Assert.True(t3s23p32Correct.allclose(t3s23p32, 0.01))
            Assert.True(t3p1d2Correct.allclose(t3p1d2, 0.01))
            Assert.True(t3p22d23Correct.allclose(t3p22d23, 0.01))
            Assert.True(t3s3p6d3Correct.allclose(t3s3p6d3, 0.01, 0.01))

    [<Test>]
    member _.TestTensorConvTranspose3D () =
        for combo in Combos.FloatingPoint do
            let t1 = combo.tensor([[[[ 0.9873,  2.7076, -0.9461],
                                       [-0.0808,  1.5441, -0.8709],
                                       [-0.8709,  0.3782,  2.0588]],

                                      [[ 1.0087, -0.8291,  0.8613],
                                       [-0.6963,  0.1493,  0.2307],
                                       [-0.0230,  1.0297,  1.7398]],

                                      [[ 2.0611, -1.6843, -1.0479],
                                       [-0.0454, -0.3567,  0.5329],
                                       [ 1.5642,  0.3775,  1.8207]]]]).unsqueeze(0)
            let t2 = combo.tensor([[[[-0.6863,  0.6292,  1.2939],
                                       [ 0.6178, -1.1568, -1.2094],
                                       [ 0.2491,  1.3155,  0.3311]],

                                      [[-0.1488,  0.1148, -2.6754],
                                       [ 1.0680,  0.5176,  0.4799],
                                       [-0.8843, -1.2587, -0.5647]],

                                      [[-0.1586,  0.1037, -0.8961],
                                       [-0.5436,  0.7449, -1.4694],
                                       [-0.5542,  0.4589,  0.9205]]],


                                     [[[-0.7661,  0.1054,  0.0801],
                                       [ 0.8272, -0.0132, -2.3537],
                                       [-0.8411,  0.6373, -0.4968]],

                                      [[ 0.4365,  1.0976, -1.0754],
                                       [ 0.6496, -0.2016, -0.5867],
                                       [ 0.7225, -0.6232,  1.1162]],

                                      [[-0.0697, -0.5219, -0.3690],
                                       [ 1.5946, -0.9011, -0.1317],
                                       [-0.5122, -1.3610, -0.1057]]]]).unsqueeze(0)
              
            let t3 = dsharp.convTranspose3d(t1, t2)
            let t3Correct = combo.tensor([[[[-0.6776, -1.2371,  3.6305,  2.9081, -1.2242],
                                               [ 0.6655, -0.5798, -3.4461, -0.7301,  0.0174],
                                               [ 0.7937,  2.2132, -0.8753,  0.5767,  3.4039],
                                               [-0.5582,  1.5194,  3.6753, -3.4734, -2.7783],
                                               [-0.2169, -1.0514,  0.7219,  2.8336,  0.6817]],

                                              [[-0.8392,  0.9142, -1.9974, -7.8834,  3.6458],
                                               [ 2.1676,  0.9441,  0.6938, -3.0770,  1.1327],
                                               [-0.9930, -0.8891, -1.5376,  2.0150, -3.1344],
                                               [-1.0463, -1.5267,  0.7838, -1.4336, -0.5480],
                                               [ 0.7644,  0.9879, -0.0247, -0.1753, -0.5864]],

                                              [[-1.7213,  2.3650, -1.0495, -3.0462, -2.8125],
                                               [ 1.9617, -4.6640,  2.4310, -3.3593,  3.9237],
                                               [-2.5857, -0.1416,  4.5485, -4.4521, -5.1023],
                                               [ 2.0645, -1.6396,  2.3854,  1.0397, -5.1477],
                                               [ 0.8926,  0.6609, -3.1227,  1.0417,  1.5156]],

                                              [[-0.4667,  0.7234, -6.6784,  5.2182,  2.0317],
                                               [ 1.7702,  0.4220, -2.9658,  1.4148, -3.4009],
                                               [-2.2808, -1.2291, -1.2356,  0.4161, -5.1288],
                                               [ 2.1092,  0.6063,  2.0487,  0.6804, -1.7714],
                                               [-1.3705, -2.8840, -3.4814, -0.7586,  0.5735]],

                                              [[-0.3269,  0.4809, -1.8555,  1.4006,  0.9390],
                                               [-1.1133,  2.5028, -3.7944,  2.0693,  1.0622],
                                               [-1.3657,  2.1418, -0.4349, -1.2597, -3.3792],
                                               [-0.8252,  1.1367, -3.5079,  0.7176, -2.1848],
                                               [-0.8669,  0.5087,  0.6042,  1.1831,  1.6760]]],


                                             [[[-0.7564, -1.9702,  1.0893,  0.1172, -0.0758],
                                               [ 0.8786,  1.0353, -2.3188, -6.3286,  2.1572],
                                               [-0.2301, -0.7514, -0.1270, -5.3238,  2.6849],
                                               [-0.6524, -1.0259,  5.5045, -2.2395, -4.4132],
                                               [ 0.7325, -0.8731, -1.0580,  1.1241, -1.0229]],

                                              [[-0.3418,  3.0070,  0.8309, -3.9259,  1.0865],
                                               [ 1.9740,  1.2583, -2.2057, -2.0378, -0.5173],
                                               [-1.1262,  2.2510, -1.0006,  5.6069, -3.5906],
                                               [-0.0575,  1.8699,  1.8174, -0.7445, -6.3896],
                                               [-0.6098, -0.0648, -0.5161, -0.2638,  1.4335]],

                                              [[-1.2076,  1.5488, -2.5398,  1.0863, -0.6611],
                                               [ 3.6711,  0.7693, -9.7912,  4.7919,  2.2017],
                                               [-3.2770,  1.9780, -3.2797,  0.7986, -2.1776],
                                               [-0.5332,  2.4850, -1.1911, -2.2108, -5.4925],
                                               [-0.8862,  2.4291, -2.9556, -1.8043,  0.8196]],

                                              [[ 0.8293,  1.0586, -4.5222,  0.5174,  0.8091],
                                               [ 2.9762, -3.5933,  0.4902,  1.3255, -0.1569],
                                               [ 0.5169, -0.9847,  2.8202, -2.1327, -4.2036],
                                               [ 1.3033,  2.2345,  2.3475, -3.3519, -0.7269],
                                               [ 1.1419, -1.1981,  0.5359, -3.1900,  1.8482]],

                                              [[-0.1437, -0.9583,  0.1916,  1.1684,  0.3867],
                                               [ 3.2899, -4.4946, -0.2590,  1.0196, -0.0586],
                                               [-1.2372, -3.3131,  2.8871,  0.0815, -0.6312],
                                               [ 2.5176, -0.5630,  2.5744, -2.3779, -0.2962],
                                               [-0.8012, -2.3222, -1.6117, -2.5178, -0.1925]]]]).unsqueeze(0)

            let t3p1 = dsharp.convTranspose3d(t1, t2, padding=1)
            let t3p1Correct = combo.tensor([[[[ 0.9441,  0.6938, -3.0770],
                                               [-0.8891, -1.5376,  2.0150],
                                               [-1.5267,  0.7838, -1.4336]],

                                              [[-4.6640,  2.4310, -3.3593],
                                               [-0.1416,  4.5485, -4.4521],
                                               [-1.6396,  2.3854,  1.0397]],

                                              [[ 0.4220, -2.9658,  1.4148],
                                               [-1.2291, -1.2356,  0.4161],
                                               [ 0.6063,  2.0487,  0.6804]]],


                                             [[[ 1.2583, -2.2057, -2.0378],
                                               [ 2.2510, -1.0006,  5.6069],
                                               [ 1.8699,  1.8174, -0.7445]],

                                              [[ 0.7693, -9.7912,  4.7919],
                                               [ 1.9780, -3.2797,  0.7986],
                                               [ 2.4850, -1.1911, -2.2108]],

                                              [[-3.5933,  0.4902,  1.3255],
                                               [-0.9847,  2.8202, -2.1327],
                                               [ 2.2345,  2.3475, -3.3519]]]]).unsqueeze(0)

            let t3p122 = dsharp.convTranspose3d(t1, t2, paddings=[1; 2; 2])
            let t3p122Correct = combo.tensor([[[[-1.5376]],

                                                  [[ 4.5485]],

                                                  [[-1.2356]]],


                                                 [[[-1.0006]],

                                                  [[-3.2797]],

                                                  [[ 2.8202]]]]).unsqueeze(0)

            let t3s2 = dsharp.convTranspose3d(t1, t2, stride=2)
            let t3s2Correct = combo.tensor([[[[-6.7761e-01,  6.2121e-01, -5.8084e-01,  1.7037e+00,  4.1528e+00,
                                                -5.9531e-01, -1.2242e+00],
                                               [ 6.0999e-01, -1.1421e+00,  4.7885e-01, -3.1322e+00, -3.8592e+00,
                                                 1.0945e+00,  1.1442e+00],
                                               [ 3.0137e-01,  1.2479e+00, -1.6300e-01,  4.5334e+00,  3.2566e+00,
                                                 -1.7926e+00, -1.4401e+00],
                                               [-4.9924e-02,  9.3474e-02,  1.0517e+00, -1.7862e+00, -2.4055e+00,
                                                 1.0074e+00,  1.0532e+00],
                                               [ 5.7757e-01, -6.5425e-01, -1.0286e+00,  2.2692e+00, -6.2927e-01,
                                                 1.4977e-01,  2.3755e+00],
                                               [-5.3806e-01,  1.0074e+00,  1.2869e+00, -4.3751e-01,  8.1462e-01,
                                                -2.3816e+00, -2.4899e+00],
                                               [-2.1691e-01, -1.1456e+00, -1.9417e-01,  4.9752e-01,  6.3803e-01,
                                                 2.7084e+00,  6.8174e-01]],

                                              [[-1.4691e-01,  1.1336e-01, -3.0443e+00,  3.1089e-01, -7.1032e+00,
                                                -1.0863e-01,  2.5313e+00],
                                               [ 1.0545e+00,  5.1107e-01,  3.3656e+00,  1.4016e+00,  2.8882e-01,
                                                 -4.8976e-01, -4.5402e-01],
                                               [-8.6109e-01, -1.2520e+00, -2.9655e+00, -3.2308e+00, -4.6937e+00,
                                                 1.0909e+00,  2.8642e+00],
                                               [-8.6301e-02, -4.1827e-02,  1.6104e+00,  7.9930e-01, -1.8915e-01,
                                                -4.5081e-01, -4.1791e-01],
                                               [ 2.0104e-01,  1.7161e-03,  9.5375e-01, -1.9002e+00, -1.4199e+00,
                                                 1.3326e+00, -5.0164e+00],
                                               [-9.3011e-01, -4.5080e-01, -1.3973e-02,  1.9577e-01,  2.3804e+00,
                                                 1.0657e+00,  9.8797e-01],
                                               [ 7.7015e-01,  1.0962e+00,  1.5728e-01, -4.7605e-01, -2.0343e+00,
                                                 -2.5915e+00, -1.1625e+00]],

                                              [[-8.4890e-01,  7.3710e-01,  5.6003e-01, -2.4080e-01, -3.9402e+00,
                                                 4.4382e-01,  1.9623e+00],
                                               [ 8.6515e-02, -4.3149e-01, -4.6548e+00,  2.9759e+00, -1.9294e+00,
                                                 -1.7011e+00,  3.4855e-01],
                                               [ 1.9484e-01,  1.3335e+00, -1.6401e+00,  4.0610e-01,  1.7460e+00,
                                                 7.5368e-01,  4.9317e-01],
                                               [-3.8631e-01,  7.4535e-01,  2.1371e-01,  9.7747e-01, -1.8335e+00,
                                                -9.1553e-01,  1.0007e+00],
                                               [ 2.5271e-02, -1.0580e+00, -1.1396e+00,  1.5921e+00,  1.4837e+00,
                                                 1.2120e+00, -3.1902e-01],
                                               [ 4.5918e-01, -6.2202e-01,  1.7381e+00, -9.0946e-01, -1.8453e+00,
                                                 -4.7914e-01, -5.1294e+00],
                                               [ 4.7689e-01, -4.2998e-01, -7.6240e-01,  1.5281e+00, -1.8506e-02,
                                                 3.2336e+00,  2.4713e+00]],

                                              [[-1.5009e-01,  1.1582e-01, -2.5754e+00, -9.5193e-02,  2.0899e+00,
                                                 9.8899e-02, -2.3044e+00],
                                               [ 1.0773e+00,  5.2215e-01, -4.0141e-01, -4.2916e-01,  5.2209e-01,
                                                 4.4587e-01,  4.1333e-01],
                                               [-7.8844e-01, -1.3496e+00,  2.0044e+00,  1.0607e+00, -7.2726e-01,
                                                -1.0577e+00, -1.1035e+00],
                                               [-7.4372e-01, -3.6046e-01, -1.7473e-01,  7.7268e-02,  3.1800e-01,
                                                 1.1941e-01,  1.1070e-01],
                                               [ 6.1924e-01,  8.7386e-01,  1.6961e-01, -6.9657e-02, -3.3020e+00,
                                                 -9.0590e-02, -4.7850e+00],
                                               [-2.4604e-02, -1.1925e-02,  1.0887e+00,  5.3302e-01,  2.3523e+00,
                                                 9.0061e-01,  8.3490e-01],
                                               [ 2.0373e-02,  2.8997e-02, -8.9760e-01, -1.2961e+00, -2.1200e+00,
                                                 -2.1900e+00, -9.8242e-01]],

                                              [[-1.5746e+00,  1.4015e+00,  3.0505e+00, -1.1458e+00, -8.5383e-01,
                                                -5.6999e-01, -2.1277e+00],
                                               [ 7.2511e-01, -1.6330e+00, -4.5649e+00,  1.3309e+00,  2.1396e+00,
                                                 1.8538e+00,  1.6567e-03],
                                               [ 9.5978e-02,  3.0736e+00,  2.4374e+00, -2.8052e+00, -3.0569e+00,
                                                 -6.2395e-01,  9.2870e-01],
                                               [ 3.5049e-01, -4.6614e-01,  7.7661e-01,  5.2383e-01,  4.1591e-01,
                                                 -4.4464e-01, -9.8344e-01],
                                               [-6.9532e-01,  6.0249e-01,  7.9457e-01, -5.6395e-02, -1.9356e+00,
                                                 2.1329e+00,  1.1855e+00],
                                               [ 9.7896e-01, -1.8267e+00, -2.1844e+00,  3.3026e-01, -1.7905e+00,
                                                 -8.1025e-01, -4.7584e+00],
                                               [ 4.0237e-01,  2.0471e+00,  2.0140e-02,  9.6920e-01,  5.6216e-01,
                                                 3.1936e+00,  2.2044e+00]],

                                              [[-3.0669e-01,  2.3666e-01, -5.2638e+00, -1.9339e-01,  4.6621e+00,
                                                -1.2032e-01,  2.8035e+00],
                                               [ 2.2014e+00,  1.0669e+00, -8.0981e-01, -8.7187e-01, -1.9274e+00,
                                                 -5.4243e-01, -5.0285e-01],
                                               [-1.8160e+00, -2.5996e+00,  5.0024e-01,  2.0791e+00,  2.7528e+00,
                                                 1.3802e+00, -8.3402e-01],
                                               [-4.8506e-02, -2.3509e-02, -4.0277e-01, -1.8465e-01,  3.9798e-01,
                                                 2.7585e-01,  2.5572e-01],
                                               [-1.9259e-01,  2.3677e-01, -3.9000e+00,  4.9234e-01, -1.5508e+00,
                                                -4.6172e-01, -5.1720e+00],
                                               [ 1.6706e+00,  8.0970e-01,  1.1538e+00,  1.9542e-01,  2.1257e+00,
                                                 9.4246e-01,  8.7370e-01],
                                               [-1.3833e+00, -1.9689e+00, -1.2171e+00, -4.7519e-01, -1.8233e+00,
                                                -2.2917e+00, -1.0281e+00]],

                                              [[-3.2691e-01,  2.1379e-01, -1.5799e+00, -1.7471e-01,  1.6755e+00,
                                                -1.0869e-01,  9.3903e-01],
                                               [-1.1205e+00,  1.5353e+00, -2.1130e+00, -1.2546e+00,  3.0446e+00,
                                                -7.8053e-01,  1.5398e+00],
                                               [-1.1351e+00,  9.4124e-01,  2.9280e+00, -8.1000e-01, -7.3459e-01,
                                                -4.2564e-01, -1.4421e+00],
                                               [ 2.4689e-02, -3.3828e-02,  2.6065e-01, -2.6570e-01,  2.3445e-01,
                                                 3.9693e-01, -7.8304e-01],
                                               [-2.2292e-01,  1.4141e-01, -1.3057e+00, -1.2455e-01, -1.2508e+00,
                                                 4.3342e-01, -1.1410e+00],
                                               [-8.5033e-01,  1.1651e+00, -2.5037e+00,  2.8120e-01, -1.5445e+00,
                                                 1.3562e+00, -2.6753e+00],
                                               [-8.6687e-01,  7.1788e-01,  1.2307e+00,  1.7326e-01, -6.6148e-01,
                                                 8.3559e-01,  1.6760e+00]]],


                                             [[[-7.5636e-01,  1.0406e-01, -1.9952e+00,  2.8539e-01,  9.4179e-01,
                                                -9.9724e-02, -7.5817e-02],
                                               [ 8.1670e-01, -1.3030e-02, -8.4078e-02, -3.5733e-02, -7.1557e+00,
                                                 1.2486e-02,  2.2269e+00],
                                               [-7.6851e-01,  6.2065e-01, -3.9573e+00,  1.8882e+00,  2.4143e-01,
                                                -6.9473e-01,  4.0029e-01],
                                               [-6.6841e-02,  1.0664e-03,  1.4675e+00, -2.0378e-02, -4.3548e+00,
                                                 1.1493e-02,  2.0498e+00],
                                               [ 7.3512e-01, -1.4328e-01, -1.6181e+00,  1.0239e+00, -1.5816e+00,
                                                 -3.3798e-01,  5.9767e-01],
                                               [-7.2039e-01,  1.1493e-02,  2.3626e+00, -4.9912e-03,  8.1287e-01,
                                                -2.7170e-02, -4.8459e+00],
                                               [ 7.3248e-01, -5.5497e-01,  1.1458e-01,  2.4101e-01, -1.9196e+00,
                                                 1.3120e+00, -1.0229e+00]],

                                              [[ 4.3095e-01,  1.0837e+00,  1.2014e-01,  2.9720e+00, -3.3247e+00,
                                                 -1.0385e+00,  1.0174e+00],
                                               [ 6.4138e-01, -1.9905e-01,  1.1797e+00, -5.4590e-01, -2.2031e+00,
                                                 1.9075e-01,  5.5506e-01],
                                               [ 6.7800e-01, -7.0397e-01,  3.8190e+00,  7.4930e-03,  2.9795e-01,
                                                 -3.6628e-01, -1.1950e-01],
                                               [-5.2492e-02,  1.6291e-02,  1.0505e+00, -3.1132e-01, -1.4716e+00,
                                                 1.7558e-01,  5.1091e-01],
                                               [-4.3850e-01, -9.0554e-01,  2.1269e+00, -5.4716e-01,  1.5862e+00,
                                                 2.8025e+00, -3.1860e+00],
                                               [-5.6574e-01,  1.7558e-01,  7.5659e-01, -7.6251e-02,  1.1156e+00,
                                                -4.1509e-01, -1.2078e+00],
                                               [-6.2916e-01,  5.4272e-01, -6.9879e-01, -2.3569e-01,  1.9095e+00,
                                                -1.2830e+00,  2.2979e+00]],

                                              [[-8.4161e-01, -4.0897e-01,  1.6287e-01, -1.5005e+00, -1.6594e+00,
                                                 5.8459e-01,  4.1813e-01],
                                               [ 2.4088e+00, -9.0297e-01,  1.1275e+00, -2.4289e+00,  7.9854e-01,
                                                 8.4120e-01, -1.9027e+00],
                                               [-8.1502e-01, -7.3211e-01, -1.5431e+00, -5.0035e+00, -7.8804e-01,
                                                 2.3154e+00,  1.1917e-02],
                                               [-7.0487e-01,  8.2002e-02,  4.2354e+00, -1.3934e+00, -1.7526e+00,
                                                 7.8171e-01, -4.2824e-01],
                                               [ 7.0546e-01,  1.1831e-01, -1.0576e+00, -2.0953e+00, -1.5189e+00,
                                                 4.4112e-01, -6.4277e-01],
                                               [-1.4078e+00,  7.8505e-01,  1.6238e+00, -3.5439e-01,  2.2488e+00,
                                                -1.8782e+00, -4.3663e+00],
                                               [ 4.6543e-01,  1.1706e+00, -9.5626e-01,  1.4146e-01, -3.0695e+00,
                                                 -1.6933e+00, -1.0821e+00]],

                                              [[ 4.4030e-01,  1.1072e+00, -1.4466e+00, -9.1001e-01,  1.2675e+00,
                                                 9.4543e-01, -9.2626e-01],
                                               [ 6.5529e-01, -2.0337e-01, -1.1304e+00,  1.6715e-01,  1.0459e+00,
                                                 -1.7366e-01, -5.0531e-01],
                                               [ 4.2479e-01, -1.3930e+00,  1.3409e+00,  6.8051e-01, -3.6292e-01,
                                                 -2.8358e-01,  7.1332e-01],
                                               [-4.5237e-01,  1.4039e-01,  5.0549e-01, -3.0095e-02,  6.2284e-02,
                                                -4.6508e-02, -1.3533e-01],
                                               [-5.1313e-01,  4.0867e-01, -1.9516e-01,  1.0372e+00, -1.4634e-02,
                                                 1.7659e+00, -1.6135e+00],
                                               [-1.4965e-02,  4.6446e-03,  6.8244e-01, -2.0760e-01,  5.2616e-01,
                                                -3.5078e-01, -1.0207e+00],
                                               [-1.6643e-02,  1.4356e-02,  7.1819e-01, -6.4171e-01,  2.4062e+00,
                                                -1.0843e+00,  1.9419e+00]],

                                              [[-1.6494e+00, -3.0921e-01,  1.1411e+00,  2.5517e-01,  9.1365e-01,
                                                -5.5999e-01, -4.0179e-01],
                                               [ 3.3135e+00, -9.3616e-01, -7.6996e+00,  7.6930e-01,  4.5803e+00,
                                                 -7.6233e-01,  2.3530e+00],
                                               [-2.1669e+00,  2.9930e-01,  1.2267e+00, -6.0504e-02,  8.5671e-01,
                                                -1.9043e+00,  3.8715e-01],
                                               [-1.1480e+00,  6.2808e-01,  1.4158e-01, -1.2980e-01,  1.6286e+00,
                                                -2.1490e-01, -1.2847e+00],
                                               [-8.0186e-01,  1.0957e+00,  9.2596e-02, -9.2809e-01, -2.2707e+00,
                                                -6.9050e-01, -7.8523e-01],
                                               [ 1.2572e+00,  1.1544e-04, -1.7244e+00, -9.3285e-01,  3.2562e+00,
                                                 -1.5918e+00, -4.5146e+00],
                                               [-1.3039e+00,  1.0282e+00, -1.6197e+00, -1.1608e+00, -2.7190e+00,
                                                -1.2076e+00, -1.0886e+00]],

                                              [[ 8.9968e-01,  2.2624e+00, -2.9517e+00, -1.8488e+00,  1.3539e+00,
                                                 -1.1502e+00,  1.1269e+00],
                                               [ 1.3390e+00, -4.1556e-01, -2.3034e+00,  3.3958e-01,  3.0738e-01,
                                                 2.1127e-01,  6.1475e-01],
                                               [ 1.4693e+00, -1.3344e+00,  9.7688e-01,  6.5812e-01, -2.0208e+00,
                                                 1.2380e+00, -1.7427e+00],
                                               [-2.9503e-02,  9.1565e-03, -2.0508e-01,  7.1917e-02,  5.5545e-01,
                                                -1.0744e-01, -3.1263e-01],
                                               [ 6.4996e-01,  1.7452e+00, -1.8257e+00,  6.3668e-01,  3.7559e-01,
                                                 1.6663e+00, -1.3631e+00],
                                               [ 1.0162e+00, -3.1537e-01, -6.7241e-01, -7.6114e-02,  9.6129e-01,
                                                 -3.6708e-01, -1.0681e+00],
                                               [ 1.1301e+00, -9.7481e-01,  2.0186e+00, -2.3527e-01,  1.7367e+00,
                                                 -1.1346e+00,  2.0322e+00]],

                                              [[-1.4373e-01, -1.0757e+00, -6.4308e-01,  8.7907e-01,  6.9455e-01,
                                                 5.4691e-01,  3.8665e-01],
                                               [ 3.2868e+00, -1.8573e+00, -2.9573e+00,  1.5177e+00, -1.4491e+00,
                                                 9.4426e-01,  1.3803e-01],
                                               [-1.0525e+00, -2.7815e+00,  6.8638e-01,  2.4785e+00,  8.0928e-01,
                                                 1.1480e+00, -8.5826e-02],
                                               [-7.2421e-02,  4.0924e-02, -5.6283e-01,  3.2143e-01,  8.9676e-01,
                                                -4.8020e-01, -7.0194e-02],
                                               [-8.5815e-02, -7.5458e-01, -4.1599e-01,  2.8844e-01, -5.0149e-01,
                                                -1.6755e+00, -7.2815e-01],
                                               [ 2.4943e+00, -1.4095e+00,  3.9596e-01, -3.4019e-01,  2.8536e+00,
                                                 -1.6406e+00, -2.3982e-01],
                                               [-8.0118e-01, -2.1289e+00, -3.5876e-01, -5.1380e-01, -9.7246e-01,
                                                -2.4779e+00, -1.9252e-01]]]]).unsqueeze(0)

            let t3s132 = dsharp.convTranspose3d(t1, t2, strides=[1;3;2])
            let t3s132Correct = combo.tensor([[[[-6.7761e-01,  6.2121e-01, -5.8084e-01,  1.7037e+00,  4.1528e+00,
                                                  -5.9531e-01, -1.2242e+00],
                                                 [ 6.0999e-01, -1.1421e+00,  4.7885e-01, -3.1322e+00, -3.8592e+00,
                                                   1.0945e+00,  1.1442e+00],
                                                 [ 2.4591e-01,  1.2988e+00,  1.0013e+00,  3.5619e+00,  6.6093e-01,
                                                   -1.2446e+00, -3.1330e-01],
                                                 [ 5.5458e-02, -5.0842e-02, -1.1643e+00,  9.7157e-01,  2.5957e+00,
                                                   -5.4796e-01, -1.1268e+00],
                                                 [-4.9924e-02,  9.3474e-02,  1.0517e+00, -1.7862e+00, -2.4055e+00,
                                                   1.0074e+00,  1.0532e+00],
                                                 [-2.0126e-02, -1.0630e-01,  3.5784e-01,  2.0313e+00,  2.9439e-01,
                                                  -1.1456e+00, -2.8838e-01],
                                                 [ 5.9770e-01, -5.4795e-01, -1.3864e+00,  2.3797e-01, -9.2366e-01,
                                                   1.2954e+00,  2.6639e+00],
                                                 [-5.3806e-01,  1.0074e+00,  1.2869e+00, -4.3751e-01,  8.1462e-01,
                                                  -2.3816e+00, -2.4899e+00],
                                                 [-2.1691e-01, -1.1456e+00, -1.9417e-01,  4.9752e-01,  6.3803e-01,
                                                   2.7084e+00,  6.8174e-01]],

                                                [[-8.3922e-01,  7.4805e-01, -1.1701e+00, -2.1076e-01, -8.7671e+00,
                                                   4.3332e-01,  3.6458e+00],
                                                 [ 1.6777e+00, -6.5582e-01,  1.6334e+00,  2.3607e+00,  1.8237e+00,
                                                   -1.4862e+00, -1.4957e+00],
                                                 [-6.2186e-01,  8.4231e-02, -2.8244e+00, -4.4988e+00, -7.5218e-01,
                                                   2.3240e+00,  8.1946e-01],
                                                 [ 4.8995e-01, -4.4742e-01, -1.0170e+00,  2.7122e-01, -3.9667e+00,
                                                   4.5149e-02,  2.6284e+00],
                                                 [-5.1653e-01,  7.6371e-01,  2.5448e+00,  6.2662e-01, -2.2715e-01,
                                                  -7.1766e-01, -6.9689e-01],
                                                 [-1.0198e-01, -8.1433e-01, -1.5133e+00, -1.7472e+00,  5.1406e-03,
                                                   1.3997e+00,  5.6814e-01],
                                                 [ 1.4539e-01, -1.1449e-01,  1.5371e+00,  6.9132e-01, -1.1799e+00,
                                                   1.3311e+00, -3.2570e+00],
                                                 [-9.4435e-01, -4.2415e-01,  6.5008e-01, -9.9539e-01,  2.2100e+00,
                                                  -9.4693e-01, -1.1162e+00],
                                                 [ 7.6441e-01,  1.0659e+00,  4.0613e-01,  8.7852e-01, -1.2599e+00,
                                                   -3.0272e-01, -5.8641e-01]],

                                                [[-1.7213e+00,  1.5151e+00, -6.6590e-02, -8.7412e-01, -1.6465e+00,
                                                  -6.5858e-01, -2.8125e+00],
                                                 [ 1.8141e+00, -1.1268e+00, -6.8575e+00,  3.5361e+00, -1.5526e+00,
                                                   9.5334e-01,  3.0709e+00],
                                                 [-9.2582e-01,  1.8949e+00, -1.6511e-01,  7.0500e-02,  1.9044e+00,
                                                  -2.8969e+00, -1.7043e+00],
                                                 [ 1.4760e-01, -1.1691e-01,  1.8544e+00, -4.7139e-02, -2.5066e+00,
                                                   2.7146e-01,  8.5278e-01],
                                                 [-7.2786e-01, -3.6811e-01, -1.0609e+00,  1.6401e+00, -7.1685e-01,
                                                  -1.1457e+00,  7.4589e-01],
                                                 [ 6.4928e-01,  7.7968e-01, -7.7280e-01,  5.1527e-02,  1.6304e+00,
                                                   1.0984e-02, -7.5546e-01],
                                                 [-9.3201e-01,  8.9124e-01,  2.3937e+00,  3.9500e-01, -4.4403e+00,
                                                   1.5589e+00, -4.1439e+00],
                                                 [ 1.4152e+00, -2.4701e+00,  5.0425e-01,  3.7800e-01,  1.3457e+00,
                                                   3.2795e-01, -4.3923e+00],
                                                 [ 8.9260e-01,  1.6870e+00, -1.2969e+00, -6.2591e-01, -2.3344e+00,
                                                   1.1500e+00,  1.5156e+00]],

                                                [[-4.6668e-01,  3.4129e-01, -6.0363e+00, -2.7939e-01,  5.2685e+00,
                                                  -3.0976e-02,  2.0317e+00],
                                                 [ 1.6530e+00,  1.8183e+00, -1.8413e+00, -1.4894e+00, -1.1774e+00,
                                                   9.9144e-02, -1.7685e+00],
                                                 [-2.3818e+00, -2.1315e+00,  1.7137e+00,  1.7396e+00,  6.3724e-01,
                                                   1.7143e+00,  1.3846e+00],
                                                 [ 1.1720e-01, -7.7444e-02,  7.7492e-01, -2.5474e-02,  7.0469e-01,
                                                   8.5115e-02, -1.6324e+00],
                                                 [ 3.3004e-01, -5.4219e-01,  5.3930e-01, -7.3462e-02,  5.3244e-02,
                                                   4.4767e-01, -8.3234e-02],
                                                 [ 4.2607e-01, -2.6242e-01, -3.8263e-01,  5.1750e-01, -2.6028e-01,
                                                   -5.6490e-01, -8.8564e-02],
                                                 [-2.2910e-01,  1.7721e-01, -4.3838e+00,  1.5015e-01, -2.4796e+00,
                                                   3.8952e-01, -6.4302e+00],
                                                 [ 1.6832e+00,  7.9254e-01,  6.2792e-01,  9.6240e-01, -3.3314e-01,
                                                   2.2384e+00, -1.6828e+00],
                                                 [-1.3705e+00, -1.9795e+00, -1.8090e+00, -2.6188e-03, -1.8396e+00,
                                                  -1.4932e+00,  5.7349e-01]],

                                                [[-3.2691e-01,  2.1379e-01, -1.5799e+00, -1.7471e-01,  1.6755e+00,
                                                  -1.0869e-01,  9.3903e-01],
                                                 [-1.1205e+00,  1.5353e+00, -2.1130e+00, -1.2546e+00,  3.0446e+00,
                                                  -7.8053e-01,  1.5398e+00],
                                                 [-1.1423e+00,  9.4595e-01,  2.8308e+00, -7.7300e-01, -9.6972e-01,
                                                  -4.8092e-01, -9.6460e-01],
                                                 [ 7.2031e-03, -4.7108e-03,  9.7273e-02, -3.7000e-02,  2.3513e-01,
                                                   5.5275e-02, -4.7754e-01],
                                                 [ 2.4689e-02, -3.3828e-02,  2.6065e-01, -2.6570e-01,  2.3445e-01,
                                                   3.9693e-01, -7.8304e-01],
                                                 [ 2.5169e-02, -2.0843e-02,  1.5588e-01, -1.6371e-01, -6.2368e-01,
                                                   2.4457e-01,  4.9054e-01],
                                                 [-2.4809e-01,  1.6225e-01, -1.4616e+00,  3.9159e-02, -6.2707e-01,
                                                   1.8885e-01, -1.6315e+00],
                                                 [-8.5033e-01,  1.1651e+00, -2.5037e+00,  2.8120e-01, -1.5445e+00,
                                                   1.3562e+00, -2.6753e+00],
                                                 [-8.6687e-01,  7.1788e-01,  1.2307e+00,  1.7326e-01, -6.6148e-01,
                                                   8.3559e-01,  1.6760e+00]]],


                                               [[[-7.5636e-01,  1.0406e-01, -1.9952e+00,  2.8539e-01,  9.4179e-01,
                                                  -9.9724e-02, -7.5817e-02],
                                                 [ 8.1670e-01, -1.3030e-02, -8.4078e-02, -3.5733e-02, -7.1557e+00,
                                                   1.2486e-02,  2.2269e+00],
                                                 [-8.3041e-01,  6.2917e-01, -2.7679e+00,  1.7255e+00, -5.4948e-01,
                                                  -6.0293e-01,  4.7008e-01],
                                                 [ 6.1903e-02, -8.5168e-03, -1.1894e+00,  1.6275e-01,  7.9091e-01,
                                                   -9.1792e-02, -6.9787e-02],
                                                 [-6.6841e-02,  1.0664e-03,  1.4675e+00, -2.0378e-02, -4.3548e+00,
                                                   1.1493e-02,  2.0498e+00],
                                                 [ 6.7964e-02, -5.1493e-02, -1.2586e+00,  9.8401e-01, -3.4686e-02,
                                                   -5.5498e-01,  4.3269e-01],
                                                 [ 6.6716e-01, -9.1791e-02, -3.5952e-01,  3.9863e-02, -1.5469e+00,
                                                   2.1700e-01,  1.6498e-01],
                                                 [-7.2039e-01,  1.1493e-02,  2.3626e+00, -4.9912e-03,  8.1287e-01,
                                                  -2.7170e-02, -4.8459e+00],
                                                 [ 7.3248e-01, -5.5497e-01,  1.1458e-01,  2.4101e-01, -1.9196e+00,
                                                   1.3120e+00, -1.0229e+00]],

                                                [[-3.4182e-01,  1.1900e+00,  8.3611e-01,  2.8846e+00, -4.0510e+00,
                                                  -9.4772e-01,  1.0865e+00],
                                                 [ 1.4758e+00, -2.1237e-01, -1.8803e+00, -5.3496e-01,  4.6082e-01,
                                                   1.7939e-01, -1.4723e+00],
                                                 [-1.3516e-01,  2.7538e-02,  3.2542e+00, -2.2157e+00,  2.0260e+00,
                                                   1.1385e+00, -1.4840e+00],
                                                 [ 4.9819e-01, -1.6209e-01,  5.9073e-01,  1.7106e+00, -2.2054e+00,
                                                   -9.3160e-01,  9.5501e-01],
                                                 [-6.2852e-01,  2.5481e-02,  2.8130e+00, -3.1329e-01, -1.6321e+00,
                                                   1.7254e-01, -3.2043e-02],
                                                 [ 5.2732e-01, -3.9340e-01,  1.2458e+00, -8.6716e-01,  8.2612e-01,
                                                   6.8973e-01, -1.0866e+00],
                                                 [-3.6248e-01, -9.5832e-01,  3.1090e-01,  5.2366e-01, -7.5841e-01,
                                                   2.4432e+00, -2.0746e+00],
                                                 [-5.8480e-01,  1.7588e-01,  1.6626e+00, -8.9840e-02,  1.3114e-01,
                                                  -4.3805e-01, -5.3029e+00],
                                                 [-6.0978e-01,  5.2804e-01, -1.5534e+00,  4.2050e-01, -6.5459e-02,
                                                  -1.7431e-01,  1.4335e+00]],

                                                [[-1.2076e+00,  8.0916e-01, -5.4423e-01, -2.5007e+00,  1.0022e+00,
                                                   1.3288e+00, -6.6113e-01],
                                                 [ 3.9347e+00, -1.1202e+00, -3.1875e+00, -2.2505e+00,  2.2782e+00,
                                                   6.9273e-01,  2.0858e+00],
                                                 [-1.5106e+00, -6.5883e-01, -5.7170e-01, -4.2417e+00,  1.6134e+00,
                                                   8.3112e-02,  1.5821e+00],
                                                 [-2.6352e-01, -7.2695e-01,  1.0058e+00, -6.7965e-01, -1.0057e+00,
                                                   7.6389e-01,  1.1598e-01],
                                                 [-6.1879e-01,  2.1381e-01,  2.7902e+00, -1.4168e+00, -2.4942e-01,
                                                   7.3121e-01, -1.2749e+00],
                                                 [-4.2349e-01,  5.1499e-01, -1.1291e+00, -2.4219e+00,  3.4505e-01,
                                                   1.3811e+00,  8.4791e-02],
                                                 [-1.1477e+00,  5.9410e-01,  6.0532e-01,  9.7264e-01, -1.9956e+00,
                                                   1.0271e+00, -2.4847e+00],
                                                 [-1.0974e-01,  7.6874e-01, -1.9692e+00, -5.5339e-01,  4.3769e+00,
                                                  -2.2300e+00, -5.5773e+00],
                                                 [-8.8625e-01,  2.1964e+00, -4.7814e-01, -9.1585e-01, -4.0720e-01,
                                                  -2.7260e+00,  8.1963e-01]],

                                                [[ 8.2934e-01,  1.7359e+00, -3.2661e+00, -1.4161e+00,  1.5997e+00,
                                                   -1.5997e+00,  8.0905e-01],
                                                 [ 2.9475e+00, -1.3245e+00, -3.7583e+00,  1.0867e+00,  1.7901e+00,
                                                   -5.6489e-01,  5.0130e-01],
                                                 [ 9.7242e-01, -2.6574e+00,  1.4017e+00,  2.1780e+00, -2.9905e+00,
                                                   -5.1923e-01, -1.2607e+00],
                                                 [ 2.8734e-02,  3.1358e-01,  1.3967e-01, -4.6944e-01,  5.4504e-01,
                                                   4.6453e-01, -6.5818e-01],
                                                 [-1.1399e+00,  6.3664e-01,  1.2467e-01, -6.2589e-02,  9.0363e-01,
                                                  -3.1530e-01, -3.4302e-01],
                                                 [ 3.2385e-01,  9.7602e-01, -3.1122e-01,  1.9145e-02, -1.4708e-01,
                                                   -6.4605e-01,  5.7041e-01],
                                                 [ 6.8437e-01,  1.7290e+00, -1.5806e+00, -1.2304e-01, -1.1253e-01,
                                                   1.0904e+00, -2.5999e+00],
                                                 [ 9.7942e-01, -2.9461e-01,  9.7261e-01, -1.0040e+00,  3.6000e+00,
                                                   -1.9349e+00, -1.2973e+00],
                                                 [ 1.1419e+00, -9.4346e-01,  1.4937e+00, -1.6367e+00,  7.3671e-01,
                                                   -3.5025e+00,  1.8482e+00]],

                                                [[-1.4373e-01, -1.0757e+00, -6.4308e-01,  8.7907e-01,  6.9455e-01,
                                                   5.4691e-01,  3.8665e-01],
                                                 [ 3.2868e+00, -1.8573e+00, -2.9573e+00,  1.5177e+00, -1.4491e+00,
                                                   9.4426e-01,  1.3803e-01],
                                                 [-1.0557e+00, -2.8052e+00,  6.4474e-01,  2.2923e+00,  7.1482e-01,
                                                   1.4262e+00,  1.1080e-01],
                                                 [ 3.1670e-03,  2.3703e-02,  4.1632e-02,  1.8617e-01,  9.4458e-02,
                                                   -2.7813e-01, -1.9663e-01],
                                                 [-7.2421e-02,  4.0924e-02, -5.6283e-01,  3.2143e-01,  8.9676e-01,
                                                  -4.8020e-01, -7.0194e-02],
                                                 [ 2.3262e-02,  6.1810e-02,  1.8750e-01,  4.8547e-01, -2.3523e-01,
                                                   -7.2527e-01, -5.6349e-02],
                                                 [-1.0908e-01, -8.1639e-01, -6.0349e-01, -1.9703e-01, -2.6626e-01,
                                                  -9.5024e-01, -6.7180e-01],
                                                 [ 2.4943e+00, -1.4095e+00,  3.9596e-01, -3.4019e-01,  2.8536e+00,
                                                   -1.6406e+00, -2.3982e-01],
                                                 [-8.0118e-01, -2.1289e+00, -3.5876e-01, -5.1380e-01, -9.7246e-01,
                                                  -2.4779e+00, -1.9252e-01]]]]).unsqueeze(0)

            let t3s2p1 = dsharp.convTranspose3d(t1, t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[[ 5.1107e-01,  3.3656e+00,  1.4016e+00,  2.8882e-01, -4.8976e-01],
                                                 [-1.2520e+00, -2.9655e+00, -3.2308e+00, -4.6937e+00,  1.0909e+00],
                                                 [-4.1827e-02,  1.6104e+00,  7.9930e-01, -1.8915e-01, -4.5081e-01],
                                                 [ 1.7161e-03,  9.5375e-01, -1.9002e+00, -1.4199e+00,  1.3326e+00],
                                                 [-4.5080e-01, -1.3973e-02,  1.9577e-01,  2.3804e+00,  1.0657e+00]],

                                                [[-4.3149e-01, -4.6548e+00,  2.9759e+00, -1.9294e+00, -1.7011e+00],
                                                 [ 1.3335e+00, -1.6401e+00,  4.0610e-01,  1.7460e+00,  7.5368e-01],
                                                 [ 7.4535e-01,  2.1371e-01,  9.7747e-01, -1.8335e+00, -9.1553e-01],
                                                 [-1.0580e+00, -1.1396e+00,  1.5921e+00,  1.4837e+00,  1.2120e+00],
                                                 [-6.2202e-01,  1.7381e+00, -9.0946e-01, -1.8453e+00, -4.7914e-01]],

                                                [[ 5.2215e-01, -4.0141e-01, -4.2916e-01,  5.2209e-01,  4.4587e-01],
                                                 [-1.3496e+00,  2.0044e+00,  1.0607e+00, -7.2726e-01, -1.0577e+00],
                                                 [-3.6046e-01, -1.7473e-01,  7.7268e-02,  3.1800e-01,  1.1941e-01],
                                                 [ 8.7386e-01,  1.6961e-01, -6.9657e-02, -3.3020e+00, -9.0590e-02],
                                                 [-1.1925e-02,  1.0887e+00,  5.3302e-01,  2.3523e+00,  9.0061e-01]],

                                                [[-1.6330e+00, -4.5649e+00,  1.3309e+00,  2.1396e+00,  1.8538e+00],
                                                 [ 3.0736e+00,  2.4374e+00, -2.8052e+00, -3.0569e+00, -6.2395e-01],
                                                 [-4.6614e-01,  7.7661e-01,  5.2383e-01,  4.1591e-01, -4.4464e-01],
                                                 [ 6.0249e-01,  7.9457e-01, -5.6395e-02, -1.9356e+00,  2.1329e+00],
                                                 [-1.8267e+00, -2.1844e+00,  3.3026e-01, -1.7905e+00, -8.1025e-01]],

                                                [[ 1.0669e+00, -8.0981e-01, -8.7187e-01, -1.9274e+00, -5.4243e-01],
                                                 [-2.5996e+00,  5.0024e-01,  2.0791e+00,  2.7528e+00,  1.3802e+00],
                                                 [-2.3509e-02, -4.0277e-01, -1.8465e-01,  3.9798e-01,  2.7585e-01],
                                                 [ 2.3677e-01, -3.9000e+00,  4.9234e-01, -1.5508e+00, -4.6172e-01],
                                                 [ 8.0970e-01,  1.1538e+00,  1.9542e-01,  2.1257e+00,  9.4246e-01]]],


                                               [[[-1.9905e-01,  1.1797e+00, -5.4590e-01, -2.2031e+00,  1.9075e-01],
                                                 [-7.0397e-01,  3.8190e+00,  7.4930e-03,  2.9795e-01, -3.6628e-01],
                                                 [ 1.6291e-02,  1.0505e+00, -3.1132e-01, -1.4716e+00,  1.7558e-01],
                                                 [-9.0554e-01,  2.1269e+00, -5.4716e-01,  1.5862e+00,  2.8025e+00],
                                                 [ 1.7558e-01,  7.5659e-01, -7.6251e-02,  1.1156e+00, -4.1509e-01]],

                                                [[-9.0297e-01,  1.1275e+00, -2.4289e+00,  7.9854e-01,  8.4120e-01],
                                                 [-7.3211e-01, -1.5431e+00, -5.0035e+00, -7.8804e-01,  2.3154e+00],
                                                 [ 8.2002e-02,  4.2354e+00, -1.3934e+00, -1.7526e+00,  7.8171e-01],
                                                 [ 1.1831e-01, -1.0576e+00, -2.0953e+00, -1.5189e+00,  4.4112e-01],
                                                 [ 7.8505e-01,  1.6238e+00, -3.5439e-01,  2.2488e+00, -1.8782e+00]],

                                                [[-2.0337e-01, -1.1304e+00,  1.6715e-01,  1.0459e+00, -1.7366e-01],
                                                 [-1.3930e+00,  1.3409e+00,  6.8051e-01, -3.6292e-01, -2.8358e-01],
                                                 [ 1.4039e-01,  5.0549e-01, -3.0095e-02,  6.2284e-02, -4.6508e-02],
                                                 [ 4.0867e-01, -1.9516e-01,  1.0372e+00, -1.4634e-02,  1.7659e+00],
                                                 [ 4.6446e-03,  6.8244e-01, -2.0760e-01,  5.2616e-01, -3.5078e-01]],

                                                [[-9.3616e-01, -7.6996e+00,  7.6930e-01,  4.5803e+00, -7.6233e-01],
                                                 [ 2.9930e-01,  1.2267e+00, -6.0504e-02,  8.5671e-01, -1.9043e+00],
                                                 [ 6.2808e-01,  1.4158e-01, -1.2980e-01,  1.6286e+00, -2.1490e-01],
                                                 [ 1.0957e+00,  9.2596e-02, -9.2809e-01, -2.2707e+00, -6.9050e-01],
                                                 [ 1.1544e-04, -1.7244e+00, -9.3285e-01,  3.2562e+00, -1.5918e+00]],

                                                [[-4.1556e-01, -2.3034e+00,  3.3958e-01,  3.0738e-01,  2.1127e-01],
                                                 [-1.3344e+00,  9.7688e-01,  6.5812e-01, -2.0208e+00,  1.2380e+00],
                                                 [ 9.1565e-03, -2.0508e-01,  7.1917e-02,  5.5545e-01, -1.0744e-01],
                                                 [ 1.7452e+00, -1.8257e+00,  6.3668e-01,  3.7559e-01,  1.6663e+00],
                                                 [-3.1537e-01, -6.7241e-01, -7.6114e-02,  9.6129e-01, -3.6708e-01]]]]).unsqueeze(0)

            let t3p1d2 = dsharp.convTranspose3d(t1, t2, padding=1, dilation=2)
            let t3p1d2Correct = combo.tensor([[[[-1.0245e-01, -5.9647e-01,  9.3921e-02, -7.5587e-01,  1.9314e-01],
                                                 [-1.2189e+00, -1.8433e+00,  1.6070e+00, -1.1514e+00,  2.3350e+00],
                                                 [ 9.2224e-02,  9.4806e-01, -1.7268e-01,  5.7531e-01, -1.8053e-01],
                                                 [ 4.2969e-01,  2.6431e+00, -2.2818e+00, -5.1769e-01, -1.5198e+00],
                                                 [ 3.7179e-02, -8.5859e-01,  1.9636e-01,  7.2871e-02,  4.9428e-02]],

                                                [[ 1.5057e-02, -2.7401e-01, -4.7146e-02,  3.9273e-01, -4.5927e+00],
                                                 [ 1.5358e+00, -4.2029e+00,  3.6310e+00,  4.4393e+00,  2.8129e+00],
                                                 [ 1.4288e+00, -5.9017e-01,  1.2119e+00, -1.0511e+00,  1.1724e+00],
                                                 [-2.1768e+00,  3.1079e+00, -5.8648e+00, -3.4127e+00, -2.3617e+00],
                                                 [-1.4544e+00,  9.4486e-01, -2.4129e+00,  1.8278e+00, -9.9002e-01]],

                                                [[-2.2211e-02, -1.1428e-01,  1.7139e-02,  1.8895e+00, -3.9936e-01],
                                                 [-1.0387e+00,  1.1806e+00, -3.1093e-01,  1.1913e+00, -3.1527e+00],
                                                 [ 1.5942e-01, -1.1409e-01,  7.7268e-02, -2.1475e-01,  7.1630e-02],
                                                 [ 1.8329e+00, -1.8513e-01,  1.5766e+00, -7.6421e-01,  9.6227e-01],
                                                 [-1.3200e-01,  6.7251e-01, -1.8789e-01,  1.0284e-01, -8.4286e-02]],

                                                [[-1.9183e-01,  4.5235e-02,  1.1921e-01,  1.6477e-01, -4.2937e-01],
                                                 [-3.3870e+00,  6.8932e-01,  1.2275e+00, -4.6907e+00, -6.1358e+00],
                                                 [-1.2204e+00,  9.5888e-01,  9.6550e-01, -2.7589e-01, -2.4401e+00],
                                                 [ 1.8659e-01,  2.9610e-01,  3.8398e+00,  5.1360e+00,  3.0689e+00],
                                                 [-5.4028e-01,  3.1447e-02,  1.1577e+00, -1.1192e+00,  1.6228e+00]],

                                                [[-2.3675e-02, -1.0882e-01,  1.5483e-02,  6.4794e-01, -1.3376e-01],
                                                 [ 2.8738e-01,  4.7778e-03, -5.1073e-01, -6.3953e-01,  2.9550e-01],
                                                 [-8.1145e-02, -6.4408e-01,  1.1118e-01,  1.1950e+00, -2.1934e-01],
                                                 [-1.0031e-01, -9.7736e-01,  3.8649e-01,  2.6536e+00, -2.2762e+00],
                                                 [-8.2723e-02, -4.4742e-01,  6.8506e-02, -5.3513e-01,  1.3741e-01]]],


                                               [[[-1.1435e-01, -2.5012e-01,  1.5733e-02, -3.1487e-02,  1.1962e-02],
                                                 [-1.4747e+00, -6.3610e-01,  1.1947e-01, -2.2041e+00,  2.0339e+00],
                                                 [ 1.2348e-01,  2.0001e-01, -1.9699e-03,  1.6360e+00, -3.5134e-01],
                                                 [ 1.5491e+00,  1.3579e+00, -5.4192e-01,  7.8988e-02, -2.0117e+00],
                                                 [-1.2555e-01, -6.3778e-01,  9.5123e-02,  4.9298e-01, -7.4163e-02]],

                                                [[ 9.4726e-01, -8.8186e-01,  1.6573e+00, -8.1649e-01, -1.6891e+00],
                                                 [ 2.4154e-01, -2.9949e+00, -6.8751e-02, -1.7125e+00,  1.9995e+00],
                                                 [ 7.0803e-01, -1.0804e-01, -3.0661e-01,  3.2285e-01, -6.6276e-02],
                                                 [ 3.9308e+00,  3.8945e+00, -2.8420e+00, -3.6102e+00,  2.7485e+00],
                                                 [ 1.4156e+00, -1.0560e+00, -1.1896e+00,  8.1470e-01,  1.9007e+00]],

                                                [[ 6.5155e-02, -6.6365e-01,  1.6384e-01,  1.0020e+00, -1.6052e-01],
                                                 [-8.9127e-02,  1.0903e+00,  1.2974e+00,  1.1690e+00, -6.2094e-01],
                                                 [ 9.6969e-02,  2.9025e-01, -3.0095e-02,  3.6201e-01, -8.7570e-02],
                                                 [ 6.9967e-02,  1.1285e+00,  3.0907e-01,  2.5184e-01, -1.5294e+00],
                                                 [ 1.0784e-01,  6.0061e-01, -9.3024e-02, -9.2099e-01,  1.6661e-01]],

                                                [[-2.6337e-01,  2.8566e-01, -1.1974e+00,  1.1181e+00, -1.8616e-01],
                                                 [ 3.3619e+00, -6.7208e-01, -1.8833e+00, -7.1228e-01,  8.5941e-02],
                                                 [ 2.2306e+00, -9.6057e-01, -1.3195e+00,  7.1460e-01,  5.8736e-03],
                                                 [-1.7553e+00,  2.0345e+00, -3.0523e+00,  1.1117e+00, -2.4375e+00],
                                                 [-1.0486e+00,  9.6933e-01, -1.8792e+00,  8.1101e-01, -5.6141e-01]],

                                                [[-1.0409e-02,  3.4735e-01, -7.7906e-02,  1.3655e-01, -5.5077e-02],
                                                 [-1.3938e+00,  3.5525e-01,  2.0966e-01, -1.8086e+00, -2.7074e-01],
                                                 [ 2.3803e-01,  9.9532e-01, -1.3451e-01, -1.1614e-01, -1.9662e-02],
                                                 [ 2.0666e+00,  9.8112e-01,  2.0048e-01, -2.8437e+00, -4.7968e-02],
                                                 [-7.6454e-02,  8.2957e-01, -2.0315e-01, -2.4032e-01, -1.5784e-02]]]]).unsqueeze(0)

            Assert.True(t3.allclose(t3Correct, 0.01, 0.01))
            Assert.True(t3p1.allclose(t3p1Correct, 0.01, 0.01))
            Assert.True(t3p122.allclose(t3p122Correct, 0.01, 0.01))
            Assert.True(t3s2.allclose(t3s2Correct, 0.01, 0.01))
            Assert.True(t3s132.allclose(t3s132Correct, 0.01, 0.01))
            Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01, 0.01))
            Assert.True(t3p1d2.allclose(t3p1d2Correct, 0.01, 0.01))


