// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Util
open System

[<TestFixture>]
type TestTensor () =

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

        let t5 = combo.tensor([[[[[ofDouble 1.; ofDouble 2.]]]]])
        let t5ShapeCorrect = [|1; 1; 1; 1; 2|]
        let t5DimCorrect = 5

        Assert.CheckEqual(t5ShapeCorrect, t5.shape)
        Assert.CheckEqual(t5DimCorrect, t5.dim)
        Assert.CheckEqual(combo.dtype, t5.dtype)

        let t6 = combo.tensor([[[[[[ofDouble 1.; ofDouble 2.]]]]]])
        let t6ShapeCorrect = [|1; 1; 1; 1; 1; 2|]
        let t6DimCorrect = 6

        Assert.CheckEqual(t6ShapeCorrect, t6.shape)
        Assert.CheckEqual(t6DimCorrect, t6.dim)
        Assert.CheckEqual(combo.dtype, t6.dtype)

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
    member _.TestTensorCreateDtypeInferredFromData () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let dataFloat32 = [1.f;2.f;3.f]
            let tensorFloat32 = dsharp.tensor(dataFloat32, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorFloat32.dtype, Dtype.Float32)

            // Exception: If data is double and no dtype is given by the user, prefer a Float32 tensor
            let dataFloat64 = [1.;2.;3.]
            let tensorFloat64 = dsharp.tensor(dataFloat64, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorFloat64.dtype, Dtype.Float32)

            let dataInt64 = [1L;2L;3L]
            let tensorInt64 = dsharp.tensor(dataInt64, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorInt64.dtype, Dtype.Int64)

            let dataInt32 = [1;2;3]
            let tensorInt32 = dsharp.tensor(dataInt32, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorInt32.dtype, Dtype.Int32)

            let dataInt16 = [1s;2s;3s]
            let tensorInt16 = dsharp.tensor(dataInt16, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorInt16.dtype, Dtype.Int16)

            let dataInt8 = [1y;2y;3y]
            let tensorInt8 = dsharp.tensor(dataInt8, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorInt8.dtype, Dtype.Int8)

            let dataByte = [1uy;2uy;3uy]
            let tensorByte = dsharp.tensor(dataByte, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorByte.dtype, Dtype.Byte)

            let dataBool = [true;true;false]
            let tensorBool = dsharp.tensor(dataBool, ?dtype=None, device=combo.device, backend=combo.backend)
            Assert.AreEqual(tensorBool.dtype, Dtype.Bool)

    [<Test>]
    member _.TestTensorHandle () =
        for combo in Combos.Float32 do
           if combo.backend = Backend.Reference then
               let t1 = combo.tensor([1.0f ; 1.0f ])
               Assert.CheckEqual([| 1.0f ; 1.0f |], (t1.primalRaw.Handle :?> float32[]))

    [<Test>]
    member _.TestTensorCreate0 () =
      for combo in Combos.AllDevicesAndBackendsFloat32 do
        let t0 = combo.tensor(1.)
        let t0Shape = t0.shape
        let t0Dim = t0.dim
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.CheckEqual(t0DimCorrect, t0Dim)
        Assert.CheckEqual(t0ShapeCorrect, t0Shape)

    [<Test>]
    member _.TestTensorCreate1 () =
      for combo in Combos.AllDevicesAndBackendsFloat32 do
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
      for combo in Combos.AllDevicesAndBackendsFloat32 do
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
      for combo in Combos.AllDevicesAndBackendsFloat32 do
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
      for combo in Combos.AllDevicesAndBackendsFloat32 do
        let t4Values = [[[[1.; 2.]]]]
        let t4 = combo.tensor(t4Values)
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4
        let t4ValuesCorrect = array4D (List.map (List.map (List.map (List.map float32))) t4Values)

        Assert.CheckEqual(t4ShapeCorrect, t4.shape)
        Assert.CheckEqual(t4DimCorrect, t4.dim)
        Assert.CheckEqual(t4ValuesCorrect, t4.toArray() :?> float32[,,,])

    [<Test>]
    member _.TestTensorCreate5 () =
      for combo in Combos.AllDevicesAndBackendsFloat32 do
        let t5Values = [[[[[1.; 2.]]]]]
        let t5 = combo.tensor(t5Values)
        let t5ShapeCorrect = [|1; 1; 1; 1; 2|]
        let t5DimCorrect = 5
        let t5ValuesCorrect = array5D (List.map (List.map (List.map (List.map (List.map float32)))) t5Values)

        Assert.CheckEqual(t5ShapeCorrect, t5.shape)
        Assert.CheckEqual(t5DimCorrect, t5.dim)
        Assert.CheckEqual(t5ValuesCorrect, t5.toArray())

    [<Test>]
    member _.TestTensorCreate6 () =
      for combo in Combos.AllDevicesAndBackendsFloat32 do
        let t6Values = [[[[[[1.; 2.]]]]]]
        let t6 = combo.tensor(t6Values)
        let t6ShapeCorrect = [|1; 1; 1; 1; 1; 2|]
        let t6DimCorrect = 6
        let t6ValuesCorrect = array6D (List.map (List.map (List.map (List.map (List.map (List.map float32))))) t6Values)

        Assert.CheckEqual(t6ShapeCorrect, t6.shape)
        Assert.CheckEqual(t6DimCorrect, t6.dim)
        Assert.CheckEqual(t6ValuesCorrect, t6.toArray())

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
    member this.TestTensorCreateFromTensor5 () =
        let t5Values = [[[[[dsharp.tensor 1.; dsharp.tensor 2.]]]]]
        let t5 = dsharp.tensor(t5Values)
        let t5ShapeCorrect = [|1; 1; 1; 1; 2|]
        let t5DimCorrect = 5
        let t5ValuesCorrect = array5D (List.map (List.map (List.map (List.map (List.map float32)))) t5Values)

        Assert.AreEqual(t5ShapeCorrect, t5.shape)
        Assert.AreEqual(t5DimCorrect, t5.dim)
        Assert.AreEqual(t5ValuesCorrect, t5.toArray())

    [<Test>]
    member this.TestTensorCreateFromTensor6 () =
        let t6Values = [[[[[[dsharp.tensor 1.; dsharp.tensor 2.]]]]]]
        let t6 = dsharp.tensor(t6Values)
        let t6ShapeCorrect = [|1; 1; 1; 1; 1; 2|]
        let t6DimCorrect = 6
        let t6ValuesCorrect = array6D (List.map (List.map (List.map (List.map (List.map (List.map float32))))) t6Values)

        Assert.AreEqual(t6ShapeCorrect, t6.shape)
        Assert.AreEqual(t6DimCorrect, t6.dim)
        Assert.AreEqual(t6ValuesCorrect, t6.toArray())

    [<Test>]
    member _.TensorCreateDistinguishByteAndInt8 () =
        let v1 = [|10uy; 25uy; 125uy; 220uy|] // Byte
        let t1 = dsharp.tensor(v1, dtype=Dtype.Float32)
        let a1 = t1.toArray1D<float32>()
        let a1Correct = [|10.; 25.; 125.; 220.|]
        Assert.AreEqual(a1Correct, a1)

        let v2 = [|10y; 25y; 125y; -20y|] // Int8
        let t2 = dsharp.tensor(v2, dtype=Dtype.Float32)
        let a2 = t2.toArray1D<float32>() // The result becomes [|10.; 25.; 125.; 236.|] when Int8 is confused with Byte
        let a2Correct = [|10.; 25.; 125.; -20.|]
        Assert.AreEqual(a2Correct, a2)

    [<Test>]
    member _.TestTensorToArray () =
        for combo in Combos.All do 
            let a = array2D [[1.; 2.]; [3.; 4.]]
            let t = combo.tensor(a)
            let tArray = t.toArray()
            let tArrayCorrect = combo.arrayCreator2D a
            Assert.CheckEqual(tArrayCorrect, tArray)

            let a1 = [|0.; 1.; 0.|]
            let t1 = combo.tensor(a1)
            let t1aFloat32 = t1.toArray1D<float32>()
            let t1aFloat32Correct = a1 |> Array.map (fun v -> System.Convert.ToSingle(v))
            Assert.AreEqual(t1aFloat32Correct, t1aFloat32)

            let t1aFloat64 = t1.toArray1D<double>()
            let t1aFloat64Correct = a1 |> Array.map (fun v -> System.Convert.ToDouble(v))
            Assert.AreEqual(t1aFloat64Correct, t1aFloat64)

            let t1aInt64 = t1.toArray1D<int64>()
            let t1aInt64Correct = a1 |> Array.map (fun v -> System.Convert.ToInt64(v))
            Assert.AreEqual(t1aInt64Correct, t1aInt64)

            let t1aInt32 = t1.toArray1D<int32>()
            let t1aInt32Correct = a1 |> Array.map (fun v -> System.Convert.ToInt32(v))
            Assert.AreEqual(t1aInt32Correct, t1aInt32)

            let t1aInt16 = t1.toArray1D<int16>()
            let t1aInt16Correct = a1 |> Array.map (fun v -> System.Convert.ToInt16(v))
            Assert.AreEqual(t1aInt16Correct, t1aInt16)

            let t1aInt8 = t1.toArray1D<int8>()
            let t1aInt8Correct = a1 |> Array.map (fun v -> System.Convert.ToSByte(v))
            Assert.AreEqual(t1aInt8Correct, t1aInt8)

            let t1aByte = t1.toArray1D<byte>()
            let t1aByteCorrect = a1 |> Array.map (fun v -> System.Convert.ToByte(v))
            Assert.AreEqual(t1aByteCorrect, t1aByte)

            let t1aBool = t1.toArray1D<bool>()
            let t1aBoolCorrect = a1 |> Array.map (fun v -> System.Convert.ToBoolean(v))
            Assert.AreEqual(t1aBoolCorrect, t1aBool)


            let a2 = array2D [[0.; 1.; 0.]; [1.; 0.; 1.]]
            let t2 = combo.tensor(a2)
            let t2aFloat32 = t2.toArray2D<float32>()
            let t2aFloat32Correct = a2 |> Array2D.map (fun v -> System.Convert.ToSingle(v))
            Assert.AreEqual(t2aFloat32Correct, t2aFloat32)

            let t2aFloat64 = t2.toArray2D<double>()
            let t2aFloat64Correct = a2 |> Array2D.map (fun v -> System.Convert.ToDouble(v))
            Assert.AreEqual(t2aFloat64Correct, t2aFloat64)

            let t2aInt64 = t2.toArray2D<int64>()
            let t2aInt64Correct = a2 |> Array2D.map (fun v -> System.Convert.ToInt64(v))
            Assert.AreEqual(t2aInt64Correct, t2aInt64)

            let t2aInt32 = t2.toArray2D<int32>()
            let t2aInt32Correct = a2 |> Array2D.map (fun v -> System.Convert.ToInt32(v))
            Assert.AreEqual(t2aInt32Correct, t2aInt32)

            let t2aInt16 = t2.toArray2D<int16>()
            let t2aInt16Correct = a2 |> Array2D.map (fun v -> System.Convert.ToInt16(v))
            Assert.AreEqual(t2aInt16Correct, t2aInt16)

            let t2aInt8 = t2.toArray2D<int8>()
            let t2aInt8Correct = a2 |> Array2D.map (fun v -> System.Convert.ToSByte(v))
            Assert.AreEqual(t2aInt8Correct, t2aInt8)

            let t2aByte = t2.toArray2D<byte>()
            let t2aByteCorrect = a2 |> Array2D.map (fun v -> System.Convert.ToByte(v))
            Assert.AreEqual(t2aByteCorrect, t2aByte)

            let t2aBool = t2.toArray2D<bool>()
            let t2aBoolCorrect = a2 |> Array2D.map (fun v -> System.Convert.ToBoolean(v))
            Assert.AreEqual(t2aBoolCorrect, t2aBool)


            let a3 = array3D [[[0.; 1.; 0.]; [1.; 0.; 1.]]]
            let t3 = combo.tensor(a3)
            let t3aFloat32 = t3.toArray3D<float32>()
            let t3aFloat32Correct = a3 |> Array3D.map (fun v -> System.Convert.ToSingle(v))
            Assert.AreEqual(t3aFloat32Correct, t3aFloat32)

            let t3aFloat64 = t3.toArray3D<double>()
            let t3aFloat64Correct = a3 |> Array3D.map (fun v -> System.Convert.ToDouble(v))
            Assert.AreEqual(t3aFloat64Correct, t3aFloat64)

            let t3aInt64 = t3.toArray3D<int64>()
            let t3aInt64Correct = a3 |> Array3D.map (fun v -> System.Convert.ToInt64(v))
            Assert.AreEqual(t3aInt64Correct, t3aInt64)

            let t3aInt32 = t3.toArray3D<int32>()
            let t3aInt32Correct = a3 |> Array3D.map (fun v -> System.Convert.ToInt32(v))
            Assert.AreEqual(t3aInt32Correct, t3aInt32)

            let t3aInt16 = t3.toArray3D<int16>()
            let t3aInt16Correct = a3 |> Array3D.map (fun v -> System.Convert.ToInt16(v))
            Assert.AreEqual(t3aInt16Correct, t3aInt16)

            let t3aInt8 = t3.toArray3D<int8>()
            let t3aInt8Correct = a3 |> Array3D.map (fun v -> System.Convert.ToSByte(v))
            Assert.AreEqual(t3aInt8Correct, t3aInt8)

            let t3aByte = t3.toArray3D<byte>()
            let t3aByteCorrect = a3 |> Array3D.map (fun v -> System.Convert.ToByte(v))
            Assert.AreEqual(t3aByteCorrect, t3aByte)

            let t3aBool = t3.toArray3D<bool>()
            let t3aBoolCorrect = a3 |> Array3D.map (fun v -> System.Convert.ToBoolean(v))
            Assert.AreEqual(t3aBoolCorrect, t3aBool)


            let a4 = array4D [[[[0.; 1.; 0.]; [1.; 0.; 1.]]]]
            let t4 = combo.tensor(a4)
            let t4aFloat32 = t4.toArray4D<float32>()
            let t4aFloat32Correct = a4 |> Array4D.map (fun v -> System.Convert.ToSingle(v))
            Assert.AreEqual(t4aFloat32Correct, t4aFloat32)

            let t4aFloat64 = t4.toArray4D<double>()
            let t4aFloat64Correct = a4 |> Array4D.map (fun v -> System.Convert.ToDouble(v))
            Assert.AreEqual(t4aFloat64Correct, t4aFloat64)

            let t4aInt64 = t4.toArray4D<int64>()
            let t4aInt64Correct = a4 |> Array4D.map (fun v -> System.Convert.ToInt64(v))
            Assert.AreEqual(t4aInt64Correct, t4aInt64)

            let t4aInt32 = t4.toArray4D<int32>()
            let t4aInt32Correct = a4 |> Array4D.map (fun v -> System.Convert.ToInt32(v))
            Assert.AreEqual(t4aInt32Correct, t4aInt32)

            let t4aInt16 = t4.toArray4D<int16>()
            let t4aInt16Correct = a4 |> Array4D.map (fun v -> System.Convert.ToInt16(v))
            Assert.AreEqual(t4aInt16Correct, t4aInt16)

            let t4aInt8 = t4.toArray4D<int8>()
            let t4aInt8Correct = a4 |> Array4D.map (fun v -> System.Convert.ToSByte(v))
            Assert.AreEqual(t4aInt8Correct, t4aInt8)

            let t4aByte = t4.toArray4D<byte>()
            let t4aByteCorrect = a4 |> Array4D.map (fun v -> System.Convert.ToByte(v))
            Assert.AreEqual(t4aByteCorrect, t4aByte)

            let t4aBool = t4.toArray4D<bool>()
            let t4aBoolCorrect = a4 |> Array4D.map (fun v -> System.Convert.ToBoolean(v))
            Assert.AreEqual(t4aBoolCorrect, t4aBool)


            let a5 = array5D [[[[[0.; 1.; 0.]; [1.; 0.; 1.]]]]]
            let t5 = combo.tensor(a5)
            let t5aFloat32 = t5.toArray5D<float32>()
            let t5aFloat32Correct = a5 |> Array5D.map (fun v -> System.Convert.ToSingle(v))
            Assert.AreEqual(t5aFloat32Correct, t5aFloat32)

            let t5aFloat64 = t5.toArray5D<double>()
            let t5aFloat64Correct = a5 |> Array5D.map (fun v -> System.Convert.ToDouble(v))
            Assert.AreEqual(t5aFloat64Correct, t5aFloat64)

            let t5aInt64 = t5.toArray5D<int64>()
            let t5aInt64Correct = a5 |> Array5D.map (fun v -> System.Convert.ToInt64(v))
            Assert.AreEqual(t5aInt64Correct, t5aInt64)

            let t5aInt32 = t5.toArray5D<int32>()
            let t5aInt32Correct = a5 |> Array5D.map (fun v -> System.Convert.ToInt32(v))
            Assert.AreEqual(t5aInt32Correct, t5aInt32)

            let t5aInt16 = t5.toArray5D<int16>()
            let t5aInt16Correct = a5 |> Array5D.map (fun v -> System.Convert.ToInt16(v))
            Assert.AreEqual(t5aInt16Correct, t5aInt16)

            let t5aInt8 = t5.toArray5D<int8>()
            let t5aInt8Correct = a5 |> Array5D.map (fun v -> System.Convert.ToSByte(v))
            Assert.AreEqual(t5aInt8Correct, t5aInt8)

            let t5aByte = t5.toArray5D<byte>()
            let t5aByteCorrect = a5 |> Array5D.map (fun v -> System.Convert.ToByte(v))
            Assert.AreEqual(t5aByteCorrect, t5aByte)

            let t5aBool = t5.toArray5D<bool>()
            let t5aBoolCorrect = a5 |> Array5D.map (fun v -> System.Convert.ToBoolean(v))
            Assert.AreEqual(t5aBoolCorrect, t5aBool)


            let a6 = array6D [[[[[[0.; 1.; 0.]; [1.; 0.; 1.]]]]]]
            let t6 = combo.tensor(a6)
            let t6aFloat32 = t6.toArray6D<float32>()
            let t6aFloat32Correct = a6 |> Array6D.map (fun v -> System.Convert.ToSingle(v))
            Assert.AreEqual(t6aFloat32Correct, t6aFloat32)

            let t6aFloat64 = t6.toArray6D<double>()
            let t6aFloat64Correct = a6 |> Array6D.map (fun v -> System.Convert.ToDouble(v))
            Assert.AreEqual(t6aFloat64Correct, t6aFloat64)

            let t6aInt64 = t6.toArray6D<int64>()
            let t6aInt64Correct = a6 |> Array6D.map (fun v -> System.Convert.ToInt64(v))
            Assert.AreEqual(t6aInt64Correct, t6aInt64)

            let t6aInt32 = t6.toArray6D<int32>()
            let t6aInt32Correct = a6 |> Array6D.map (fun v -> System.Convert.ToInt32(v))
            Assert.AreEqual(t6aInt32Correct, t6aInt32)

            let t6aInt16 = t6.toArray6D<int16>()
            let t6aInt16Correct = a6 |> Array6D.map (fun v -> System.Convert.ToInt16(v))
            Assert.AreEqual(t6aInt16Correct, t6aInt16)

            let t6aInt8 = t6.toArray6D<int8>()
            let t6aInt8Correct = a6 |> Array6D.map (fun v -> System.Convert.ToSByte(v))
            Assert.AreEqual(t6aInt8Correct, t6aInt8)

            let t6aByte = t6.toArray6D<byte>()
            let t6aByteCorrect = a6 |> Array6D.map (fun v -> System.Convert.ToByte(v))
            Assert.AreEqual(t6aByteCorrect, t6aByte)

            let t6aBool = t6.toArray6D<bool>()
            let t6aBoolCorrect = a6 |> Array6D.map (fun v -> System.Convert.ToBoolean(v))
            Assert.AreEqual(t6aBoolCorrect, t6aBool)

    [<Test>]
    member _.TestTensorSaveSaveAndLoadToSpecificConfiguration () =
        let fileName = System.IO.Path.GetTempFileName()
        for combo in Combos.All do 
            let a = combo.tensor([[1,2],[3,4]])
            a.save(fileName)
            let b = Tensor.load(fileName, device=combo.device, dtype=combo.dtype, backend=combo.backend)
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
            let b = Tensor.load(fileName, device=combo.device, dtype=combo.dtype, backend=combo.backend)
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
            // printfn "%A" i
            System.GC.Collect()
        Assert.True(true)


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
            // Empty tensor with no data
            // This is not a scalar, and shape [|0|] signifies the absence of data and dimensions
            let tvoid = combo.empty()
            Assert.CheckEqual(tvoid.shape, ([|0|]: int32[]) )
            Assert.CheckEqual(tvoid.dtype, combo.dtype)

            // Scalar (zero-dimensional) tensor
            // Shape [||] siginifes zero dimensions
            // Tensor data is not initialized and can be random or zero depending on the backend used
            let t0 = combo.empty([])
            Assert.CheckEqual(t0.shape, ([| |]: int32[]) )
            Assert.CheckEqual(t0.dtype, combo.dtype)

            // Vector
            // Tensor data is not initialized and can be random or zero depending on the backend used
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
    member _.TestTensorConvertViaIConvertible () =
        for combo in Combos.IntegralAndFloatingPoint do
            let v = 2.
            let t = combo.tensor(v)
            let tsingle = Convert.ToSingle t
            let tdouble = Convert.ToDouble t
            let tint16 = Convert.ToInt16 t
            let tint32 = Convert.ToInt32 t
            let tint64 = Convert.ToInt64 t
            let tsingleCorrect = Convert.ToSingle v
            let tdoubleCorrect = Convert.ToDouble v
            let tint16Correct = Convert.ToInt16 v
            let tint32Correct = Convert.ToInt32 v
            let tint64Correct = Convert.ToInt64 v
            Assert.CheckEqual(tsingleCorrect, tsingle)
            Assert.CheckEqual(tdoubleCorrect, tdouble)
            Assert.CheckEqual(tint16Correct, tint16)
            Assert.CheckEqual(tint32Correct, tint32)
            Assert.CheckEqual(tint64Correct, tint64)

            let t2 = combo.full([4], t) // You can use a scalar tensor as a scalar and the types are used correctly
            let t2Correct = combo.tensor([2.; 2.; 2.; 2. ])
            Assert.CheckEqual(t2, t2Correct)

            let t3 = t2 + (t :> scalar)  // You can use a scalar tensor as a scalar and the types are used correctly
            let t3Correct = combo.tensor([4.; 4.; 4.; 4. ])
            Assert.CheckEqual(t3, t3Correct)

    [<Test>]
    member _.TestTensorToScalar () =
        for combo in Combos.All do
            let t = 1.
            let t0 = combo.tensor(t)
            let t1 = combo.tensor([t])
            let t2 = combo.tensor([[t]])
            let t3 = combo.tensor([[[t]]])

            let t0s = float t0
            let t1s = float t1
            let t2s = float t2
            let t3s = float t3

            Assert.CheckEqual(t, t0s)
            Assert.CheckEqual(t, t1s)
            Assert.CheckEqual(t, t2s)
            Assert.CheckEqual(t, t3s)

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
        Assert.CheckEqual(2, t1slice1.Shape[0])

        Assert.CheckEqual(3, (t1slice2.GetItem(0) |> Convert.ToInt32))
        Assert.CheckEqual(1, t1slice2.Dim)
        Assert.CheckEqual(1, t1slice2.Shape[0])

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
        let t1_0 = t1[0]
        let t1_1 = t1[1]
        let t1_0_s = t1_0.toDouble()
        let t1_1_s = t1_1.toDouble()
        Assert.CheckEqual(2.0, t1_0_s)
        Assert.CheckEqual(3.0, t1_1_s)
        Assert.CheckEqual(4.0, (t1[2].toDouble()))
        Assert.CheckEqual(5.0, (t1[3].toDouble()))

        let t2 = combo.tensor([[2.]; [3.]])
        Assert.CheckEqual(2.0, (t2[0,0].toDouble()))
        Assert.CheckEqual(3.0, (t2[1,0].toDouble()))

        let t2b = combo.tensor([[1.;2.]; [3.;4.]])
        Assert.CheckEqual(1.0, (t2b[0,0].toDouble()))
        Assert.CheckEqual(2.0, (t2b[0,1].toDouble()))
        Assert.CheckEqual(3.0, (t2b[1,0].toDouble()))
        Assert.CheckEqual(4.0, (t2b[1,1].toDouble()))

        let t3 = combo.tensor([[[2.; 3.]]])
        Assert.CheckEqual(2.0, (t3[0,0,0].toDouble()))
        Assert.CheckEqual(3.0, (t3[0,0,1].toDouble()))

        let t4 = combo.tensor([[[[1.]]]])
        Assert.CheckEqual(1.0, (t4[0,0,0,0].toDouble()))

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
    member _.TestTensorLinspace () =
        for combo in Combos.FloatingPoint do
            let t = combo.linspace(0, 5, 5)
            let tCorrect = combo.tensor([0.0000, 1.2500, 2.5000, 3.7500, 5.0000])
            Assert.True(tCorrect.allclose(t, 0.1))

            let t = combo.linspace(-4, 5, 8)
            let tCorrect = combo.tensor([-4.0000, -2.7143, -1.4286, -0.1429,  1.1429,  2.4286,  3.7143,  5.0000])
            Assert.True(tCorrect.allclose(t, 0.1))

    [<Test>]
    member _.TestTensorLogspace () =
        for combo in Combos.FloatingPoint do
            let t = combo.logspace(0, 5, 5)
            let tCorrect = combo.tensor([1.0000e+00, 1.7783e+01, 3.1623e+02, 5.6234e+03, 1.0000e+05])
            Assert.True(tCorrect.allclose(t, 0.1))

            let t = combo.logspace(-10., 1., 10, Math.E)
            let tCorrect = combo.tensor([4.5400e-05, 1.5412e-04, 5.2320e-04, 1.7761e-03, 6.0294e-03, 2.0468e-02, 6.9483e-02, 2.3588e-01, 8.0074e-01, 2.7183e+00])
            Assert.True(tCorrect.allclose(t, 0.1))

            let t = combo.logspace(1, 10, 10)
            let tCorrect = combo.tensor([1.0000e+01, 1.0000e+02, 1.0000e+03, 1.0000e+04, 1.0000e+05, 1.0000e+06, 1.0000e+07, 1.0000e+08, 1.0000e+09, 1.0000e+10])
            Assert.True(tCorrect.allclose(t, 0.1))

    [<Test>]
    member _.TestTensorZeroSize () =
        for combo in Combos.AllExcept16s do
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

        for combo in Combos.IntegralAndFloatingPointExcept16s do
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

        for combo in Combos.IntegralAndFloatingPointExcept16s do
            let t = combo.tensor([])

            let tSub = t - 2
            let tSubCorrect = t
            Assert.CheckEqual(tSubCorrect, tSub)

            let tDiv = t / 2
            let tDivCorrect = t.cast(Dtype.divisionType t.dtype t.dtype)
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

        for combo in Combos.FloatingPointExcept16s do
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
            let m1stddev = m1.float().std()
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
            let m2stddev = m2.float().std(dim=1)
            let m2meanCorrect = combo.tensor([1.3001, 0.3001]).float()
            let m2stddevCorrect = combo.tensor([0.7810, 0.6404]).float()
            Assert.CheckEqual(m2dtypeCorrect, m2dtype)
            Assert.True(m2meanCorrect.allclose(m2mean, 0.15))
            Assert.True(m2stddevCorrect.allclose(m2stddev, 0.15))

    [<Test>]
    member _.TestTensorBernoulli () =
        for combo in Combos.FloatingPointExcept16s do
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
        for combo in Combos.FloatingPointExcept16s do
            for p in [0.; 0.2; 0.8; 1.] do
                let t = combo.ones([100;100;8;8])
                let d = dsharp.dropout2d(t, p)
                let m = d.mean() |> float
                let mCorrect = 1. - p
                Assert.True(abs(mCorrect - m) < 0.1)

    [<Test>]
    member _.TestTensorDropout3d () =
        for combo in Combos.FloatingPointExcept16s do
            for p in [0.; 0.2; 0.8; 1.] do
                let t = combo.ones([100;100;8;8;8])
                let d = dsharp.dropout3d(t, p)
                let m = d.mean() |> float
                let mCorrect = 1. - p
                Assert.True(abs(mCorrect - m) < 0.1)

    [<Test>]
    member _.TestTensorToString () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let tempty = combo.tensor([])
            let t0 = combo.tensor(2.)
            let t1 = combo.tensor([[2.]; [2.]])
            let t2 = combo.tensor([[[2.; 2.]]])
            let t3 = combo.tensor([[1.;2.]; [3.;4.]])
            let t4 = combo.tensor([[[[1.]]]])
            let temptyString = tempty.ToString()
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
                | Float16
                | BFloat16
                | Float32 -> "."
                | Float64 -> "."
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
            let temptyStringCorrect = "tensor([])"
            let t0StringCorrect = sprintf "tensor(2%s%s)" suffix extraText
            let t1StringCorrect = sprintf "tensor([[2%s],\n        [2%s]]%s)" suffix suffix extraText
            let t2StringCorrect = sprintf "tensor([[[2%s, 2%s]]]%s)" suffix suffix extraText
            let t3StringCorrect = sprintf "tensor([[1%s, 2%s],\n        [3%s, 4%s]]%s)" suffix suffix suffix suffix extraText
            let t4StringCorrect = sprintf "tensor([[[[1%s]]]]%s)" suffix extraText
            Assert.CheckEqual(temptyStringCorrect, temptyString)
            Assert.CheckEqual(t0StringCorrect, t0String)
            Assert.CheckEqual(t1StringCorrect, t1String)
            Assert.CheckEqual(t2StringCorrect, t2String)
            Assert.CheckEqual(t3StringCorrect, t3String)
            Assert.CheckEqual(t4StringCorrect, t4String)

        let t0Bool = dsharp.tensor([ 0.; 1. ], dtype=Dtype.Bool)
        let t0BoolToString = t0Bool.ToString()
        let t0BoolToStringCorrect = sprintf "tensor([false,  true],dtype=Bool)" 
        Assert.CheckEqual(t0BoolToStringCorrect, t0BoolToString)

        let t1Bool = dsharp.tensor([ false; true ], dtype=Dtype.Bool)
        let t1BoolToString = t1Bool.ToString()
        let t1BoolToStringCorrect = sprintf "tensor([false,  true],dtype=Bool)" 
        Assert.CheckEqual(t1BoolToStringCorrect, t1BoolToString)

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

            if combo.dtype.IsFloatingPoint then
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
                let t1 = combo1.tensor([0, 1, 2, 3])
                let t2 = t1.move(combo2.device, combo2.dtype, combo2.backend)
                let t2b = t2.move(combo1.device, combo1.dtype, combo1.backend)
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
            let t1b = t1.move(?device=None, dtype=combo1.dtype, ?backend=None)
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
    member _.TestTensorEqTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 5.; 4.])
            let t1t2Eq = t1.eq(t2)
            let t1t2EqCorrect = combo.tensor([1.; 1.; 0.; 0.], dtype=Dtype.Bool)

            Assert.CheckEqual(t1t2EqCorrect, t1t2Eq)
            Assert.CheckEqual(Dtype.Bool, t1t2Eq.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            // Test bool type separately
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolEq = t1Bool.eq(t2Bool)
            let t1Boolt2BoolEqCorrect = combo.tensor([true; false; false; true ], dtype=Dtype.Bool)

            Assert.CheckEqual(t1Boolt2BoolEqCorrect, t1Boolt2BoolEq)

    [<Test>]
    member _.TestTensorNeq () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 5.; 4.])
            let t1t2Neq = t1.ne(t2)
            let t1t2NeqCorrect = combo.tensor([0.; 0.; 1.; 1.], dtype=Dtype.Bool)

            Assert.CheckEqual(t1t2NeqCorrect, t1t2Neq)
            Assert.CheckEqual(Dtype.Bool, t1t2Neq.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            // Test bool type separately
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolNeq = t1Bool.ne(t2Bool)
            let t1Boolt2BoolNeqCorrect = combo.tensor([false; true; true; false ], dtype=Dtype.Bool)

            Assert.CheckEqual(t1Boolt2BoolNeqCorrect, t1Boolt2BoolNeq)

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
        for combo in Combos.FloatingPointExcept16s do 
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
      for combo in Combos.IntegralAndFloatingPointExcept16s do 

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
                  let t7b = combo.tensor(ArrayND.init shape (fun is -> double (Array.sum is) + 2.0))
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
                          [if p1 > 0 then combo.tensor(t6vs[0..p1-1]);
                           if p2 > 0 then combo.tensor(t6vs[p1..p1+p2-1]);
                           if p3 > 0 then combo.tensor(t6vs[p1+p2..])]

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
                          [if p1 > 0 then combo.tensor([ t7vs1[0..p1-1];     t7vs2[0..p1-1] ]);
                           if p2 > 0 then combo.tensor([ t7vs1[p1..p1+p2-1]; t7vs2[p1..p1+p2-1] ]);
                           if p3 > 0 then combo.tensor([ t7vs1[p1+p2..];     t7vs2[p1+p2..] ])]

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
      for combo in Combos.FloatingPointExcept16s do 
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
                  let t6b = combo.tensor(ArrayND.init shape (fun is -> double (Array.sum is) + 2.0))
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

        // Integer and bool tensors get cast to the default floating point type for division
        for combo in Combos.IntegralAndBool do 
            let t1a = combo.tensor([2; 3; 4])
            let t1b = combo.tensor([1; 2; 3])
            let i1 = t1a / t1b
            let i1Correct = t1a.cast(Dtype.Default) / t1b.cast(Dtype.Default)
            Assert.True(i1Correct.allclose(i1, 0.01))

            let t2a = combo.tensor(6)
            let t2b = combo.tensor([1; 2; 3])
            let i2 = t2a / t2b
            let i2Correct = t2a.cast(Dtype.Default) / t2b.cast(Dtype.Default)
            Assert.True(i2Correct.allclose(i2, 0.01))

            let t3a = combo.tensor([6; 12; 18])
            let t3b = combo.tensor(3)
            let i3 = t3a / t3b
            let i3Correct = t3a.cast(Dtype.Default) / t3b.cast(Dtype.Default)
            Assert.True(i3Correct.allclose(i3, 0.01))

    [<Test>]
    member _.TestTensorPowTT () =
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([8.0766; 3.3030; 2.1732; 8.9448; 1.1028])
            let t2 = combo.tensor([5.1067; 7.4633; 3.6027; 9.0070; 7.3012])
            let t3 = t1.matmul(t2)
            let t3Correct = t1.dot(t2)

            Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul12 () =
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([8.0766; 3.3030; 2.1732; 8.9448; 1.1028])
            let t2 = combo.tensor([[5.1067; 0.0681];
                                    [7.4633; 3.6027];
                                    [9.0070; 7.3012];
                                    [2.6639; 2.8728];
                                    [7.9229; 2.3695]])
            let t3 = t1.matmul(t2)
            let t3Correct = t1.expand([1;5]).matmul(t2).squeeze(0)

            Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul13 () =
        for combo in Combos.FloatingPointExcept16s do
            // 5 --> 1x5 --> 3x1x5 (batching expansion)
            let t1 = combo.tensor([8.0766; 3.3030; 2.1732; 8.9448; 1.1028])
            
            // 3x5x2 (batch dimension is 3)
            let t2 = combo.tensor([[[5.1067; 0.0681];
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
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                    [5.1067; 7.4633; 3.6027; 9.0070; 7.3012]])
            let t2 = combo.tensor([0.0681; 3.6027; 7.3012; 2.8728; 2.3695])
            let t3 = t1.matmul(t2)
            let t3Correct = t1.matmul(t2.unsqueeze(1)).squeeze(1)

            Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul31 () =
        for combo in Combos.FloatingPointExcept16s do
            //2 x 2 x 5
            let t1 = combo.tensor([[[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                     [5.1067; 7.4633; 3.6027; 9.0070; 7.3012]];
                                    [[9.0766; 4.3030; 2.1732; 8.9448; 1.1028];
                                     [3.1067; 5.4633; 3.6027; 9.0070; 7.3012]]])
            
            // 5 --> 5x1 (matmul expand) -> 2x5x1 (batch expand)
            let t2 = combo.tensor([0.0681; 3.6027; 7.3012; 2.8728; 2.3695])
            // 2x2x5 * 2x5x1 --> 2x2x1 --> 2x2 (reverse matmul expand)
            let t3 = t1.matmul(t2)
            let t3Correct = t1.matmul(t2.unsqueeze(1)).squeeze(2)

            Assert.AreEqual([|2;2|], t3.shape)
            Assert.True(t3.allclose(t3Correct, 0.001))

    [<Test>]
    member _.TestTensorMatMul33 () =
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                    [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                    [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
            let t2 = combo.tensor([[5.1067; 0.0681];
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
    member _.TestTensorMatMul44 () =
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                    [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                    [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
            let t2 = combo.tensor([[5.1067; 0.0681];
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
    member _.TestTensorMatMulBroadcast1 () =
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                    [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                    [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
            let t2 = combo.tensor([[5.1067; 0.0681];
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
    member _.TestTensorMatMulBroadcast2 () =
        for combo in Combos.FloatingPointExcept16s do
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                    [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                    [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
            let t2 = combo.tensor([[5.1067; 0.0681];
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
            let t1Sum = t1.sum(0)
            let t1SumCorrect = combo.tensor([4.; 6.], dtype=combo.dtype.SummationType)

            Assert.CheckEqual(t1SumCorrect, t1Sum)
            Assert.CheckEqual(t1Sum.dtype, combo.dtype.SummationType)
    
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
            let t = combo.randn([2;2;2])
            let tvariance_3 = t.var(-3)
            let tvariance_2 = t.var(-2)
            let tvariance_1 = t.var(-1)
            let tvariance0 = t.var(0)
            let tvariance1 = t.var(1)
            let tvariance2 = t.var(2)

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
            input = np[[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]]
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
        for combo in Combos.FloatingPointExcept16s do 
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416];
                [0.6078;0.4742;0.7844;0.0967];
                [0.1416;0.1559;0.6452;0.1417]];
 
                [[0.0848;0.4156;0.5542;0.4166];
                [0.5187;0.0520;0.4763;0.1509];
                [0.4767;0.8096;0.1729;0.6671]]])
            let tStddev = t.std()
            let tStddevCorrect = combo.tensor(0.2398)

            Assert.True(tStddev.allclose(tStddevCorrect, 0.01))
            Assert.CheckEqual(tStddev.dtype, combo.dtype)

            // stddev, dim={0,1,2,3}, keepDim=true
            let tStddev0 = t.std(0)
            let tStddev0Correct = combo.tensor([[0.2078; 0.2375; 0.2326; 0.0530];
                [0.0630; 0.2985; 0.2179; 0.0383];
                [0.2370; 0.4623; 0.3339; 0.3715]])
            let tStddev1 = t.std(1)
            let tStddev1Correct = combo.tensor([[0.2331; 0.2981; 0.2911; 0.1304];
                [0.2393; 0.3789; 0.2014; 0.2581]])
            let tStddev2 = t.std(2)
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
            let tStddev0 = t.std(0, keepDim=true)
            let tStddev0Correct = combo.tensor([[[0.2078; 0.2375; 0.2326; 0.0530];[0.0630; 0.2985; 0.2179; 0.0383];[0.2370; 0.4623; 0.3339; 0.3715]]])
            let tStddev1 = t.std(1, keepDim=true)
            let tStddev1Correct = combo.tensor([[[0.2331; 0.2981; 0.2911; 0.1304]];[[0.2393; 0.3789; 0.2014; 0.2581]]])
            let tStddev2 = t.std(2, keepDim=true)
            let tStddev2Correct = combo.tensor([[[0.2277]; [0.2918]; [0.2495]];[[0.1996]; [0.2328]; [0.2753]]])

            Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
            Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
            Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))

    [<Test>]
    member _.TestTensorVariance () =
        for combo in Combos.FloatingPointExcept16s do 
            (* Python:
            import torch
            input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
            input.var()
            *)
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
            let tVariance = t.var()
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
            let tVariance0 = t.var(0)
            let tVariance0Correct = combo.tensor([[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]])
            let tVariance1 = t.var(1)
            let tVariance1Correct = combo.tensor([[0.0543; 0.0888; 0.0847; 0.0170];[0.0573; 0.1436; 0.0406; 0.0666]])
            let tVariance2 = t.var(2)
            let tVariance2Correct = combo.tensor([[0.0519; 0.0852; 0.0622];[0.0398; 0.0542; 0.0758]])

            Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
            Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
            Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))
            Assert.CheckEqual(tVariance0.dtype, combo.dtype)
            Assert.CheckEqual(tVariance1.dtype, combo.dtype)
            Assert.CheckEqual(tVariance2.dtype, combo.dtype)

            let tVarianceBiased = t.var(unbiased=false)
            let tVarianceBiasedCorrect = combo.tensor(0.0551)

            Assert.True(tVarianceBiased.allclose(tVarianceBiasedCorrect, 0.01))

            let tVarianceBiased0 = t.var(0, unbiased=false)
            let tVarianceBiased0Correct = combo.tensor([[0.0216, 0.0282, 0.0271, 0.0014],
                                                        [0.0020, 0.0446, 0.0237, 0.0007],
                                                        [0.0281, 0.1068, 0.0558, 0.0690]])
            let tVarianceBiased1 = t.var(1, unbiased=false)
            let tVarianceBiased1Correct = combo.tensor([[0.0362, 0.0592, 0.0565, 0.0113],
                                                        [0.0382, 0.0957, 0.0271, 0.0444]])
            let tVarianceBiased2 = t.var(2, unbiased=false)
            let tVarianceBiased2Correct = combo.tensor([[0.0389, 0.0639, 0.0467],
                                                        [0.0299, 0.0407, 0.0568]])

            Assert.True(tVarianceBiased0.allclose(tVarianceBiased0Correct, 0.01, 0.01))
            Assert.True(tVarianceBiased1.allclose(tVarianceBiased1Correct, 0.01, 0.01))
            Assert.True(tVarianceBiased2.allclose(tVarianceBiased2Correct, 0.01, 0.01))
            Assert.CheckEqual(tVarianceBiased0.dtype, combo.dtype)
            Assert.CheckEqual(tVarianceBiased1.dtype, combo.dtype)
            Assert.CheckEqual(tVarianceBiased2.dtype, combo.dtype)

        let tPrecisionCheckData = dsharp.tensor([1e10+4.0; 1e10+7.0; 1e10+13.0;1e10+16.0],dtype=Float64)
        let tPrecisionCheck = tPrecisionCheckData.var()
        let tPrecisionCheck0 = tPrecisionCheckData.var(0)
        let tPrecisionCheckCorrect = dsharp.tensor(30.0,dtype=Float64)
        Assert.True(tPrecisionCheck.allclose(tPrecisionCheckCorrect, 0.01, 0.01))
        Assert.True(tPrecisionCheck0.allclose(tPrecisionCheckCorrect, 0.01, 0.01))

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
            let tVariance0 = t.var(0, keepDim=true)
            let tVariance0Correct = combo.tensor([[[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]]])
            let tVariance1 = t.var(1, keepDim=true)
            let tVariance1Correct = combo.tensor([[[0.0543; 0.0888; 0.0847; 0.0170]];[[0.0573; 0.1436; 0.0406; 0.0666]]])
            let tVariance2 = t.var(2, keepDim=true)
            let tVariance2Correct = combo.tensor([[[0.0519];[0.0852];[0.0622]];[[0.0398];[0.0542];[0.0758]]])

            Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
            Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
            Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))
            Assert.CheckEqual(tVariance0.dtype, combo.dtype)
            Assert.CheckEqual(tVariance1.dtype, combo.dtype)
            Assert.CheckEqual(tVariance2.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorCovariance () =
        (* Python:
        import numpy as np
        a = np.array([[0.3787,0.7515,0.2252,0.3416],
            [0.6078,0.4742,0.7844,0.0967],
            [0.1416,0.1559,0.6452,0.1417]])
        a0 = a[[0],0]
        a1 = a[0,:]
        a2 = a[0:1,:]
        fweights = np.array([1,7,7,4])
        aweights = np.array([0.7241, 0.2481, 0.4878, 0.6862])


        np.cov(a0)
        np.cov(a1)
        np.cov(a2)

        np.cov(a)
        np.cov(a,ddof=0)
        np.cov(a,fweights=fweights)
        np.cov(a,aweights=aweights)
        *)
        for combo in Combos.FloatingPointExcept16s do
            let t = combo.tensor([[0.3787;0.7515;0.2252;0.3416];
                                  [0.6078;0.4742;0.7844;0.0967];
                                  [0.1416;0.1559;0.6452;0.1417]])
            let t0 = t.[0,0]
            let t1 = t.[0]
            let t2 = t.[0].view([1;-1])
            let fweights = combo.tensor([1;7;7;4],dtype=Dtype.Int32)
            let aweights = combo.tensor([0.7241; 0.2481; 0.4878; 0.6862])

            // to suppress printfn from this test "Warning: degress of freedom <= 0"
            Console.SetOut(IO.TextWriter.Null)
            let t0Unbiased = t0.cov()
            Assert.True(t0Unbiased.isnan().toBool())
            // restore stdout
            let stdout = new IO.StreamWriter(Console.OpenStandardOutput())
            stdout.AutoFlush <- true
            Console.SetOut(stdout)
            
            let t0Biased = t0.cov(correction= int64 0)
            let t0BiasedCorrect = combo.tensor(0)
            Assert.True(t0Biased.allclose(t0BiasedCorrect,0.01,0.01))
            
            let t1Unbiased = t1.cov()
            let t1UnbiasedCorrect = combo.tensor(0.0518731)
            Assert.True(t1Unbiased.allclose(t1UnbiasedCorrect,0.01,0.01))


            let t1Biased = t1.cov(correction= int64 0)
            let t1BiasedCorrect = combo.tensor(0.03890482)
            Assert.True(t1Biased.allclose(t1BiasedCorrect,0.01,0.01))

            Assert.True(t1.cov().allclose(t2.cov(),0.01,0.01))
            Assert.True(t1.cov(correction= int64 0).allclose(t2.cov(correction= int64 0),0.01,0.01))

            let tUnbiased = t.cov()
            let tUnbiasedCorrect = combo.tensor([[ 0.0518731 , -0.01221014, -0.03185672],
                                                 [-0.01221014,  0.08516011,  0.04919771],
                                                 [-0.03185672,  0.04919771,  0.06224549]])
            Assert.True(tUnbiased.allclose(tUnbiasedCorrect,0.01,0.01))

            let tBiased = t.cov(correction= int64 0)
            let tBiasedCorrect = 
                combo.tensor([[ 0.03890482, -0.0091576 , -0.02389254],
                              [-0.0091576 ,  0.06387008,  0.03689828],
                              [-0.02389254,  0.03689828,  0.04668411]])
            Assert.True(tBiased.allclose(tBiasedCorrect,0.01,0.01))

            let tUnbiasedFWeights = t.cov(fweights=fweights)
            let tUnbiasedFWeightsCorrect = 
                combo.tensor([[ 0.05789406, -0.01862841, -0.04269081],
                              [-0.01862841,  0.0682321 ,  0.0523144 ],
                              [-0.04269081,  0.0523144 ,  0.06026907]])
            Assert.True(tUnbiasedFWeights.allclose(tUnbiasedFWeightsCorrect,0.01,0.01))

            let tBiasedFWeights = t.cov(fweights=fweights,correction= int64 0)
            let tBiasedFWeightsCorrect = 
                combo.tensor([[ 0.054847  , -0.01764797, -0.04044393],
                              [-0.01764797,  0.06464094,  0.04956101],
                              [-0.04044393,  0.04956101,  0.05709701]])
            Assert.True(tBiasedFWeights.allclose(tBiasedFWeightsCorrect,0.01,0.01))

            let tUnbiasedAWeights = t.cov(aweights=aweights)
            let tUnbiasedAWeightsCorrect = 
                combo.tensor([[ 0.03039008, -0.00885102, -0.02299303],
                              [-0.00885102,  0.10213812,  0.05019765],
                              [-0.02299303,  0.05019765,  0.06144794]])
            Assert.True(tUnbiasedAWeights.allclose(tUnbiasedAWeightsCorrect,0.01,0.01))

            let tBiasedAWeights = t.cov(aweights=aweights,correction= int64 0)
            let tBiasedAWeightsCorrect = 
                combo.tensor([[ 0.0218481 , -0.00636319, -0.0165302 ],
                              [-0.00636319,  0.07342935,  0.0360882 ],
                              [-0.0165302 ,  0.0360882 ,  0.04417628]])
            Assert.True(tBiasedAWeights.allclose(tBiasedAWeightsCorrect,0.01,0.01))

            let tUnbiasedFWeightsAWeights = t.cov(fweights=fweights,aweights=aweights)
            let tUnbiasedFWeightsAWeightsCorrect = 
                combo.tensor([[ 0.04020249, -0.01536804, -0.03199123],
                              [-0.01536804,  0.09027013,  0.06286618],
                              [-0.03199123,  0.06286618,  0.0633787 ]])
            Assert.True(tUnbiasedFWeightsAWeights.allclose(tUnbiasedFWeightsAWeightsCorrect,0.01,0.01))

            let tBiasedFWeightsAWeights = t.cov(fweights=fweights,aweights=aweights,correction= int64 0)
            let tBiasedFWeightsAWeightsCorrect = 
                combo.tensor([[ 0.03776553, -0.01443647, -0.03005202],
                              [-0.01443647,  0.08479822,  0.05905542],
                              [-0.03005202,  0.05905542,  0.05953687]])
            Assert.True(tBiasedFWeightsAWeights.allclose(tBiasedFWeightsAWeightsCorrect,0.01,0.01))

    [<Test>]
    member _.TestTensorCorrCoef () =
        (* Python:
        import numpy as np
        a = np.array([[0.3787,0.7515,0.2252,0.3416],
            [0.6078,0.4742,0.7844,0.0967],
            [0.1416,0.1559,0.6452,0.1417]])
        np.corrcoef(a)
        *)
        for combo in Combos.FloatingPointExcept16s do
            let t = combo.tensor([[0.3787;0.7515;0.2252;0.3416];
                                  [0.6078;0.4742;0.7844;0.0967];
                                  [0.1416;0.1559;0.6452;0.1417]])
            let tCorrCoef = t.corrcoef()
            let tCorrCoefCorrect =
                combo.tensor([[ 1.        , -0.18370941, -0.56062968],
                              [-0.18370941,  1.        ,  0.67572941],
                              [-0.56062968,  0.67572941,  1.        ]])
            Assert.True(tCorrCoef.allclose(tCorrCoefCorrect,0.01,0.01))
            
    [<Test>]
    member _.TestTensorPermuteT () =
        for combo in Combos.All do 
            let t = combo.arange(2*3*4*5).view([2;3;4;5]).cast(combo.dtype)
            
            let t0123 = t.permute([0;1;2;3])
            let t0123Correct = t

            let t1023 = t.permute([1;0;2;3])
            let t1023Correct = combo.tensor([[[[  0,   1,   2,   3,   4],
                                                  [  5,   6,   7,   8,   9],
                                                  [ 10,  11,  12,  13,  14],
                                                  [ 15,  16,  17,  18,  19]],

                                                 [[ 60,  61,  62,  63,  64],
                                                  [ 65,  66,  67,  68,  69],
                                                  [ 70,  71,  72,  73,  74],
                                                  [ 75,  76,  77,  78,  79]]],


                                                [[[ 20,  21,  22,  23,  24],
                                                  [ 25,  26,  27,  28,  29],
                                                  [ 30,  31,  32,  33,  34],
                                                  [ 35,  36,  37,  38,  39]],

                                                 [[ 80,  81,  82,  83,  84],
                                                  [ 85,  86,  87,  88,  89],
                                                  [ 90,  91,  92,  93,  94],
                                                  [ 95,  96,  97,  98,  99]]],


                                                [[[ 40,  41,  42,  43,  44],
                                                  [ 45,  46,  47,  48,  49],
                                                  [ 50,  51,  52,  53,  54],
                                                  [ 55,  56,  57,  58,  59]],

                                                 [[100, 101, 102, 103, 104],
                                                  [105, 106, 107, 108, 109],
                                                  [110, 111, 112, 113, 114],
                                                  [115, 116, 117, 118, 119]]]])

            let t1032 = t.permute([1;0;3;2])
            let t1032Correct = combo.tensor([[[[  0,   5,  10,  15],
                                                  [  1,   6,  11,  16],
                                                  [  2,   7,  12,  17],
                                                  [  3,   8,  13,  18],
                                                  [  4,   9,  14,  19]],

                                                 [[ 60,  65,  70,  75],
                                                  [ 61,  66,  71,  76],
                                                  [ 62,  67,  72,  77],
                                                  [ 63,  68,  73,  78],
                                                  [ 64,  69,  74,  79]]],


                                                [[[ 20,  25,  30,  35],
                                                  [ 21,  26,  31,  36],
                                                  [ 22,  27,  32,  37],
                                                  [ 23,  28,  33,  38],
                                                  [ 24,  29,  34,  39]],

                                                 [[ 80,  85,  90,  95],
                                                  [ 81,  86,  91,  96],
                                                  [ 82,  87,  92,  97],
                                                  [ 83,  88,  93,  98],
                                                  [ 84,  89,  94,  99]]],


                                                [[[ 40,  45,  50,  55],
                                                  [ 41,  46,  51,  56],
                                                  [ 42,  47,  52,  57],
                                                  [ 43,  48,  53,  58],
                                                  [ 44,  49,  54,  59]],

                                                 [[100, 105, 110, 115],
                                                  [101, 106, 111, 116],
                                                  [102, 107, 112, 117],
                                                  [103, 108, 113, 118],
                                                  [104, 109, 114, 119]]]])
            let t3210 = t.permute([3;2;1;0])
            let t3210Correct = combo.tensor([[[[  0,  60],
                                                  [ 20,  80],
                                                  [ 40, 100]],

                                                 [[  5,  65],
                                                  [ 25,  85],
                                                  [ 45, 105]],

                                                 [[ 10,  70],
                                                  [ 30,  90],
                                                  [ 50, 110]],

                                                 [[ 15,  75],
                                                  [ 35,  95],
                                                  [ 55, 115]]],


                                                [[[  1,  61],
                                                  [ 21,  81],
                                                  [ 41, 101]],

                                                 [[  6,  66],
                                                  [ 26,  86],
                                                  [ 46, 106]],

                                                 [[ 11,  71],
                                                  [ 31,  91],
                                                  [ 51, 111]],

                                                 [[ 16,  76],
                                                  [ 36,  96],
                                                  [ 56, 116]]],


                                                [[[  2,  62],
                                                  [ 22,  82],
                                                  [ 42, 102]],

                                                 [[  7,  67],
                                                  [ 27,  87],
                                                  [ 47, 107]],

                                                 [[ 12,  72],
                                                  [ 32,  92],
                                                  [ 52, 112]],

                                                 [[ 17,  77],
                                                  [ 37,  97],
                                                  [ 57, 117]]],


                                                [[[  3,  63],
                                                  [ 23,  83],
                                                  [ 43, 103]],

                                                 [[  8,  68],
                                                  [ 28,  88],
                                                  [ 48, 108]],

                                                 [[ 13,  73],
                                                  [ 33,  93],
                                                  [ 53, 113]],

                                                 [[ 18,  78],
                                                  [ 38,  98],
                                                  [ 58, 118]]],


                                                [[[  4,  64],
                                                  [ 24,  84],
                                                  [ 44, 104]],

                                                 [[  9,  69],
                                                  [ 29,  89],
                                                  [ 49, 109]],

                                                 [[ 14,  74],
                                                  [ 34,  94],
                                                  [ 54, 114]],

                                                 [[ 19,  79],
                                                  [ 39,  99],
                                                  [ 59, 119]]]])

            Assert.CheckEqual(t0123Correct, t0123)
            Assert.CheckEqual(t1023Correct, t1023)
            Assert.CheckEqual(t1032Correct, t1032)
            Assert.CheckEqual(t3210Correct, t3210)
            Assert.CheckEqual(t0123.dtype, combo.dtype)

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
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let t1Bool = combo.tensor([true;false], dtype=Dtype.Bool)
            let t1BoolSignCorrect = combo.tensor([true; false], dtype=Dtype.Bool)

            Assert.CheckEqual(t1BoolSignCorrect, t1Bool.sign())

    [<Test>]
    member _.TestTensorFloorT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Floor = t1.floor()
            let t1FloorCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            Assert.True(t1Floor.allclose(t1FloorCorrect, 0.01))
            Assert.CheckEqual(t1Floor.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).floor())

    [<Test>]
    member _.TestTensorCeilT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Ceil = t1.ceil()
            let t1CeilCorrect = combo.tensor([1.; 1.; 1.; 1.; 1.])

            Assert.True(t1Ceil.allclose(t1CeilCorrect, 0.01))
            Assert.CheckEqual(t1Ceil.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).ceil())

    [<Test>]
    member _.TestTensorRoundT () =
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let t1 = combo.tensor([true; false], dtype=Dtype.Bool)
            isInvalidOp (fun () -> t1.abs())

    [<Test>]
    member _.TestTensorReluT () =
        for combo in Combos.SignedIntegralAndFloatingPointExcept16s do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.; 10.])
            let t1Relu = t1.relu()
            let t1ReluCorrect = combo.tensor([0.; 0.; 0.; 3.; 10.])

            Assert.CheckEqual(t1ReluCorrect, t1Relu)
            Assert.CheckEqual(t1Relu.dtype, combo.dtype)

        // Test bool separately
        for combo in Combos.AllDevicesAndBackendsFloat32 do
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
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Sigmoid = t1.sigmoid()
            let t1SigmoidCorrect = combo.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])

            Assert.True(t1Sigmoid.allclose(t1SigmoidCorrect, 0.01))
            Assert.CheckEqual(t1Sigmoid.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
          isInvalidOp(fun () -> combo.tensor([1.0]).sigmoid())

    [<Test>]
    member _.TestTensorSoftplusT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05])
            let t1Softplus = t1.softplus()
            let t1SoftplusCorrect = combo.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])

            Assert.True(t1Softplus.allclose(t1SoftplusCorrect, 0.01))
            Assert.CheckEqual(t1Softplus.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).softplus())

    [<Test>]
    member _.TestTensorExpT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9139; -0.5907;  1.9422; -0.7763; -0.3274])
            let t1Exp = t1.exp()
            let t1ExpCorrect = combo.tensor([2.4940; 0.5539; 6.9742; 0.4601; 0.7208])

            Assert.True(t1Exp.allclose(t1ExpCorrect, 0.01))
            Assert.CheckEqual(t1Exp.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).exp())

    [<Test>]
    member _.TestTensorLogT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
            let t1Log = t1.log()
            let t1LogCorrect = combo.tensor([-2.0516; -0.5426; -0.4301; -0.9727; -0.9100])

            Assert.True(t1Log.allclose(t1LogCorrect, 0.01))
            Assert.CheckEqual(t1Log.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).log())

    [<Test>]
    member _.TestTensorSafeLog () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.; -5.; System.Double.NegativeInfinity; 0.6505; 0.3781; 0.4025])
            let epsilon = 1e-12
            let t1Log = t1.safelog(epsilon)
            let t1LogCorrect = combo.tensor([-27.631, -27.631, -27.631, -0.430014, -0.972597, -0.91006])

            Assert.True(t1Log.allclose(t1LogCorrect, 0.01))
            Assert.CheckEqual(t1Log.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorLog10T () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
            let t1Log10 = t1.log10()
            let t1Log10Correct = combo.tensor([-0.8911; -0.2357; -0.1868; -0.4224; -0.3952])

            Assert.True(t1Log10.allclose(t1Log10Correct, 0.01))
            Assert.CheckEqual(t1Log10.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).log10())

    [<Test>]
    member _.TestTensorSqrtT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Sqrt = t1.sqrt()
            let t1SqrtCorrect = combo.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])

            Assert.True(t1Sqrt.allclose(t1SqrtCorrect, 0.01))
            Assert.CheckEqual(t1Sqrt.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sqrt())

    [<Test>]
    member _.TestTensorSinT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Sin = t1.sin()
            let t1SinCorrect = combo.tensor([-0.9828;  0.9991; -0.3698; -0.7510;  0.9491])

            Assert.True(t1Sin.allclose(t1SinCorrect, 0.01))
            Assert.CheckEqual(t1Sin.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sin())

    [<Test>]
    member _.TestTensorCosT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Cos = t1.cos()
            let t1CosCorrect = combo.tensor([-0.1849;  0.0418; -0.9291;  0.6603;  0.3150])

            Assert.True(t1Cos.allclose(t1CosCorrect, 0.01))
            Assert.CheckEqual(t1Cos.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).cos())

    [<Test>]
    member _.TestTensorTanT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Tan = t1.tan()
            let t1TanCorrect = combo.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])

            Assert.True(t1Tan.allclose(t1TanCorrect, 0.01))
            Assert.CheckEqual(t1Tan.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).tan())

    [<Test>]
    member _.TestTensorSinhT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Sinh = t1.sinh()
            let t1SinhCorrect = combo.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])

            Assert.True(t1Sinh.allclose(t1SinhCorrect, 0.01))
            Assert.CheckEqual(t1Sinh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sinh())

    [<Test>]
    member _.TestTensorCoshT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Cosh = t1.cosh()
            let t1CoshCorrect = combo.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])

            Assert.True(t1Cosh.allclose(t1CoshCorrect, 0.01))
            Assert.CheckEqual(t1Cosh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).cosh())

    [<Test>]
    member _.TestTensorTanhT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Tanh = t1.tanh()
            let t1TanhCorrect = combo.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])

            Assert.True(t1Tanh.allclose(t1TanhCorrect, 0.01))
            Assert.CheckEqual(t1Tanh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).tanh())

    [<Test>]
    member _.TestTensorAsinT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Asin = t1.asin()
            let t1AsinCorrect = combo.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])

            Assert.True(t1Asin.allclose(t1AsinCorrect, 0.01))
            Assert.CheckEqual(t1Asin.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).asin())

    [<Test>]
    member _.TestTensorAcosT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Acos = t1.acos()
            let t1AcosCorrect = combo.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])

            Assert.True(t1Acos.allclose(t1AcosCorrect, 0.01))
            Assert.CheckEqual(t1Acos.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).acos())

    [<Test>]
    member _.TestTensorAtanT () =
        for combo in Combos.FloatingPointExcept16s do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Atan = t1.atan()
            let t1AtanCorrect = combo.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])

            Assert.True(t1Atan.allclose(t1AtanCorrect, 0.01))
            Assert.CheckEqual(t1Atan.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).atan())

    [<Test>]
    member _.TestTensorNumToBool () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([-2., -1., -0.5, 0., 0.5, 1., 2.])
            let tb = t.bool()
            let tbCorrect = combo.tensor([true, true, true, false, true, true, true], dtype=Dtype.Bool)

            Assert.CheckEqual(tbCorrect, tb)

        for combo in Combos.Integral do
            let t = combo.tensor([-2., -1., 0., 1., 2.])
            let tb = t.bool()
            let tbCorrect = combo.tensor([true, true, false, true, true], dtype=Dtype.Bool)

            Assert.CheckEqual(tbCorrect, tb)

    [<Test>]
    member _.TestTensorSliceWithNegativeIndex () =
        for combo in Combos.All do
            let t = combo.tensor([[[ 0,  1,  2,  3],
                                    [ 4,  5,  6,  7],
                                    [ 8,  9, 10, 11]],

                                    [[12, 13, 14, 15],
                                    [16, 17, 18, 19],
                                    [20, 21, 22, 23]]])

            let t1 = t[-1]
            let t1Correct = combo.tensor([[12, 13, 14, 15],
                                            [16, 17, 18, 19],
                                            [20, 21, 22, 23]])
            Assert.CheckEqual(t1Correct, t1)

            let t2 = t[0, -1]
            let t2Correct = combo.tensor([ 8,  9, 10, 11])
            Assert.CheckEqual(t2Correct, t2)

            let t3 = t[0, 0, -1]
            let t3Correct = combo.tensor(3)
            Assert.CheckEqual(t3Correct, t3)

            let t3 = t[0, 0, -2]
            let t3Correct = combo.tensor(2)
            Assert.CheckEqual(t3Correct, t3)

            let t3 = t[0, -2, -1]
            let t3Correct = combo.tensor(7)
            Assert.CheckEqual(t3Correct, t3)

    [<Test>]
    member _.TestTensorSlice () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1.;2.])
            let t1s1 = t1[0]
            let t1s2 = t1[*]
            let t1s3 = t1[1..]
            let t1s4 = t1[..0] // In Python this is [:1] because in Python upper limits are exclusive whereas in F# they are inclusive
            let t1s1Correct = combo.tensor(1.)
            let t1s2Correct = combo.tensor([1.;2.])
            let t1s3Correct = combo.tensor([2.])
            let t1s4Correct = combo.tensor([1.])

            let t2 = combo.tensor([[1.;2.];[3.;4.]])
            let t2s1 = t2[0]
            let t2s2 = t2[*]
            let t2s3 = t2[0,0]
            let t2s4 = t2[0,*]
            let t2s5 = t2[*,0]
            let t2s6 = t2[*,*]
            let t2s7 = t2[1..]
            let t2s8 = t2[..0] // In Python this is [:1] because in Python upper limits are exclusive whereas in F# they are inclusive
            let t2s1Correct = combo.tensor([1.;2.])
            let t2s2Correct = combo.tensor([[1.;2.];[3.;4.]])
            let t2s3Correct = combo.tensor(1.)
            let t2s4Correct = combo.tensor([1.;2.])
            let t2s5Correct = combo.tensor([1.;3.])
            let t2s6Correct = combo.tensor([[1.;2.];[3.;4.]])
            let t2s7Correct = combo.tensor([[3.; 4.]])
            let t2s8Correct = combo.tensor([[1.; 2.]])

            let t2b = combo.tensor([[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]])
            let t2bs1 = t2b[1..,2..]
            let t2bs1Correct = combo.tensor([[7.;8.];[11.;12.]])
            let t2bs2 = t2b[1..2,2..3]
            let t2bs2Correct = combo.tensor([[7.;8.];[11.;12.]])

            let t3 = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
            let t3s1  = t3[0]
            let t3s2  = t3[*]
            let t3s3  = t3[0,0]
            let t3s4  = t3[0,*]
            let t3s5  = t3[*,0]
            let t3s6  = t3[*,*]
            let t3s7  = t3[0,0,0]
            let t3s8  = t3[0,0,*]
            let t3s9  = t3[0,*,0]
            let t3s10 = t3[0,*,*]
            let t3s11 = t3[*,0,0]
            let t3s12 = t3[*,0,*]
            let t3s13 = t3[*,*,0]
            let t3s14 = t3[*,*,*]
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
            let t4s1 = t4[0]
            let t4s2 = t4[0,*,*,*]
            let t4s1Correct = combo.tensor([[[1]];
                                             [[2]];
                                             [[3]]])
            let t4s2Correct = t4s1Correct

            Assert.CheckEqual(t1s1Correct, t1s1)
            Assert.CheckEqual(t1s2Correct, t1s2)
            Assert.CheckEqual(t1s3Correct, t1s3)
            Assert.CheckEqual(t1s4Correct, t1s4)

            Assert.CheckEqual(t2s1Correct, t2s1)
            Assert.CheckEqual(t2s2Correct, t2s2)
            Assert.CheckEqual(t2s3Correct, t2s3)
            Assert.CheckEqual(t2s4Correct, t2s4)
            Assert.CheckEqual(t2s5Correct, t2s5)
            Assert.CheckEqual(t2s6Correct, t2s6)
            Assert.CheckEqual(t2s7Correct, t2s7)
            Assert.CheckEqual(t2s8Correct, t2s8)

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

            let t = combo.tensor([1, 2, 3, 4])
            let tUnsqueezeMinus2 = t.unsqueeze(-2)
            let tUnsqueezeMinus2Correct = combo.tensor([[1, 2, 3, 4]])
            let tUnsqueezeMinus1 = t.unsqueeze(-1)
            let tUnsqueezeMinus1Correct = combo.tensor([[1],
                                                        [2],
                                                        [3],
                                                        [4]])
            let tUnsqueeze0 = t.unsqueeze(0)
            let tUnsqueeze0Correct = combo.tensor([[1, 2, 3, 4]])            
            let tUnsqueeze1 = t.unsqueeze(1)
            let tUnsqueeze1Correct = combo.tensor([[1],
                                                        [2],
                                                        [3],
                                                        [4]])

            Assert.True(tUnsqueezeMinus2Correct.allclose(tUnsqueezeMinus2, 0.01))
            Assert.True(tUnsqueezeMinus1Correct.allclose(tUnsqueezeMinus1, 0.01))
            Assert.True(tUnsqueeze0Correct.allclose(tUnsqueeze0, 0.01))
            Assert.True(tUnsqueeze1Correct.allclose(tUnsqueeze1, 0.01))

    [<Test>]
    member _.TestTensorUnsqueezeAs () =
        for combo in Combos.All do
            let t1 = combo.ones(3)
            let t2 = combo.ones([3;4])
            let t3 = combo.ones([3;4;2])

            let t1u1 = t1.unsqueezeAs(t1)
            let t1u1Shape = t1u1.shape
            let t1u1ShapeCorrect = t1.shape
            let t1u2 = t1.unsqueezeAs(t2)
            let t1u2Shape = t1u2.shape
            let t1u2ShapeCorrect = [|3;1|]
            let t1u3 = t1.unsqueezeAs(t3)
            let t1u3Shape = t1u3.shape
            let t1u3ShapeCorrect = [|3;1;1|]
            let t3u1 = t3.unsqueezeAs(t1)
            let t3u1Shape = t3u1.shape
            let t3u1ShapeCorrect = [|3;4;2|]

            Assert.AreEqual(t1u1ShapeCorrect, t1u1Shape)
            Assert.AreEqual(t1u2ShapeCorrect, t1u2Shape)
            Assert.AreEqual(t1u3ShapeCorrect, t1u3Shape)
            Assert.AreEqual(t3u1ShapeCorrect, t3u1Shape)

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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.SignedIntegralAndFloatingPointExcept16s do 
            let t = combo.tensor([-4,-3,-2,-1,0,1,2,3,4])
            let tClamped = dsharp.clamp(t, -2, 3)
            let tClampedCorrect = combo.tensor([-2, -2, -2, -1,  0,  1,  2,  3,  3])
            Assert.CheckEqual(tClampedCorrect, tClamped)

    [<Test>]
    member _.TestTensorClampInf () =
        for combo in Combos.FloatingPointExcept16s do 
            let t = combo.tensor([System.Double.NegativeInfinity, System.Double.PositiveInfinity])
            let tClamped = dsharp.clamp(t, -100, 100)
            let tClampedCorrect = combo.tensor([-100, 100])
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
    member _.TestTensorUnflatten () =
        for combo in Combos.All do
            let t1 = combo.randint(0, 2, [20; 20])

            let t1f1shape = dsharp.unflatten(t1, 0, [2;10]).shape
            let t1f1shapeCorrect = [|2;10;20|]

            let t1f2shape = dsharp.unflatten(t1, 1, [2;10]).shape
            let t1f2shapeCorrect = [|20;2;10|]

            Assert.CheckEqual(t1f1shapeCorrect, t1f1shape)
            Assert.CheckEqual(t1f2shapeCorrect, t1f2shape)

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
    member _.TestTensorScatter () =
        for combo in Combos.All do 
            let t1 = combo.tensor([0,1,2,3,4])
            let t1g = dsharp.scatter(t1, 0, combo.tensor([0, 2, 1, 3, 4], dtype=Dtype.Int32), destinationShape=[5])
            let t1gCorrect = combo.tensor([0., 2., 1., 3., 4.])

            let t2 = combo.tensor([[1,2,3],[4,5,6]])
            let t2g0 = dsharp.scatter(t2, 0, combo.tensor([[0, 1, 1], [1, 0, 0]], dtype=Dtype.Int32), destinationShape=[2;3])
            let t2g0Correct = combo.tensor([[1., 5., 6.],
                                             [4., 2., 3.]])
            let t2g1 = dsharp.scatter(t2, 1, combo.tensor([[0, 2, 1], [2, 0, 1]], dtype=Dtype.Int32), destinationShape=[2;3])
            let t2g1Correct = combo.tensor([[1., 3., 2.],
                                             [5., 6., 4.]])

            Assert.True(t1gCorrect.allclose(t1g, 0.01))
            Assert.CheckEqual(combo.dtype, t1g.dtype)

            Assert.True(t2g0Correct.allclose(t2g0, 0.01))
            Assert.CheckEqual(combo.dtype, t2g0.dtype)

            Assert.True(t2g1Correct.allclose(t2g1, 0.01))
            Assert.CheckEqual(combo.dtype, t2g1.dtype)

    [<Test>]
    member _.TestTensorMaxElementwise () =
        for combo in Combos.All do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t2 = combo.tensor([1.;3.;21.;2.])
            let t1Max = t1.max(t2)
            let t1MaxCorrect = combo.tensor([4.;3.;21.;3.])

            Assert.CheckEqual(t1MaxCorrect, t1Max)
            Assert.CheckEqual(combo.dtype, t1Max.dtype)

            let t2 = combo.tensor([4.;1.;0.;3.])
            let t2Max = t2.max(t2.scalarLike(2))
            let t2MaxCorrect = combo.tensor([4.;2.;2.;3.])

            Assert.CheckEqual(t2MaxCorrect, t2Max)
            Assert.CheckEqual(combo.dtype, t2Max.dtype)

    [<Test>]
    member _.TestTensorMinElementwise () =
        for combo in Combos.All do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t2 = combo.tensor([1.;3.;21.;2.])
            let t1Min = t1.min(t2)
            let t1MinCorrect = combo.tensor([1.;1.;20.;2.])

            Assert.CheckEqual(t1MinCorrect, t1Min)
            Assert.CheckEqual(combo.dtype, t1Min.dtype)

            let t2 = combo.tensor([4.;1.;0.;3.])
            let t2Min = t2.min(t2.scalarLike(1))
            let t2MinCorrect = combo.tensor([1.;1.;0.;1.])

            Assert.CheckEqual(t2MinCorrect, t2Min)
            Assert.CheckEqual(combo.dtype, t2Min.dtype)

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
        for combo in Combos.All do 
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
    member _.TestTensorMaxDim () =
        for combo in Combos.All do
            let t = combo.tensor([[5.5834, 5.6240],
                                    [0.7616, 1.8576],
                                    [7.3116, 9.7464]])

            let tmax0 = t.max(dim=0)
            let tmax0Correct = combo.tensor([7.3116, 9.7464])
            let tmax0KeepDim = t.max(dim=0, keepDim=true)
            let tmax0KeepDimCorrect = combo.tensor([[7.3116, 9.7464]])

            let tmax1 = t.max(dim=1)
            let tmax1Correct = combo.tensor([5.6240, 1.8576, 9.7464])
            let tmax1KeepDim = t.max(dim=1, keepDim=true)
            let tmax1KeepDimCorrect = combo.tensor([[5.6240],
                                                    [1.8576],
                                                    [9.7464]])

            Assert.True(tmax0Correct.allclose(tmax0, 0.01))
            Assert.True(tmax0KeepDimCorrect.allclose(tmax0KeepDim, 0.01))
            Assert.True(tmax1Correct.allclose(tmax1, 0.01))
            Assert.True(tmax1KeepDimCorrect.allclose(tmax1KeepDim, 0.01))

    [<Test>]
    member _.TestTensorMinDim () =
        for combo in Combos.All do
            let t = combo.tensor([[5.5834, 5.6240],
                                    [0.7616, 1.8576],
                                    [7.3116, 9.7464]])

            let tmin0 = t.min(dim=0)
            let tmin0Correct = combo.tensor([0.7616, 1.8576])
            let tmin0KeepDim = t.min(dim=0, keepDim=true)
            let tmin0KeepDimCorrect = combo.tensor([[0.7616, 1.8576]])

            let tmin1 = t.min(dim=1)
            let tmin1Correct = combo.tensor([5.5834, 0.7616, 7.3116])
            let tmin1KeepDim = t.min(dim=1, keepDim=true)
            let tmin1KeepDimCorrect = combo.tensor([[5.5834],
                                                    [0.7616],
                                                    [7.3116]])

            Assert.True(tmin0Correct.allclose(tmin0, 0.01))
            Assert.True(tmin0KeepDimCorrect.allclose(tmin0KeepDim, 0.01))
            Assert.True(tmin1Correct.allclose(tmin1, 0.01))
            Assert.True(tmin1KeepDimCorrect.allclose(tmin1KeepDim, 0.01))

    [<Test>]
    member _.TestTensorArgmax () =
        for combo in Combos.IntegralAndFloatingPointExcept16s do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t1Argmax = t1.argmax(0)
            let t1ArgmaxCorrect = combo.tensor(2, dtype=Dtype.Int32)

            let t1ArgmaxKeepDim = dsharp.argmax(t1, 0, keepDim=true)
            let t1ArgmaxKeepDimCorrect = combo.tensor([2], dtype=Dtype.Int32)

            let t2 = combo.tensor([[1.;4.];[2.;3.]])
            let t2Argmax = t2.argmax(0)
            let t2ArgmaxCorrect = combo.tensor([1,0], dtype=Dtype.Int32)

            let t2ArgmaxKeepDim = t2.argmax(0, keepDim=true)
            let t2ArgmaxKeepDimCorrect = combo.tensor([[1;0]], dtype=Dtype.Int32)

            let t2ArgmaxKeepDim1 = t2.argmax(1, keepDim=true)
            let t2ArgmaxKeepDim1Correct = combo.tensor([[1];[1]], dtype=Dtype.Int32)

            let t3 = combo.tensor([[[ 7.6884; 65.9125;  4.0114];
                                 [46.7944; 61.5331; 40.1627];
                                 [48.3240;  4.9910; 50.1571]];

                                [[13.4777; 65.7656; 36.8161];
                                 [47.8268; 42.2229;  5.6115];
                                 [43.4779; 77.8675; 95.7660]];

                                [[59.8422; 47.1146; 36.7614];
                                 [71.6328; 18.5912; 27.7328];
                                 [49.9120; 60.3023; 53.0838]]])

            let t3Argmax0 = t3.argmax(0)
            let t3Argmax0Correct = combo.tensor([[2, 0, 1], [2, 0, 0], [2, 1, 1]],dtype=Dtype.Int32)
        
            let t3Argmax1 = t3.argmax(1)
            let t3Argmax1Correct = combo.tensor([[2, 0, 2], [1, 2, 2], [1, 2, 2]],dtype=Dtype.Int32)
        
            let t3Argmax2 = t3.argmax(2)
            let t3Argmax2Correct = combo.tensor([[1, 1, 2],[1, 0, 2],[0, 0, 1]],dtype=Dtype.Int32)
        
            let t4 = combo.tensor([[[[1;2]]]])
            let t4Argmax = t4.argmax()
            let t4ArgmaxCorrect = [| 0;0;0;1 |]

            let t4Argmax0 = t4.argmax(0)
            let t4Argmax0Correct = combo.tensor([[[0;0]]],dtype=Dtype.Int32)

            let t5 = combo.tensor([[[[1;2]]]]).unsqueeze(0)
            let t5Argmax = t5.argmax()
            let t5ArgmaxCorrect = [| 0;0;0;0;1 |]

            let t5Argmax0 = t5.argmax(0)
            let t5Argmax0Correct = combo.tensor([[[[0;0]]]],dtype=Dtype.Int32)

            let t6 = combo.tensor([[[[1;2]]]]).unsqueeze(0).unsqueeze(0)
            let t6Argmax = t6.argmax()
            let t6ArgmaxCorrect = [| 0;0;0;0;0;1 |]

            let t6Argmax0 = t6.argmax(0)
            let t6Argmax0Correct = combo.tensor([[[[0;0]]]],dtype=Dtype.Int32).unsqueeze(0)

            Assert.CheckEqual(t1ArgmaxCorrect, t1Argmax)
            Assert.CheckEqual(t1ArgmaxKeepDimCorrect, t1ArgmaxKeepDim)

            Assert.CheckEqual(t2ArgmaxCorrect, t2Argmax)
            Assert.CheckEqual(t2ArgmaxKeepDimCorrect, t2ArgmaxKeepDim)
            Assert.CheckEqual(t2ArgmaxKeepDim1Correct, t2ArgmaxKeepDim1)

            Assert.CheckEqual(t3Argmax0Correct, t3Argmax0)
            Assert.CheckEqual(t3Argmax1Correct, t3Argmax1)
            Assert.CheckEqual(t3Argmax2Correct, t3Argmax2)

            Assert.CheckEqual(t4ArgmaxCorrect, t4Argmax)
            Assert.CheckEqual(t4Argmax0Correct, t4Argmax0)

            Assert.CheckEqual(t5ArgmaxCorrect, t5Argmax)
            Assert.CheckEqual(t5Argmax0Correct, t5Argmax0)

            Assert.CheckEqual(t6ArgmaxCorrect, t6Argmax)
            Assert.CheckEqual(t6Argmax0Correct, t6Argmax0)

        for combo in Combos.Bool do
            let t1 = combo.tensor([true; false])
            let t1Argmax = t1.argmax()
            let t1ArgmaxCorrect = [| 0 |]
            Assert.CheckEqual(t1ArgmaxCorrect, t1Argmax)

            let t2 = combo.tensor([[true; false];[false; true]])
            let t2Argmax = t2.argmax(0)
            let t2ArgmaxCorrect = combo.tensor([0; 1], dtype=Dtype.Int32)
            Assert.CheckEqual(t2ArgmaxCorrect, t2Argmax)

    [<Test>]
    member _.TestTensorArgmin () =
        for combo in Combos.IntegralAndFloatingPointExcept16s do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t1Argmin = t1.argmin(0)
            let t1ArgminCorrect = combo.tensor(1, dtype=Dtype.Int32)

            let t1ArgminKeepDim = dsharp.argmin(t1, 0, keepDim=true)
            let t1ArgminKeepDimCorrect = combo.tensor([1], dtype=Dtype.Int32)

            let t2 = combo.tensor([[1.;4.];[2.;3.]])
            let t2Argmin = t2.argmin(0)
            let t2ArgminCorrect = combo.tensor([0,1], dtype=Dtype.Int32)

            let t2ArgminKeepDim = t2.argmin(0, keepDim=true)
            let t2ArgminKeepDimCorrect = combo.tensor([[0,1]], dtype=Dtype.Int32)

            let t2ArgminKeepDim1 = t2.argmin(1, keepDim=true)
            let t2ArgminKeepDim1Correct = combo.tensor([[0],[0]], dtype=Dtype.Int32)

            let t3 = combo.tensor([[[ 7.6884; 65.9125;  4.0114];
                                 [46.7944; 61.5331; 40.1627];
                                 [48.3240;  4.9910; 50.1571]];

                                [[13.4777; 65.7656; 36.8161];
                                 [47.8268; 42.2229;  5.6115];
                                 [43.4779; 77.8675; 95.7660]];

                                [[59.8422; 47.1146; 36.7614];
                                 [71.6328; 18.5912; 27.7328];
                                 [49.9120; 60.3023; 53.0838]]])

            let t3Argmin0 = t3.argmin(0)
            let t3Argmin0Correct = combo.tensor([[0, 2, 0],[0, 2, 1],[1, 0, 0]],dtype=Dtype.Int32)
        
            let t3Argmin1 = t3.argmin(1)
            let t3Argmin1Correct = combo.tensor([[0, 2, 0],[0, 1, 1],[2, 1, 1]],dtype=Dtype.Int32)
        
            let t3Argmin2 = t3.argmin(2)
            let t3Argmin2Correct = combo.tensor([[2, 2, 1],[0, 2, 0],[2, 1, 0]],dtype=Dtype.Int32)

            let t4 = combo.tensor([[[[1;2]]]])
            let t4Argmin = t4.argmin()
            let t4ArgminCorrect = [| 0;0;0;0 |]

            let t4Argmin0 = t4.argmin(0)
            let t4Argmin0Correct = combo.tensor([[[0;0]]],dtype=Dtype.Int32)

            let t5 = combo.tensor([[[[1;2]]]]).unsqueeze(0)
            let t5Argmin = t5.argmin()
            let t5ArgminCorrect = [| 0;0;0;0;0 |]

            let t5Argmin0 = t5.argmin(0)
            let t5Argmin0Correct = combo.tensor([[[[0;0]]]],dtype=Dtype.Int32)

            let t6 = combo.tensor([[[[1;2]]]]).unsqueeze(0).unsqueeze(0)
            let t6Argmin = t6.argmin()
            let t6ArgminCorrect = [| 0;0;0;0;0;0 |]

            let t6Argmin0 = t6.argmin(0)
            let t6Argmin0Correct = combo.tensor([[[[0;0]]]],dtype=Dtype.Int32).unsqueeze(0)

            Assert.CheckEqual(t1ArgminCorrect, t1Argmin)
            Assert.CheckEqual(t1ArgminKeepDimCorrect, t1ArgminKeepDim)

            Assert.CheckEqual(t2ArgminCorrect, t2Argmin)
            Assert.CheckEqual(t2ArgminKeepDimCorrect, t2ArgminKeepDim)
            Assert.CheckEqual(t2ArgminKeepDim1Correct, t2ArgminKeepDim1)

            Assert.CheckEqual(t3Argmin0Correct, t3Argmin0)
            Assert.CheckEqual(t3Argmin1Correct, t3Argmin1)
            Assert.CheckEqual(t3Argmin2Correct, t3Argmin2)

            Assert.CheckEqual(t4ArgminCorrect, t4Argmin)
            Assert.CheckEqual(t4Argmin0Correct, t4Argmin0)

            Assert.CheckEqual(t5ArgminCorrect, t5Argmin)
            Assert.CheckEqual(t5Argmin0Correct, t5Argmin0)

            Assert.CheckEqual(t6ArgminCorrect, t6Argmin)
            Assert.CheckEqual(t6Argmin0Correct, t6Argmin0)

        for combo in Combos.Bool do
            let t1 = combo.tensor([true; false])
            let t1Argmin = t1.argmin()
            let t1ArgminCorrect = [| 1 |]
            Assert.CheckEqual(t1ArgminCorrect, t1Argmin)

            let t2 = combo.tensor([[true; false];[false; true]])
            let t2Argmin = t2.argmin(0)
            let t2ArgminCorrect = combo.tensor([1; 0], dtype=Dtype.Int32)
            Assert.CheckEqual(t2ArgminCorrect, t2Argmin)

    [<Test>]
    member _.TestTensorMaxBinary () =
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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
        for combo in Combos.FloatingPointExcept16s do 
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

            let t2a = combo.tensor([[[[0.2059, 0.2048],
                                       [0.4209, 0.4259],
                                       [0.7000, 0.0548],
                                       [0.8414, 0.4659]],
                             
                                      [[0.5841, 0.3222],
                                       [0.3938, 0.6895],
                                       [0.9694, 0.9424],
                                       [0.1203, 0.3375]],
                             
                                      [[0.8713, 0.6122],
                                       [0.1029, 0.6634],
                                       [0.2255, 0.0801],
                                       [0.7698, 0.8014]]],
                             
                             
                                     [[[0.5361, 0.2349],
                                       [0.9668, 0.7860],
                                       [0.8522, 0.2324],
                                       [0.3445, 0.2305]],
                             
                                      [[0.8407, 0.0118],
                                       [0.5815, 0.7817],
                                       [0.3957, 0.3336],
                                       [0.3524, 0.2699]],
                             
                                      [[0.4809, 0.0975],
                                       [0.9397, 0.7868],
                                       [0.0906, 0.2595],
                                       [0.7253, 0.7387]]]])
            let t2b = combo.tensor([[[[0.2288, 0.7639],
                                       [0.6365, 0.8856],
                                       [0.4637, 0.6467],
                                       [0.2029, 0.7676]],
                             
                                      [[0.8061, 0.7229],
                                       [0.8075, 0.6854],
                                       [0.4748, 0.9848],
                                       [0.6456, 0.3080]],
                             
                                      [[0.5581, 0.5572],
                                       [0.0235, 0.9419],
                                       [0.1633, 0.3299],
                                       [0.9430, 0.8501]]],
                             
                             
                                     [[[0.5505, 0.8364],
                                       [0.8939, 0.3968],
                                       [0.5038, 0.3390],
                                       [0.4534, 0.2653]],
                             
                                      [[0.5895, 0.2453],
                                       [0.1937, 0.5779],
                                       [0.2846, 0.4662],
                                       [0.6777, 0.2539]],
                             
                                      [[0.9834, 0.2274],
                                       [0.7440, 0.5305],
                                       [0.4400, 0.0725],
                                       [0.7565, 0.9452]]]])
            let l7 = dsharp.bceLoss(t2a, t2b)
            let l7Correct = combo.tensor(0.7858)
            Assert.True(l7Correct.allclose(l7, 0.01, 0.01))

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
        for combo in Combos.FloatingPointExcept16s do
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

        let fileName = System.IO.Path.GetTempFileName() + ".png"
        let t0 = dsharp.rand([3; 16; 16])
        t0.saveImage(fileName, resize=(20, 20))
        let t1 = dsharp.loadImage(fileName)
        let t1Shape = t1.shape
        let t1ShapeCorrect = [|3; 20; 20|]
        let t2 = dsharp.loadImage(fileName, resize=(12, 12))
        let t2Shape = t2.shape
        let t2ShapeCorrect = [|3; 12; 12|]

        Assert.AreEqual(t1ShapeCorrect, t1Shape)
        Assert.AreEqual(t2ShapeCorrect, t2Shape)

    [<Test>]
    member _.TestTensorDepth () =
        for combo in Combos.FloatingPoint do 
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
    member _.TestTensorAncestors () =
        let a = dsharp.randn(5).reverseDiff()
        let b = a.sin().exp() + a
        let _, s = b.ancestors()
        let sCorrect = """TensorR [|5|] AddTT
 TensorR [|5|] ExpT
  TensorR [|5|] SinT
   TensorR [|5|] NewT
 TensorR [|5|] NewT"""
        Assert.AreEqual(sCorrect, s)

    [<Test>]
    member _.TestTensorIEnumerable () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1,2,3])
            t1.unstack() |> Seq.iteri (fun i v -> Assert.CheckEqual(t1[i], v))
            let t2 = combo.tensor([[1,2,3], [4,5,6]])
            t2.unstack() |> Seq.iteri (fun i v -> Assert.CheckEqual(t2[i,*], v))

    [<Test>]
    member _.TestTensorFSharpCoreOps () =
        for combo in Combos.FloatingPointExcept16s do 
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
