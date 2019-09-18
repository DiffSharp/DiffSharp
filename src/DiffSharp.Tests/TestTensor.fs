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
    member this.TestTensorAdd () =
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
    member this.TestTensorSub () =
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
    member this.TestTensorNeg () =
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Neg = -t1
        let t1NegCorrect = Tensor.Create([-1.; -2.; -3.])

        Assert.AreEqual(t1Neg, t1NegCorrect)

    [<Test>]
    member this.TestTensorSum () =
        let t1 = Tensor.Create([1.; 2.; 3.])
        let t1Sum = t1.Sum()
        let t1SumCorrect = Tensor.Create(6.)

        let t2 = Tensor.Create([[1.; 2.]; [3.; 4.]])
        let t2Sum = t2.Sum()
        let t2SumCorrect = Tensor.Create(10.)

        Assert.AreEqual(t1Sum, t1SumCorrect)
        Assert.AreEqual(t2Sum, t2SumCorrect)