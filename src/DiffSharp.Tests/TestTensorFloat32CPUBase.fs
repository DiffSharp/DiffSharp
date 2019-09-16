namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestTensorFloat32CPUBase () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestAdd () =
        let t1 = Tensor.Create([1.f; 2.f]) + Tensor.Create([3.f; 4.f])
        let t1Correct = Tensor.Create([4.f; 6.f])

        let t2 = Tensor.Create([1.f; 2.f]) + Tensor.Create(5.f)
        let t2Correct = Tensor.Create([6.f; 7.f])

        let t3 = Tensor.Create([1.f; 2.f]) + 5.f
        let t3Correct = Tensor.Create([6.f; 7.f])

        Assert.AreEqual(t1, t1Correct)
        Assert.AreEqual(t2, t2Correct)
        Assert.AreEqual(t3, t3Correct)

    