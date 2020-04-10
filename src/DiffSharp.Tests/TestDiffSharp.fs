namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDiffSharp () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestZeros () =
        let t = dsharp.zeros([2;3])
        let tCorrect = dsharp.tensor([[0,0,0],[0,0,0]])
        Assert.AreEqual(tCorrect, t)

    [<Test>]
    member this.TestOnes () =
        let t = dsharp.ones([2;3])
        let tCorrect = dsharp.tensor([[1,1,1],[1,1,1]])
        Assert.AreEqual(tCorrect, t)

    [<Test>]
    member this.TestRand () =
        dsharp.seed(123)
        let t = dsharp.rand([2;3])
        let tCorrect = dsharp.tensor([[0.984557, 0.907815, 0.743546], 
                                        [0.811642, 0.738779, 0.048315]])
        Assert.True(tCorrect.allclose(t))
