namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDiffSharp () =

    let rosenBrock (x:Tensor) = (1. - x.[0])**2 + 100. * (x.[1] - x.[0]**2)**2
    let rosenBrockGrad (x:Tensor) = dsharp.stack([-2*(1-x.[0])-400*x.[0]*(-(x.[0]**2) + x.[1]); 200*(-(x.[0]**2) + x.[1])])

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
        let t = dsharp.rand([1000])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(0.5)
        let tStddev = t.stddev()
        let tStddevCorrect = dsharp.tensor(1./12.) |> dsharp.sqrt
        Assert.True(tMeanCorrect.allclose(tMean, 0.1))
        Assert.True(tStddevCorrect.allclose(tStddev, 0.1))

    [<Test>]
    member this.TestRandn () =
        let t = dsharp.randn([1000])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(0.)
        let tStddev = t.stddev()
        let tStddevCorrect = dsharp.tensor(1.)
        printfn "%A %A" tMean tMeanCorrect
        printfn "%A %A" tStddev tStddevCorrect
        Assert.True(tMeanCorrect.allclose(tMean, 0.1, 0.1))
        Assert.True(tStddevCorrect.allclose(tStddev, 0.1, 0.1))

    [<Test>]
    member this.TestSeed () =
        dsharp.seed(123)
        let t = dsharp.rand([10])
        dsharp.seed(123)
        let t2 = dsharp.rand([10])
        Assert.AreEqual(t, t2)

    [<Test>]
    member this.TestGrad () =
        let x = dsharp.tensor([1.5;2.5])
        let g = dsharp.grad rosenBrock x
        let gCorrect = rosenBrockGrad x
        printfn "%A %A" g gCorrect
        Assert.AreEqual(gCorrect, g)