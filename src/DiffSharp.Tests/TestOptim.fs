namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim

[<TestFixture>]
type TestOptim () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestOptimSGD () =
        // Trains a linear regressor
        let n, din, dout = 4, 100, 10
        let inputs  = dsharp.randn([n; din])
        let targets = dsharp.randn([n; dout])
        let net = Linear(din, dout)

        let lr, steps = 1e-2, 1000
        let optimizer = SGD(net, learningRate=dsharp.tensor(lr))
        for _ in 0..steps do
            net.reverseDiff()
            let y = net.forward(inputs)
            let loss = dsharp.mseLoss(y, targets)
            loss.reverse()
            optimizer.step()
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.01))