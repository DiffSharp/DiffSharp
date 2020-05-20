namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Model


type ModelStyle1a() =
    inherit Model()
    let fc1 = Linear(10, 16)
    let fc2 = Linear(16, 20)
    do base.add([fc1; fc2], ["fc1"; "fc2"])
    override __.forward(x) =
        x
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward

type ModelStyle1b() =
    inherit Model()
    let fc1 = Linear(20, 32)
    let fc2 = Linear(32, 30)
    let p = Parameter(dsharp.randn([]))
    do base.add([fc1; fc2; p], ["fc1"; "fc2"; "p"])
    override __.forward(x) =
        x
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward
        |> dsharp.mul p.value

[<TestFixture>]
type TestModel () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestParameterDictFlattenUnflatten () =
        let d1t1 = Parameter <| dsharp.randn([15;5])
        let d1t2 = Parameter <| dsharp.randn(4)
        let d1 = ParameterDict()
        d1.add("w", d1t1)
        d1.add("b", d1t2)
        let d1flat = d1.flatten()
        let d1flatCorrect = dsharp.cat([d1t1.value.flatten(); d1t2.value.flatten()])
        Assert.AreEqual(d1flatCorrect, d1flat)

        let d2t1 = Parameter <| dsharp.randn([15;5])
        let d2t2 = Parameter <| dsharp.randn(4)
        let d2 = ParameterDict()
        d2.add("w", d2t1)
        d2.add("b", d2t2)
        let d2flat = d2.flatten()
        Assert.AreNotEqual(d1flatCorrect, d2flat)

        let d3 = d2.unflattenToNew(d1flat)
        let d3flat = d3.flatten()
        Assert.AreEqual(d1flatCorrect, d3flat)

    [<Test>]
    member this.TestModelCreationStyle1 () =
        let net = ModelStyle1a()
        Assert.AreEqual(516, net.nparameters)

        let net2 = ModelStyle1b()
        Assert.AreEqual(1663, net2.nparameters)

    [<Test>]
    member this.TestModelCreationStyle2 () =
        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let net = Model.create [fc1; fc2] 
                    (dsharp.view [-1; 10]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward)
        Assert.AreEqual(682, net.nparameters)

        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let p = Parameter(dsharp.randn([]))
        let net2 = Model.create [fc1; fc2; p] 
                    (dsharp.view [-1; 28*28]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward
                    >> dsharp.mul p.value)
        Assert.AreEqual(683, net2.nparameters)

    [<Test>]
    member this.TestModelCompose () =
        let net1 = ModelStyle1a()
        let net2 = ModelStyle1b()
        let net3 = Model.compose net1 net2
        Assert.AreEqual(516 + 1663, net3.nparameters)

        let x = dsharp.randn([5;10])
        let y = net3.forward(x)
        Assert.AreEqual([5;30], y.shape)

    [<Test>]
    member this.TestModelParametersDiff () =
        let net = ModelStyle1a()

        Assert.True(net.getParameters().isNoDiff())

        let p = net.getParameters()
        let p = p.forwardDiff(p.onesLike())
        net.setParameters(p)
        Assert.True(net.getParameters().isForwardDiff())

        net.noDiff()
        Assert.True(net.getParameters().isNoDiff())

        let p = net.getParameters()
        let p = p.reverseDiff()
        net.setParameters(p)
        Assert.True(net.getParameters().isReverseDiff())

        net.noDiff()
        Assert.True(net.getParameters().isNoDiff())

        let p = net.getParameters()
        let x = dsharp.randn([1;10])
        ignore <| dsharp.grad (net.forwardCompose dsharp.sum x) p
        Assert.True(net.getParameters().isNoDiff())

    [<Test>]
    member this.TestModelLinear () =
        // Trains a linear regressor
        let n, din, dout = 4, 100, 10
        let inputs  = dsharp.randn([n; din])
        let targets = dsharp.randn([n; dout])
        let net = Linear(din, dout)

        let lr, steps = 1e-2, 1000
        let loss = net.forwardLoss dsharp.mseLoss
        let mutable p = net.getParameters()
        for _ in 0..steps do
            let g = dsharp.grad (loss inputs targets) p
            p <- p - lr * g
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.01))

    [<Test>]
    member this.TestModelConv1d () =
        // Trains a little binary classifier
        let cin, din = 1, 16
        let cout = 2
        let k = 3
        let inputs  = dsharp.randn([2; cin; din])
        let conv1 = Conv1d(cin, cout, k)
        let fcin = inputs.[0] |> dsharp.unsqueeze 0 |> conv1.forward |> dsharp.nelement
        let fc1 = Linear(fcin, 2)
        let net = Model.create [conv1; fc1] (conv1.forward >> dsharp.relu >> dsharp.flatten 1 >> fc1.forward)
        let targets = dsharp.tensor([0; 1])
        let targetsp = dsharp.tensor([[1,0],[0,1]])
        let lr, steps = 1e-2, 250
        let loss = net.forwardLoss dsharp.crossEntropyLoss
        let mutable p = net.getParameters()
        for _ in 0..steps do
            let g = dsharp.grad (loss inputs targets) p
            p <- p - lr * g
        let y = inputs |> net.forward |> dsharp.softmax 1
        printfn "%A %A" targetsp y
        Assert.True(targetsp.allclose(y, 0.1, 0.1))

    [<Test>]
    member this.TestModelConv2d () =
        // Trains a little binary classifier
        let cin, hin, win = 1, 6, 6
        let cout = 2
        let k = 3
        let inputs  = dsharp.randn([2; cin; hin; win])
        let conv1 = Conv2d(cin, cout, k)
        let fcin = inputs.[0] |> dsharp.unsqueeze 0 |> conv1.forward |> dsharp.nelement
        let fc1 = Linear(fcin, 2)
        let net = Model.create [conv1; fc1] (conv1.forward >> dsharp.relu >> dsharp.flatten 1 >> fc1.forward)
        let targets = dsharp.tensor([0; 1])
        let targetsp = dsharp.tensor([[1,0],[0,1]])
        let lr, steps = 1e-2, 250
        let loss = net.forwardLoss dsharp.crossEntropyLoss
        let mutable p = net.getParameters()
        for _ in 0..steps do
            let g = dsharp.grad (loss inputs targets) p
            p <- p - lr * g
        let y = inputs |> net.forward |> dsharp.softmax 1
        Assert.True(targetsp.allclose(y, 0.1, 0.1))

    [<Test>]
    member this.TestModelConv3d () =
        // Trains a little binary classifier
        let cin, din, hin, win = 1, 6, 6, 6
        let cout = 2
        let k = 3
        let inputs  = dsharp.randn([2; cin; din; hin; win])
        let conv1 = Conv3d(cin, cout, k)
        let fcin = inputs.[0] |> dsharp.unsqueeze 0 |> conv1.forward |> dsharp.nelement
        let fc1 = Linear(fcin, 2)
        let net = Model.create [conv1; fc1] (conv1.forward >> dsharp.relu >> dsharp.flatten 1 >> fc1.forward)
        let targets = dsharp.tensor([0; 1])
        let targetsp = dsharp.tensor([[1,0],[0,1]])
        let lr, steps = 1e-2, 250
        let loss = net.forwardLoss dsharp.crossEntropyLoss
        let mutable p = net.getParameters()
        for _ in 0..steps do
            let g = dsharp.grad (loss inputs targets) p
            p <- p - lr * g
        let y = inputs |> net.forward |> dsharp.softmax 1
        Assert.True(targetsp.allclose(y, 0.1, 0.1))        