namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim

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

    [<Test>]
    member _.TestParameterDictFlattenUnflatten () =
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
    member _.TestModelCreationStyle1 () =
        let net = ModelStyle1a()
        Assert.AreEqual(516, net.nparameters)

        let net2 = ModelStyle1b()
        Assert.AreEqual(1663, net2.nparameters)

    [<Test>]
    member _.TestModelCreationStyle2 () =
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
                    (dsharp.view [-1; 10]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward
                    >> dsharp.mul p.value)
        Assert.AreEqual(683, net2.nparameters)

    [<Test>]
    member _.TestModelCreationStyle3 () =
        let net = dsharp.view [-1; 10] --> Linear(10, 32) --> dsharp.relu --> Linear(32, 10)
        Assert.AreEqual(682, net.nparameters)        

    [<Test>]
    member _.TestModelUsageStyle1 () =
        let net = ModelStyle1a()
        let x = dsharp.randn([1; 10])
        let y = net.forward x |> dsharp.sin
        Assert.AreEqual([1;20], y.shape)

    [<Test>]
    member _.TestModelUsageStyle2 () =
        let net = ModelStyle1a()
        let x = dsharp.randn([1; 10])
        let y = x --> net --> dsharp.sin
        Assert.AreEqual([1;20], y.shape)

    [<Test>]
    member _.TestModelCompose () =
        let net1 = ModelStyle1a()
        let net2 = ModelStyle1b()
        let net3 = Model.compose net1 net2
        Assert.AreEqual(516 + 1663, net3.nparameters)

        let x = dsharp.randn([5;10])
        let y = net3.forward(x)
        Assert.AreEqual([5;30], y.shape)

    [<Test>]
    member _.TestModelParametersDiff () =
        let net = ModelStyle1a()

        Assert.True(net.parameters.isNoDiff())

        let p = net.parameters
        let p = p.forwardDiff(p.onesLike())
        net.parameters <- p
        Assert.True(net.parameters.isForwardDiff())

        net.noDiff()
        Assert.True(net.parameters.isNoDiff())

        let p = net.parameters
        let p = p.reverseDiff()
        net.parameters <- p
        Assert.True(net.parameters.isReverseDiff())

        net.noDiff()
        Assert.True(net.parameters.isNoDiff())

        let p = net.parameters
        let x = dsharp.randn([1;10])
        ignore <| dsharp.grad (net.forwardCompose dsharp.sum x) p
        Assert.True(net.parameters.isNoDiff())

    [<Test>]
    member _.TestModelForwardParameters () =
        let net = ModelStyle1a()
        let f = net.forwardParameters
        let p = net.parameters
        let x = dsharp.randn([1;10])
        let y = f x p
        Assert.AreEqual([1;20], y.shape)

    [<Test>]
    member _.TestModelForwardCompose () =
        let net = ModelStyle1a()
        let f = net.forwardCompose dsharp.sin
        let p = net.parameters
        let x = dsharp.randn([1;10])
        let y = f x p
        Assert.AreEqual([1;20], y.shape)

    [<Test>]
    member _.TestModelForwardLoss () =
        let net = ModelStyle1a()
        let f = net.forwardLoss dsharp.mseLoss
        let p = net.parameters
        let x = dsharp.randn([1;10])
        let t = dsharp.randn([1;20])
        let y = f x t p
        Assert.AreEqual([], y.shape)

    [<Test>]
    member _.TestModelSaveLoadParameters () =
        let net1 = ModelStyle1a()
        let p1 = net1.parameters
        let fileName = System.IO.Path.GetTempFileName()
        net1.saveParameters(fileName)

        let net2 = ModelStyle1a()
        let p2 = net2.parameters
        Assert.AreNotEqual(p1, p2)

        net2.loadParameters(fileName)
        let p2 = net2.parameters
        Assert.AreEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.AreEqual(y1, y2)

    [<Test>]
    member _.TestModelSaveLoad () =
        let net1 = ModelStyle1a()
        let p1 = net1.parameters
        let fileName = System.IO.Path.GetTempFileName()
        net1.save(fileName)

        let net2 = Model.load(fileName)
        let p2 = net2.parameters
        Assert.AreEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.AreEqual(y1, y2)

    [<Test>]
    member _.TestModelMove () =
        let confBackup = dsharp.config()
        for combo1 in Combos.FloatingPoint do
            dsharp.config(combo1.dtype, combo1.device, combo1.backend)
            let net = dsharp.view [-1; 2] --> Linear(2, 4) --> dsharp.relu --> Linear(4, 1)
            Assert.AreEqual(combo1.dtype, net.parameters.dtype)
            Assert.AreEqual(combo1.device, net.parameters.device)
            Assert.AreEqual(combo1.backend, net.parameters.backend)
            for combo2 in Combos.FloatingPoint do
                // printfn "\n%A %A" (combo1.dtype, combo1.device, combo1.backend) (combo2.dtype, combo2.device, combo2.backend)
                net.move(combo2.dtype, combo2.device, combo2.backend)
                Assert.AreEqual(combo2.dtype, net.parameters.dtype)
                Assert.AreEqual(combo2.device, net.parameters.device)
                Assert.AreEqual(combo2.backend, net.parameters.backend)
        dsharp.config(confBackup)

    [<Test>]
    member _.TestModelClone () =
        let net1 = ModelStyle1a()
        let p1 = net1.parameters

        let net2 = net1.clone()
        let p2 = net2.parameters
        Assert.AreEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.AreEqual(y1, y2)

    [<Test>]
    member _.TestModelLinear () =
        // Trains a linear regressor
        let n, din, dout = 4, 100, 10
        let inputs  = dsharp.randn([n; din])
        let targets = dsharp.randn([n; dout])
        let net = Linear(din, dout)

        let lr, steps = 1e-2, 1000
        let loss = net.forwardLoss dsharp.mseLoss
        let mutable p = net.parameters
        for _ in 0..steps do
            let g = dsharp.grad (loss inputs targets) p
            p <- p - lr * g
        let y = net.forward inputs
        Assert.True(targets.allclose(y, 0.01))

    [<Test>]
    member _.TestModelConv1d () =
        // Trains a little binary classifier
        let din, cin, cout, k = 16, 1, 2, 3
        let inputs  = dsharp.randn([2; cin; din])
        let conv1 = Conv1d(cin, cout, k)
        let fcin = inputs.[0] --> dsharp.unsqueeze 0 --> conv1 --> dsharp.nelement
        let net = conv1 --> dsharp.relu --> dsharp.flatten 1 --> Linear(fcin, 2)
        let targets = dsharp.tensor([0; 1])
        let targetsp = dsharp.tensor([[1,0],[0,1]])
        let dataset = TensorDataset(inputs, targets)
        let dataloader = dataset.loader(8, shuffle=true)        
        let lr, iters = 1e-2, 250
        Optimizer.sgd(net, dataloader, dsharp.crossEntropyLoss, lr=dsharp.tensor(lr), iters=iters)
        let y = inputs --> net --> dsharp.softmax 1
        Assert.True(targetsp.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestModelConv2d () =
        // Trains a little binary classifier
        let cin, hin, win, cout, k = 1, 6, 6, 2, 3
        let inputs  = dsharp.randn([2; cin; hin; win])
        let conv1 = Conv2d(cin, cout, k)
        let fcin = inputs.[0] --> dsharp.unsqueeze 0 --> conv1 --> dsharp.nelement
        let net = conv1 --> dsharp.relu --> dsharp.flatten 1 --> Linear(fcin, 2)
        let targets = dsharp.tensor([0; 1])
        let targetsp = dsharp.tensor([[1,0],[0,1]])
        let dataset = TensorDataset(inputs, targets)
        let dataloader = dataset.loader(8, shuffle=true)        
        let lr, iters = 1e-2, 250
        Optimizer.sgd(net, dataloader, dsharp.crossEntropyLoss, lr=dsharp.tensor(lr), iters=iters)
        let y = inputs --> net --> dsharp.softmax 1
        Assert.True(targetsp.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestModelConv3d () =
        // Trains a little binary classifier
        let cin, din, hin, win, cout, k = 1, 6, 6, 6, 2, 3
        let inputs  = dsharp.randn([2; cin; din; hin; win])
        let conv1 = Conv3d(cin, cout, k)
        let fcin = inputs.[0] --> dsharp.unsqueeze 0 --> conv1 --> dsharp.nelement
        let net = conv1 --> dsharp.relu --> dsharp.flatten 1 --> Linear(fcin, 2)
        let targets = dsharp.tensor([0; 1])
        let targetsp = dsharp.tensor([[1,0],[0,1]])
        let dataset = TensorDataset(inputs, targets)
        let dataloader = dataset.loader(8, shuffle=true)        
        let lr, iters = 1e-2, 250
        Optimizer.sgd(net, dataloader, dsharp.crossEntropyLoss, lr=dsharp.tensor(lr), iters=iters)
        let y = inputs --> net --> dsharp.softmax 1
        Assert.True(targetsp.allclose(y, 0.1, 0.1))