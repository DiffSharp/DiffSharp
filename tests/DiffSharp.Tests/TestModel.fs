namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Compose
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
        Assert.CheckEqual(d1flatCorrect, d1flat)

        let d2t1 = Parameter <| dsharp.randn([15;5])
        let d2t2 = Parameter <| dsharp.randn(4)
        let d2 = ParameterDict()
        d2.add("w", d2t1)
        d2.add("b", d2t2)
        let d2flat = d2.flatten()
        Assert.AreNotEqual(d1flatCorrect, d2flat)

        let d3 = d2.unflattenToNew(d1flat)
        let d3flat = d3.flatten()
        Assert.CheckEqual(d1flatCorrect, d3flat)

    [<Test>]
    member _.TestModelCreationStyle1 () =
        let net = ModelStyle1a()
        Assert.CheckEqual(516, net.nparameters)

        let net2 = ModelStyle1b()
        Assert.CheckEqual(1663, net2.nparameters)

    [<Test>]
    member _.TestModelCreationStyle2 () =
        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let net = Model.create [fc1; fc2] 
                    (dsharp.view [-1; 10]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward)
        Assert.CheckEqual(682, net.nparameters)

        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let p = Parameter(dsharp.randn([]))
        let net2 = Model.create [fc1; fc2; p] 
                    (dsharp.view [-1; 10]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward
                    >> dsharp.mul p.value)
        Assert.CheckEqual(683, net2.nparameters)

    [<Test>]
    member _.TestModelCreationStyle3 () =
        let net = dsharp.view [-1; 10] --> Linear(10, 32) --> dsharp.relu --> Linear(32, 10)
        Assert.CheckEqual(682, net.nparameters)        

    [<Test>]
    member _.TestModelUsageStyle1 () =
        let net = ModelStyle1a()
        let x = dsharp.randn([1; 10])
        let y = net.forward x |> dsharp.sin
        Assert.CheckEqual([| 1;20 |], y.shape)

    [<Test>]
    member _.TestModelUsageStyle2 () =
        let net = ModelStyle1a()
        let x = dsharp.randn([1; 10])
        let y = x --> net --> dsharp.sin
        Assert.CheckEqual([| 1;20 |], y.shape)

    [<Test>]
    member _.TestModelCompose () =
        let net1 = ModelStyle1a()
        let net2 = ModelStyle1b()
        let net3 = Model.compose net1 net2
        Assert.CheckEqual(516 + 1663, net3.nparameters)

        let x = dsharp.randn([5;10])
        let y = net3.forward(x)
        Assert.CheckEqual([|5;30|], y.shape)

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
        Assert.CheckEqual([|1;20|], y.shape)

    [<Test>]
    member _.TestModelForwardCompose () =
        let net = ModelStyle1a()
        let f = net.forwardCompose dsharp.sin
        let p = net.parameters
        let x = dsharp.randn([1;10])
        let y = f x p
        Assert.CheckEqual([|1;20|], y.shape)

    [<Test>]
    member _.TestModelForwardLoss () =
        let net = ModelStyle1a()
        let f = net.forwardLoss dsharp.mseLoss
        let p = net.parameters
        let x = dsharp.randn([1;10])
        let t = dsharp.randn([1;20])
        let y = f x t p
        Assert.CheckEqual(([| |]: int array), y.shape)

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
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelSaveLoad () =
        let net1 = ModelStyle1a()
        let p1 = net1.parameters
        let fileName = System.IO.Path.GetTempFileName()
        net1.save(fileName)

        let net2 = Model.load(fileName)
        let p2 = net2.parameters
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelMove () =
        let confBackup = dsharp.config()
        for combo1 in Combos.FloatingPoint do
            dsharp.config(combo1.dtype, combo1.device, combo1.backend)
            let net = dsharp.view [-1; 2] --> Linear(2, 4) --> dsharp.relu --> Linear(4, 1)
            Assert.CheckEqual(combo1.dtype, net.parameters.dtype)
            Assert.CheckEqual(combo1.device, net.parameters.device)
            Assert.CheckEqual(combo1.backend, net.parameters.backend)
            for combo2 in Combos.FloatingPoint do
                // printfn "\n%A %A" (combo1.dtype, combo1.device, combo1.backend) (combo2.dtype, combo2.device, combo2.backend)
                net.move(combo2.dtype, combo2.device, combo2.backend)
                Assert.CheckEqual(combo2.dtype, net.parameters.dtype)
                Assert.CheckEqual(combo2.device, net.parameters.device)
                Assert.CheckEqual(combo2.backend, net.parameters.backend)
        dsharp.config(confBackup)

    [<Test>]
    member _.TestModelClone () =
        let net1 = ModelStyle1a()
        let p1 = net1.parameters

        let net2 = net1.clone()
        let p2 = net2.parameters
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelTrainEval () =
        let m = Linear(1, 2) --> Linear(2, 3) --> Linear(3, 4)
        Assert.CheckEqual(Mode.Train, m.mode)
        Assert.CheckEqual(Mode.Train, m.allModels.[0].mode)
        Assert.CheckEqual(Mode.Train, m.allModels.[1].mode)
        Assert.CheckEqual(Mode.Train, m.allModels.[2].mode)

        m.eval()
        Assert.CheckEqual(Mode.Eval, m.mode)
        Assert.CheckEqual(Mode.Eval, m.allModels.[0].mode)
        Assert.CheckEqual(Mode.Eval, m.allModels.[1].mode)
        Assert.CheckEqual(Mode.Eval, m.allModels.[2].mode)

        m.train()
        Assert.CheckEqual(Mode.Train, m.mode)
        Assert.CheckEqual(Mode.Train, m.allModels.[0].mode)
        Assert.CheckEqual(Mode.Train, m.allModels.[1].mode)
        Assert.CheckEqual(Mode.Train, m.allModels.[2].mode)

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
        optim.sgd(net, dataloader, dsharp.crossEntropyLoss, lr=dsharp.tensor(lr), iters=iters)
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
        optim.sgd(net, dataloader, dsharp.crossEntropyLoss, lr=dsharp.tensor(lr), iters=iters)
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
        optim.sgd(net, dataloader, dsharp.crossEntropyLoss, lr=dsharp.tensor(lr), iters=iters)
        let y = inputs --> net --> dsharp.softmax 1
        Assert.True(targetsp.allclose(y, 0.1, 0.1))

    [<Test>]
    member _.TestModelConvTranspose1d () =
        let x = dsharp.randn([5; 3; 12])
        let m = ConvTranspose1d(3, 4, 3)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|5; 4; 14|]
        Assert.CheckEqual(yShapeCorrect, yShape)

        let x = dsharp.randn([3; 3; 12])
        let m = ConvTranspose1d(3, 5, 2, dilation=5)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|3; 5; 17|]
        Assert.CheckEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestModelConvTranspose2d () =
        let x = dsharp.randn([3; 3; 12; 6])
        let m = ConvTranspose2d(3, 5, 3)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|3; 5; 14; 8|]
        Assert.CheckEqual(yShapeCorrect, yShape)

        let x = dsharp.randn([2; 3; 12; 6])
        let m = ConvTranspose2d(3, 1, 5, stride=2)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|2; 1; 27; 15|]
        Assert.CheckEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestModelConvTranspose3d () =
        let x = dsharp.randn([2; 3; 12; 6; 6])
        let m = ConvTranspose3d(3, 2, 3)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|2; 2; 14; 8; 8|]
        Assert.CheckEqual(yShapeCorrect, yShape)

        let x = dsharp.randn([2; 3; 12; 6; 6])
        let m = ConvTranspose3d(3, 2, 2, padding=1)
        let y = x --> m
        let yShape = y.shape
        let yShapeCorrect = [|2; 2; 11; 5; 5|]
        Assert.CheckEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestModelDropout () =
        let m = Dropout(1.)
        let x = dsharp.randn([10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.CheckEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.CheckEqual(x, xeval)

    [<Test>]
    member _.TestModelDropout2d () =
        let m = Dropout2d(1.)
        let x = dsharp.randn([10;4;10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.CheckEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.CheckEqual(x, xeval)

    [<Test>]
    member _.TestModelDropout3d () =
        let m = Dropout3d(1.)
        let x = dsharp.randn([10;4;10;10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.CheckEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.CheckEqual(x, xeval)

    [<Test>]
    member _.TestModelBatchNorm1d () =
        let m = BatchNorm1d(6, momentum=dsharp.tensor(0.1), trackRunningStats=true)
        let x = dsharp.tensor([[ -16.5297,  232.6709,  -52.6495,   54.6292,   49.3702, -166.4819],
                                [  97.1978,  -91.1589,  -27.9064,  -20.3609,  -32.6582, -171.3310],
                                [  63.8823,  129.6659, -114.9046,   12.8199, -210.7685, -167.7993],
                                [  54.1845,   86.1106, -153.1860,   69.3225,   89.2402,   61.2882]])

        m.train()
        let z0 = x --> m
        let z0Correct = dsharp.tensor([[-1.5984,  1.2252,  0.6961,  0.7234,  0.6557, -0.5566],
                                        [ 1.1470, -1.5425,  1.1952, -1.4017, -0.0560, -0.6053],
                                        [ 0.3428,  0.3448, -0.5596, -0.4614, -1.6012, -0.5698],
                                        [ 0.1087, -0.0274, -1.3318,  1.1398,  1.0016,  1.7318]])
        let mean0 = m.mean
        let mean0Correct = dsharp.tensor([  4.9684,   8.9322,  -8.7162,   2.9103,  -2.6204, -11.1081])
        let var0 = m.variance
        let var0Correct = dsharp.tensor([ 229.6887, 1826.2400,  328.6053,  166.9337, 1772.3845, 1321.8143])
        let weight0 = m.weight
        let weight0Correct = dsharp.tensor([1., 1., 1., 1., 1., 1.])
        let bias0 = m.bias
        let bias0Correct = dsharp.tensor([0., 0., 0., 0., 0., 0.])

        Assert.True(z0Correct.allclose(z0, 0.1, 0.1))
        Assert.True(mean0Correct.allclose(mean0, 0.1, 0.1))
        Assert.True(var0Correct.allclose(var0, 0.1, 0.1))
        Assert.True(weight0Correct.allclose(weight0, 0.1, 0.1))
        Assert.True(bias0Correct.allclose(bias0, 0.1, 0.1))

        let optimizer = SGD(m)
        for _=1 to 99 do
            m.reverseDiff()
            let z = x --> m
            dsharp.mseLoss(z, x).reverse()
            optimizer.step()

        let z100 = x --> m
        let z100Correct = dsharp.tensor([[-2.0832,  8.7395, -1.0358,  2.4735,  2.2373, -5.9437],
                                            [ 4.2655, -4.4521,  0.2504, -2.0172, -1.1145, -6.1482],
                                            [ 2.4057,  4.5434, -4.2720, -0.0302, -8.3922, -5.9992],
                                            [ 1.8643,  2.7692, -6.2620,  3.3534,  3.8664,  3.6654]])
        let mean100 = m.mean
        let mean100Correct = dsharp.tensor([  49.6825,   89.3200,  -87.1595,   29.1020,  -26.2034, -111.0784])
        let var100 = m.variance
        let var100Correct = dsharp.tensor([ 2287.8325, 18252.9648,  3276.9749,  1660.2977, 17714.4199, 13208.8271])
        let weight100 = m.weight
        let weight100Correct = dsharp.tensor([2.3124, 4.7663, 2.5771, 2.1132, 4.7098, 4.1991])
        let bias100 = m.bias
        let bias100Correct = dsharp.tensor([ 1.6131,  2.9000, -2.8299,  0.9449, -0.8508, -3.6064])

        Assert.True(z100Correct.allclose(z100, 0.1, 0.1))
        Assert.True(mean100Correct.allclose(mean100, 0.1, 0.1))
        Assert.True(var100Correct.allclose(var100, 0.1, 0.1))
        Assert.True(weight100Correct.allclose(weight100, 0.1, 0.1))
        Assert.True(bias100Correct.allclose(bias100, 0.1, 0.1))

        m.eval()
        let zEval = x --> m
        let zEvalCorrect = dsharp.tensor([[-1.5880,  7.9572, -1.2762,  2.2688,  1.8236, -5.6307],
                                            [ 3.9102, -3.4671, -0.1623, -1.6204, -1.0792, -5.8078],
                                            [ 2.2996,  4.3234, -4.0789,  0.1004, -7.3819, -5.6788],
                                            [ 1.8307,  2.7868, -5.8023,  3.0308,  3.2344,  2.6911]])

        Assert.True(zEvalCorrect.allclose(zEval, 0.1, 0.1))

    [<Test>]
    member _.TestModelBatchNorm1dWithChannel () =
        let m = BatchNorm1d(3, momentum=dsharp.tensor(0.1), trackRunningStats=true)
        let x = dsharp.tensor([[[-149.1423,  -30.7808, -130.7123,  118.5613,  -50.9501],
                                 [  25.7468, -160.6043,  -70.1356,  -11.4244,  114.6217],
                                 [  97.1559,  -37.8110, -206.4251,   -8.9415,  -76.7563]],

                                [[ 178.9990,  152.6738,   66.8462,  146.0069,   34.9528],
                                 [  31.2735,   60.0766,  124.1946,  -95.0830,   26.5270],
                                 [   7.8316,  -22.9795, -105.4548,  168.2986, -144.0582]],

                                [[  -0.9826,   66.9230,  -68.1120,   85.3303, -135.3473],
                                 [  -5.7174,    6.7239,  -89.4217,  -99.6134, -282.3056],
                                 [  66.7493,  -23.1203,   13.9502,  -43.4741,  -87.9603]],

                                [[  30.9997,   31.3321,  111.1568,   79.0237,  -29.4414],
                                 [  25.0114,  -51.1452,  -78.8680, -183.9044,  -48.7126],
                                 [ 184.8902, -119.3005,  -19.2157,  -49.4947, -139.3300]]])

        m.train()
        let z0 = x --> m
        let z0Correct = dsharp.tensor([[[-1.8266, -0.5877, -1.6337,  0.9755, -0.7988],
                                         [ 0.6578, -1.2609, -0.3295,  0.2750,  1.5728],
                                         [ 1.2738, -0.1079, -1.8341,  0.1877, -0.5066]],

                                        [[ 1.6081,  1.3325,  0.4342,  1.2628,  0.1003],
                                         [ 0.7147,  1.0112,  1.6714, -0.5863,  0.6658],
                                         [ 0.3594,  0.0439, -0.8004,  2.0022, -1.1956]],

                                        [[-0.2758,  0.4350, -0.9785,  0.6276, -1.6822],
                                         [ 0.3338,  0.4619, -0.5280, -0.6330, -2.5140],
                                         [ 0.9626,  0.0425,  0.4220, -0.1659, -0.6213]],

                                        [[ 0.0590,  0.0624,  0.8980,  0.5616, -0.5737],
                                         [ 0.6502, -0.1339, -0.4194, -1.5008, -0.1089],
                                         [ 2.1720, -0.9421,  0.0825, -0.2275, -1.1472]]])
        let mean0 = m.mean
        let mean0Correct = dsharp.tensor([ 2.5367, -3.8138, -2.7272])
        let var0 = m.variance
        let var0Correct = dsharp.tensor([ 961.6732,  993.8416, 1005.2363])
        let weight0 = m.weight
        let weight0Correct = dsharp.tensor([1., 1., 1.])
        let bias0 = m.bias
        let bias0Correct = dsharp.tensor([0., 0., 0.])

        Assert.True(z0Correct.allclose(z0, 0.1, 0.1))
        Assert.True(mean0Correct.allclose(mean0, 0.1, 0.1))
        Assert.True(var0Correct.allclose(var0, 0.1, 0.1))
        Assert.True(weight0Correct.allclose(weight0, 0.1, 0.1))
        Assert.True(bias0Correct.allclose(bias0, 0.1, 0.1))

        let optimizer = SGD(m)
        for _=1 to 99 do
            m.reverseDiff()
            let z = x --> m
            dsharp.mseLoss(z, x).reverse()
            optimizer.step()

        let z100 = x --> m
        let z100Correct = dsharp.tensor([[[-11.2386,  -2.5167,  -9.8805,   8.4880,  -4.0030],
                                             [  2.2607, -11.4413,  -4.7893,  -0.4724,   8.7955],
                                             [  7.3997,  -2.5167, -14.9054,  -0.3956,  -5.3782]],

                                            [[ 12.9415,  11.0017,   4.6772,  10.5104,   2.3271],
                                             [  2.6671,   4.7849,   9.4994,  -6.6237,   2.3181],
                                             [  0.8368,  -1.4270,  -7.4867,  12.6268, -10.3231]],

                                            [[ -0.3210,   4.6829,  -5.2676,   6.0393, -10.2220],
                                             [ -0.0528,   0.8620,  -6.2074,  -6.9568, -20.3898],
                                             [  5.1657,  -1.4374,   1.2863,  -2.9328,  -6.2014]],

                                            [[  2.0358,   2.0602,   7.9424,   5.5746,  -2.4180],
                                             [  2.2066,  -3.3930,  -5.4314, -13.1546,  -3.2142],
                                             [ 13.8458,  -8.5040,  -1.1505,  -3.3752,  -9.9757]]])
        let mean100 = m.mean
        let mean100Correct = dsharp.tensor([ 25.3662, -38.1371, -27.2717])
        let var100 = m.variance
        let var100Correct = dsharp.tensor([ 9607.5020,  9929.1797, 10043.1221])
        let weight100 = m.weight
        let weight100Correct = dsharp.tensor([7.0400, 7.1413, 7.1768])
        let bias100 = m.bias
        let bias100Correct = dsharp.tensor([ 1.6207, -2.4366, -1.7424])

        Assert.True(z100Correct.allclose(z100, 0.1, 0.1))
        Assert.True(mean100Correct.allclose(mean100, 0.1, 0.1))
        Assert.True(var100Correct.allclose(var100, 0.1, 0.1))
        Assert.True(weight100Correct.allclose(weight100, 0.1, 0.1))
        Assert.True(bias100Correct.allclose(bias100, 0.1, 0.1))

        m.eval()
        let zEval = x --> m
        let zEvalCorrect = dsharp.tensor([[[-10.9131,  -2.4120,  -9.5894,   8.3142,  -3.8606],
                                             [  2.1418, -11.2135,  -4.7299,  -0.5222,   8.5112],
                                             [  7.1683,  -2.4972, -14.5723,  -0.4297,  -5.2862]],

                                            [[ 12.6551,  10.7643,   4.5999,  10.2855,   2.3092],
                                             [  2.5378,   4.6021,   9.1972,  -6.5178,   2.1977],
                                             [  0.7715,  -1.4350,  -7.3414,  12.2631, -10.1059]],

                                            [[ -0.2718,   4.6054,  -5.0932,   5.9275,  -9.9223],
                                             [ -0.1132,   0.7784,  -6.1120,  -6.8425, -19.9355],
                                             [  4.9908,  -1.4451,   1.2096,  -2.9027,  -6.0886]],

                                            [[  2.0253,   2.0492,   7.7824,   5.4745,  -2.3158],
                                             [  2.0890,  -3.3689,  -5.3557, -12.8834,  -3.1945],
                                             [ 13.4513,  -8.3330,  -1.1655,  -3.3339,  -9.7673]]])

        Assert.True(zEvalCorrect.allclose(zEval, 0.1, 0.1))

    [<Test>]
    member _.TestModelBatchNorm2d () =
        let m = BatchNorm2d(3, momentum=dsharp.tensor(0.1), trackRunningStats=true)
        let x = dsharp.tensor([[[[-7.7716e+01,  9.5762e+01,  1.0315e+02,  1.1872e+01],
                                  [ 2.1943e+02, -7.1335e+01, -1.8787e+01,  1.9394e+02],
                                  [-1.0419e+02,  1.1854e+02, -1.4793e+02, -4.1594e+01],
                                  [ 1.3719e+02,  8.9766e+01,  1.0997e+02, -5.9439e+00]],

                                 [[ 8.3717e+01, -2.3541e+02, -2.6503e+02,  3.4806e+01],
                                  [ 4.1034e+01, -9.1234e+00,  1.3147e+02,  9.3037e+01],
                                  [-8.2944e+00, -1.1279e+02,  3.1441e-01, -5.8712e+01],
                                  [ 1.1284e-01, -1.2127e+02,  9.6771e+01,  4.4762e+00]],

                                 [[ 9.3470e+01,  1.3881e+02,  3.9392e+01,  1.3340e+02],
                                  [ 6.0517e+00, -1.5608e+02, -6.1606e+01,  1.2483e+02],
                                  [-1.5972e+02, -4.5167e+01, -8.0421e+01, -1.7988e+01],
                                  [ 5.2329e+01,  1.5187e+02,  9.1073e+01,  2.0072e+01]]],


                                [[[-2.9044e+02, -9.9492e+01, -8.4986e+01,  1.0289e+01],
                                  [ 7.1335e+01, -6.5161e+01, -5.8109e+01,  4.9377e+01],
                                  [-6.3812e+01,  2.7927e+01, -5.5035e+01,  5.4773e+00],
                                  [ 4.4723e+00, -2.0010e+01,  1.7842e+02, -1.7935e+00]],

                                 [[-1.2239e+02,  8.2393e+01,  4.5658e+01, -4.0712e+01],
                                  [ 6.7032e+01,  1.1340e+02, -2.5236e+01,  3.5875e+01],
                                  [ 1.9278e+02, -8.7840e+01,  1.1571e+01, -6.2274e+01],
                                  [ 6.6944e+01, -3.3211e+01,  6.3044e+01,  1.0125e+02]],

                                 [[-9.0048e+01, -1.4724e+02, -6.4564e+01,  9.1224e+01],
                                  [-1.3021e+02, -5.8248e+01, -8.4391e+01, -9.6276e+01],
                                  [-1.0409e+02,  7.0371e+01,  3.5041e+01,  1.3038e+02],
                                  [-4.5460e+01, -9.2695e+01, -4.4244e+01, -9.9095e+01]]]])

        m.train()
        let z0 = x --> m
        let z0Correct = dsharp.tensor([[[[-0.7976,  0.8377,  0.9074,  0.0469],
                                          [ 2.0035, -0.7374, -0.2421,  1.7632],
                                          [-1.0472,  1.0524, -1.4594, -0.4571],
                                          [ 1.2283,  0.7812,  0.9717, -0.1210]],

                                         [[ 0.8173, -2.3983, -2.6967,  0.3245],
                                          [ 0.3872, -0.1182,  1.2985,  0.9112],
                                          [-0.1098, -1.1628, -0.0231, -0.6178],
                                          [-0.0251, -1.2481,  0.9488,  0.0188]],

                                         [[ 1.1120,  1.5879,  0.5444,  1.5312],
                                          [ 0.1945, -1.5073, -0.5157,  1.4412],
                                          [-1.5455, -0.3431, -0.7132, -0.0579],
                                          [ 0.6802,  1.7250,  1.0869,  0.3416]]],


                                        [[[-2.8029, -1.0029, -0.8661,  0.0320],
                                          [ 0.6075, -0.6792, -0.6128,  0.4005],
                                          [-0.6665,  0.1983, -0.5838, -0.0133],
                                          [-0.0228, -0.2536,  1.6169, -0.0819]],

                                         [[-1.2595,  0.8039,  0.4338, -0.4365],
                                          [ 0.6492,  1.1164, -0.2805,  0.3352],
                                          [ 1.9162, -0.9113,  0.0903, -0.6537],
                                          [ 0.6483, -0.3609,  0.6090,  0.9939]],

                                         [[-0.8142, -1.4146, -0.5467,  1.0885],
                                          [-1.2357, -0.4804, -0.7548, -0.8796],
                                          [-0.9616,  0.8696,  0.4987,  1.4994],
                                          [-0.3462, -0.8420, -0.3334, -0.9092]]]])
        let mean0 = m.mean
        let mean0Correct = dsharp.tensor([ 0.6893,  0.2606, -1.2476])
        let var0 = m.variance
        let var0Correct = dsharp.tensor([1162.5181, 1017.6503,  937.8387])
        let weight0 = m.weight
        let weight0Correct = dsharp.tensor([1., 1., 1.])
        let bias0 = m.bias
        let bias0Correct = dsharp.tensor([0., 0., 0.])

        Assert.True(z0Correct.allclose(z0, 0.1, 0.1))
        Assert.True(mean0Correct.allclose(mean0, 0.1, 0.1))
        Assert.True(var0Correct.allclose(var0, 0.1, 0.1))
        Assert.True(weight0Correct.allclose(weight0, 0.1, 0.1))
        Assert.True(bias0Correct.allclose(bias0, 0.1, 0.1))

        let optimizer = SGD(m)
        for _=1 to 99 do
            m.reverseDiff()
            let z = x --> m
            dsharp.mseLoss(z, x).reverse()
            optimizer.step()

        let z100 = x --> m
        let z100Correct = dsharp.tensor([[[[-5.7119e+00,  6.9024e+00,  7.4399e+00,  8.0244e-01],
                                              [ 1.5895e+01, -5.2479e+00, -1.4269e+00,  1.4041e+01],
                                              [-7.6372e+00,  8.5584e+00, -1.0817e+01, -3.0853e+00],
                                              [ 9.9147e+00,  6.4664e+00,  7.9359e+00, -4.9303e-01]],

                                             [[ 6.1137e+00, -1.7286e+01, -1.9457e+01,  2.5275e+00],
                                              [ 2.9841e+00, -6.9353e-01,  9.6153e+00,  6.7971e+00],
                                              [-6.3274e-01, -8.2949e+00, -1.5241e-03, -4.3295e+00],
                                              [-1.6303e-02, -8.9161e+00,  7.0709e+00,  3.0363e-01]],

                                             [[ 7.0128e+00,  1.0355e+01,  3.0264e+00,  9.9564e+00],
                                              [ 5.6869e-01, -1.1383e+01, -4.4187e+00,  9.3243e+00],
                                              [-1.1651e+01, -3.2069e+00, -5.8057e+00, -1.2034e+00],
                                              [ 3.9801e+00,  1.1318e+01,  6.8361e+00,  1.6022e+00]]],


                                            [[[-2.1180e+01, -7.2953e+00, -6.2405e+00,  6.8729e-01],
                                              [ 5.1262e+00, -4.7989e+00, -4.2862e+00,  3.5296e+00],
                                              [-4.7009e+00,  1.9699e+00, -4.0627e+00,  3.3745e-01],
                                              [ 2.6437e-01, -1.5158e+00,  1.2913e+01, -1.9124e-01]],

                                             [[-8.9988e+00,  6.0166e+00,  3.3231e+00, -3.0096e+00],
                                              [ 4.8903e+00,  8.2905e+00, -1.8749e+00,  2.6059e+00],
                                              [ 1.4110e+01, -6.4652e+00,  8.2383e-01, -4.5906e+00],
                                              [ 4.8839e+00, -2.4597e+00,  4.5979e+00,  7.3993e+00]],

                                             [[-6.5154e+00, -1.0731e+01, -4.6368e+00,  6.8472e+00],
                                              [-9.4756e+00, -4.1712e+00, -6.0983e+00, -6.9745e+00],
                                              [-7.5502e+00,  5.3100e+00,  2.7056e+00,  9.7333e+00],
                                              [-3.2285e+00, -6.7105e+00, -3.1389e+00, -7.1823e+00]]]])
        let mean100 = m.mean
        let mean100Correct = dsharp.tensor([  6.8930,   2.6056, -12.4755])
        let var100 = m.variance
        let var100Correct = dsharp.tensor([11615.9023, 10167.2607,  9369.1621])
        let weight100 = m.weight
        let weight100Correct = dsharp.tensor([7.7136, 7.2769, 7.0230])
        let bias100 = m.bias
        let bias100Correct = dsharp.tensor([ 0.4404,  0.1665, -0.7971])

        Assert.True(z100Correct.allclose(z100, 0.1, 0.1))
        Assert.True(mean100Correct.allclose(mean100, 0.1, 0.1))
        Assert.True(var100Correct.allclose(var100, 0.1, 0.1))
        Assert.True(weight100Correct.allclose(weight100, 0.1, 0.1))
        Assert.True(bias100Correct.allclose(bias100, 0.1, 0.1))

        m.eval()
        let zEval = x --> m
        let zEvalCorrect = dsharp.tensor([[[[-5.6150e+00,  6.8007e+00,  7.3297e+00,  7.9676e-01],
                                              [ 1.5652e+01, -5.1584e+00, -1.3975e+00,  1.3827e+01],
                                              [-7.5101e+00,  8.4307e+00, -1.0640e+01, -3.0298e+00],
                                              [ 9.7656e+00,  6.3716e+00,  7.8179e+00, -4.7833e-01]],

                                             [[ 6.0201e+00, -1.7011e+01, -1.9149e+01,  2.4904e+00],
                                              [ 2.9398e+00, -6.7999e-01,  9.4666e+00,  6.6927e+00],
                                              [-6.2016e-01, -8.1617e+00,  1.1242e-03, -4.2587e+00],
                                              [-1.3422e-02, -8.7731e+00,  6.9622e+00,  3.0147e-01]],

                                             [[ 6.8898e+00,  1.0179e+01,  2.9662e+00,  9.7872e+00],
                                              [ 5.4717e-01, -1.1216e+01, -4.3617e+00,  9.1650e+00],
                                              [-1.1480e+01, -3.1690e+00, -5.7269e+00, -1.1970e+00],
                                              [ 3.9049e+00,  1.1127e+01,  6.7159e+00,  1.5644e+00]]],


                                            [[[-2.0840e+01, -7.1736e+00, -6.1354e+00,  6.8342e-01],
                                              [ 5.0525e+00, -4.7165e+00, -4.2118e+00,  3.4810e+00],
                                              [-4.6200e+00,  1.9458e+00, -3.9918e+00,  3.3908e-01],
                                              [ 2.6715e-01, -1.4850e+00,  1.2716e+01, -1.8129e-01]],

                                             [[-8.8545e+00,  5.9246e+00,  3.2735e+00, -2.9597e+00],
                                              [ 4.8160e+00,  8.1626e+00, -1.8428e+00,  2.5675e+00],
                                              [ 1.3891e+01, -6.3608e+00,  8.1349e-01, -4.5157e+00],
                                              [ 4.8096e+00, -2.4184e+00,  4.5282e+00,  7.2855e+00]],

                                             [[-6.4254e+00, -1.0575e+01, -4.5764e+00,  6.7269e+00],
                                              [-9.3390e+00, -4.1181e+00, -6.0149e+00, -6.8773e+00],
                                              [-7.4440e+00,  5.2139e+00,  2.6505e+00,  9.5676e+00],
                                              [-3.1903e+00, -6.6174e+00, -3.1020e+00, -7.0818e+00]]]])

        Assert.True(zEvalCorrect.allclose(zEval, 0.1, 0.1))

    [<Test>]
    member _.TestModelBatchNorm3d () =
        let m = BatchNorm3d(3, momentum=dsharp.tensor(0.1), trackRunningStats=true)
        let x = dsharp.tensor([[[[  -1.9917, -125.1875],
                                   [ -10.8246,   -0.6371]],

                                  [[ -29.9101,   62.9125],
                                   [-103.9648,  -40.4188]]],


                                 [[[   2.1155, -179.4632],
                                   [  14.3901,   79.1110]],

                                  [[ 256.2570,  110.3948],
                                   [  66.7616,  105.1888]]],


                                 [[[-122.7142,  120.5997],
                                   [  72.4510,  101.4663]],

                                  [[   9.6043,  143.2797],
                                   [   2.2688, -127.6234]]]]).unsqueeze(0)

        m.train()
        let z0 = x --> m
        let z0Correct = dsharp.tensor([[[[ 0.5206, -1.6712],
                                           [ 0.3634,  0.5447]],

                                          [[ 0.0239,  1.6753],
                                           [-1.2936, -0.1631]]],


                                         [[[-0.4750, -2.0509],
                                           [-0.3685,  0.1933]],

                                          [[ 1.7307,  0.4648],
                                           [ 0.0861,  0.4196]]],


                                         [[[-1.5039,  0.9747],
                                           [ 0.4842,  0.7798]],

                                          [[-0.1560,  1.2057],
                                           [-0.2307, -1.5539]]]]).unsqueeze(0)
        let mean0 = m.mean
        let mean0Correct = dsharp.tensor([-3.1253,  5.6844,  2.4917])
        let var0 = m.variance
        let var0Correct = dsharp.tensor([ 361.9645, 1518.0892, 1102.2589])
        let weight0 = m.weight
        let weight0Correct = dsharp.tensor([1., 1., 1.])
        let bias0 = m.bias
        let bias0Correct = dsharp.tensor([0., 0., 0.])

        Assert.True(z0Correct.allclose(z0, 0.1, 0.1))
        Assert.True(mean0Correct.allclose(mean0, 0.1, 0.1))
        Assert.True(var0Correct.allclose(var0, 0.1, 0.1))
        Assert.True(weight0Correct.allclose(weight0, 0.1, 0.1))
        Assert.True(bias0Correct.allclose(bias0, 0.1, 0.1))

        let optimizer = SGD(m)
        for _=1 to 99 do
            m.reverseDiff()
            let z = x --> m
            dsharp.mseLoss(z, x).reverse()
            optimizer.step()

        let z100 = x --> m
        let z100Correct = dsharp.tensor([[[[  0.3601,  -9.5626],
                                           [ -0.3514,   0.4692]],

                                          [[ -1.8886,   5.5877],
                                           [ -7.8533,  -2.7350]]],


                                         [[[ -0.3095, -13.3858],
                                           [  0.5745,   5.2353]],

                                          [[ 17.9923,   7.4882],
                                           [  4.3460,   7.1133]]],


                                         [[[ -9.2480,   8.6175],
                                           [  5.0822,   7.2126]],

                                          [[  0.4676,  10.2828],
                                           [ -0.0710,  -9.6084]]]]).unsqueeze(0)
        let mean100 = m.mean
        let mean100Correct = dsharp.tensor([-31.2520,  56.8431,  24.9159])
        let var100 = m.variance
        let var100Correct = dsharp.tensor([ 3610.5591, 15171.5293, 11013.3271])
        let weight100 = m.weight
        let weight100Correct = dsharp.tensor([4.5272, 8.2974, 7.2080])
        let bias100 = m.bias
        let bias100Correct = dsharp.tensor([-1.9967,  3.6318,  1.5919])

        Assert.True(z100Correct.allclose(z100, 0.1, 0.1))
        Assert.True(mean100Correct.allclose(mean100, 0.1, 0.1))
        Assert.True(var100Correct.allclose(var100, 0.1, 0.1))
        Assert.True(weight100Correct.allclose(weight100, 0.1, 0.1))
        Assert.True(bias100Correct.allclose(bias100, 0.1, 0.1))

        m.eval()
        let zEval = x --> m
        let zEvalCorrect = dsharp.tensor([[[[  0.2078,  -9.0741],
                                               [ -0.4577,   0.3099]],

                                              [[ -1.8956,   5.0979],
                                               [ -7.4751,  -2.6874]]],


                                             [[[ -0.0549, -12.2868],
                                               [  0.7720,   5.1318]],

                                              [[ 17.0651,   7.2392],
                                               [  4.2999,   6.8885]]],


                                             [[[ -8.5479,   8.1639],
                                               [  4.8568,   6.8497]],

                                              [[  0.5402,   9.7216],
                                               [  0.0364,  -8.8851]]]]).unsqueeze(0)

        Assert.True(zEvalCorrect.allclose(zEval, 0.1, 0.1))