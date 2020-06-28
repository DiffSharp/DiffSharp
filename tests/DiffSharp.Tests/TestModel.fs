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
    member _.TestModelTrainEval () =
        let m = Linear(1, 2) --> Linear(2, 3) --> Linear(3, 4)
        Assert.AreEqual(Mode.Train, m.mode)
        Assert.AreEqual(Mode.Train, m.allModels.[0].mode)
        Assert.AreEqual(Mode.Train, m.allModels.[1].mode)
        Assert.AreEqual(Mode.Train, m.allModels.[2].mode)

        m.eval()
        Assert.AreEqual(Mode.Eval, m.mode)
        Assert.AreEqual(Mode.Eval, m.allModels.[0].mode)
        Assert.AreEqual(Mode.Eval, m.allModels.[1].mode)
        Assert.AreEqual(Mode.Eval, m.allModels.[2].mode)

        m.train()
        Assert.AreEqual(Mode.Train, m.mode)
        Assert.AreEqual(Mode.Train, m.allModels.[0].mode)
        Assert.AreEqual(Mode.Train, m.allModels.[1].mode)
        Assert.AreEqual(Mode.Train, m.allModels.[2].mode)

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

    [<Test>]
    member _.TestModelDropout () =
        let m = Dropout(1.)
        let x = dsharp.randn([10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.AreEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.AreEqual(x, xeval)

    [<Test>]
    member _.TestModelDropout2d () =
        let m = Dropout2d(1.)
        let x = dsharp.randn([10;4;10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.AreEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.AreEqual(x, xeval)

    [<Test>]
    member _.TestModelDropout3d () =
        let m = Dropout3d(1.)
        let x = dsharp.randn([10;4;10;10;10])
        
        m.train()
        let xtrain = x --> m
        Assert.AreEqual(x.zerosLike(), xtrain)
        m.eval()
        let xeval = x --> m
        Assert.AreEqual(x, xeval)


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