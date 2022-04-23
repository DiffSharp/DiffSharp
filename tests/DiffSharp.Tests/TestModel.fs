// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Compose
open DiffSharp.Model
open DiffSharp.Data
open DiffSharp.Optim

type ModelStyle1() =
    inherit Model() 
    let fc1:Model = Linear(10, 16)
    let fc2:Model = Linear(16, 20)
    do base.addModel((fc1, "fc1"), (fc2, "fc2"))
    override __.forward(x) =
        x
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward

type ModelStyle1WithParamBuffer() =
    inherit Model()
    let fc1:Model = Linear(20, 32)
    let fc2:Model = Linear(32, 30)
    let p = Parameter(dsharp.randn([]))
    let b = Parameter(dsharp.randn([]))
    do base.addModel((fc1, "fc1"), (fc2, "fc2"))
    do base.addParameter((p, "p"))
    do base.addBuffer((b, "b"))
    override __.forward(x) =
        b.value <- x.min()
        x
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward
        |> dsharp.mul p.value

type GenericModelFloatFloat() =
    inherit Model<float,float>()
    let fc1:Model = Linear(1, 2)
    let fc2:Model = Linear(2, 1)
    do base.addModel((fc1, "fc1"), (fc2, "fc2"))
    do base.init (fun (_, t) -> t.onesLike())
    override __.forward(x) =
        x |> dsharp.tensor
        |> dsharp.view([1; -1])
        |> fc1.forward
        |> fc2.forward
        |> float

type GenericModelIntString() =
    inherit Model<int,string>()
    let fc1:Model = Linear(1, 2)
    let fc2:Model = Linear(2, 1)
    do base.addModel((fc1, "fc1"), (fc2, "fc2"))
    do base.init (fun (_, t) -> t.onesLike())
    override __.forward(x) =
        x |> float32 |> dsharp.tensor
        |> dsharp.view([1; -1])
        |> fc1.forward
        |> fc2.forward
        |> int
        |> string


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
    member _.TestParameterDictFlattenForwardDiff () =
        let t1p = dsharp.randn([2;5])
        let t1d = dsharp.randnLike t1p
        let t2p = dsharp.randn([4])
        let t2d = dsharp.randnLike t2p

        let p1 = Parameter <| t1p.forwardDiff(t1d)
        let p2 = Parameter <| t2p.forwardDiff(t2d)

        let d = ParameterDict()
        d.add(["p1", p1; "p2", p2])

        let dflat = d.flatten(differentiable=true)
        Assert.True(dflat.isForwardDiff)

        let dflatp = dflat.primal
        let dflatd = dflat.derivative
        let dflatpCorrect = dsharp.cat([t1p.view(-1); t2p])
        let dflatdCorrect = dsharp.cat([t1d.view(-1); t2d])
        
        Assert.CheckEqual(dflatpCorrect, dflatp)
        Assert.CheckEqual(dflatdCorrect, dflatd)

    [<Test>]
    member _.TestParameterDictUnflattenForwardDiff () =
        let t1p = dsharp.randn([2;5])
        let t2p = dsharp.randn([4])

        let p1 = Parameter <| t1p
        let p2 = Parameter <| t2p

        let d = ParameterDict()
        d.add(["p1", p1; "p2", p2])

        let dflatp = dsharp.randn([14])
        let dflatd = dsharp.randn([14])
        let dflat = dflatp.forwardDiff(dflatd)

        Assert.False(d["p1"].isForwardDiff)
        Assert.False(d["p2"].isForwardDiff)
        d.unflatten(dflat, differentiable=true)
        Assert.True(d["p1"].isForwardDiff)
        Assert.True(d["p2"].isForwardDiff)

        let dp1 = d["p1"]
        let dp1p = dp1.primal
        let dp1d = dp1.derivative
        let dp2 = d["p2"]
        let dp2p = dp2.primal
        let dp2d = dp2.derivative

        let dfp = dflatp.split([2*5; 4])
        let dp1pCorrect, dp2pCorrect = dfp[0].view([2;5]), dfp[1]
        let dfd = dflatd.split([2*5; 4])
        let dp1dCorrect, dp2dCorrect = dfd[0].view([2;5]), dfd[1]

        Assert.CheckEqual(dp1pCorrect, dp1p)
        Assert.CheckEqual(dp2pCorrect, dp2p)
        Assert.CheckEqual(dp1dCorrect, dp1d)
        Assert.CheckEqual(dp2dCorrect, dp2d)

    [<Test>]
    member _.TestParameterDictFlattenReverseDiff () =
        let t1p = dsharp.randn([2;5])
        let t1d = dsharp.randnLike t1p
        let t2p = dsharp.randn([4])
        let t2d = dsharp.randnLike t2p

        let p1 = Parameter <| t1p.reverseDiff(t1d)
        let p2 = Parameter <| t2p.reverseDiff(t2d)

        let d = ParameterDict()
        d.add(["p1", p1; "p2", p2])

        let dflat = d.flatten(differentiable=true)
        Assert.True(dflat.isReverseDiff)

        let dflatp = dflat.primal
        let dflatd = dflat.derivative
        let dflatpCorrect = dsharp.cat([t1p.view(-1); t2p])
        let dflatdCorrect = dsharp.cat([t1d.view(-1); t2d])
        
        Assert.CheckEqual(dflatpCorrect, dflatp)
        Assert.CheckEqual(dflatdCorrect, dflatd)

    [<Test>]
    member _.TestParameterDictUnflattenReverseDiff () =
        let t1p = dsharp.randn([2;5])
        let t2p = dsharp.randn([4])

        let p1 = Parameter <| t1p
        let p2 = Parameter <| t2p

        let d = ParameterDict()
        d.add(["p1", p1; "p2", p2])

        let dflatp = dsharp.randn([14])
        let dflatd = dsharp.randn([14])
        let dflat = dflatp.reverseDiff(dflatd)

        Assert.False(d["p1"].isReverseDiff)
        Assert.False(d["p2"].isReverseDiff)
        d.unflatten(dflat, differentiable=true)
        Assert.True(d["p1"].isReverseDiff)
        Assert.True(d["p2"].isReverseDiff)

        let dp1 = d["p1"]
        let dp1p = dp1.primal
        let dp1d = dp1.derivative
        let dp2 = d["p2"]
        let dp2p = dp2.primal
        let dp2d = dp2.derivative

        let dfp = dflatp.split([2*5; 4])
        let dp1pCorrect, dp2pCorrect = dfp[0].view([2;5]), dfp[1]
        let dfd = dflatd.split([2*5; 4])
        let dp1dCorrect, dp2dCorrect = dfd[0].view([2;5]), dfd[1]

        Assert.CheckEqual(dp1pCorrect, dp1p)
        Assert.CheckEqual(dp2pCorrect, dp2p)
        Assert.CheckEqual(dp1dCorrect, dp1d)
        Assert.CheckEqual(dp2dCorrect, dp2d)

    [<Test>]
    member _.TestModelCreationStyle1 () =
        let batchSize = 2

        let net = ModelStyle1()
        let x = dsharp.randn([batchSize; 10])
        let y = net.forward(x)
        Assert.CheckEqual(516, net.nparameters)
        Assert.AreEqual([|batchSize; 20|], y.shape)

        let net2 = ModelStyle1WithParamBuffer()
        let x2 = dsharp.randn([batchSize; 20])
        let y2 = net2.forward(x2)
        Assert.CheckEqual(1663, net2.nparameters)
        Assert.AreEqual([|batchSize; 30|], y2.shape)

    [<Test>]
    member _.TestModelCreationStyle2 () =
        let batchSize = 2

        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let net = Model(dsharp.view [-1; 10]
                        >> fc1.forward
                        >> dsharp.relu
                        >> fc2.forward, 
                        models=[fc1; fc2])
        let x = dsharp.randn([batchSize; 10])
        let y = net.forward(x)
        Assert.CheckEqual(682, net.nparameters)
        Assert.AreEqual([|batchSize; 10|], y.shape)
        
        // check these properties exist
        fc1.weight |> ignore
        fc1.bias |> ignore

        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let p = Parameter(dsharp.randn([]))
        let net2 = Model(dsharp.view [-1; 10]
                        >> fc1.forward
                        >> dsharp.relu
                        >> fc2.forward
                        >> dsharp.mul p.value, 
                        parameters=[p], 
                        models=[fc1; fc2])
        let x2 = dsharp.randn([batchSize; 10])
        let y2 = net2.forward(x2)
        Assert.CheckEqual(683, net2.nparameters)
        Assert.AreEqual([|batchSize; 10|], y2.shape)

    [<Test>]
    member _.TestModelCreationStyle3 () =
        let batchSize = 2
        let net = dsharp.view [-1; 10] --> Linear(10, 32) --> dsharp.relu --> Linear(32, 10)
        let x = dsharp.randn([batchSize; 10])
        let y = net.forward(x)
        Assert.CheckEqual(682, net.nparameters)        
        Assert.AreEqual([|batchSize; 10|], y.shape)

    [<Test>]
    member _.TestModelCreationStyle4 () =
        let batchSize = 2
        let net =
            Model(dsharp.view [-1; 10])
            --> Linear(10, 32)
            --> Model(dsharp.relu)
            --> Linear(32, 10)
        let x = dsharp.randn([batchSize; 10])
        let y = net.forward(x)
        Assert.CheckEqual(682, net.nparameters)
        Assert.AreEqual([|batchSize; 10|], y.shape)

    [<Test>]
    member _.TestModelUsageStyle1 () =
        let net = ModelStyle1()
        let x = dsharp.randn([1; 10])
        let y = net.forward x |> dsharp.sin
        Assert.CheckEqual([| 1;20 |], y.shape)

    [<Test>]
    member _.TestModelUsageStyle2 () =
        let net = ModelStyle1()
        let x = dsharp.randn([1; 10])
        let y = x --> net --> dsharp.sin
        Assert.CheckEqual([| 1;20 |], y.shape)

    [<Test>]
    member _.TestModelInit () =
        let net = Linear(10, 10)
        let wBefore = net.parameters["Linear-weight"]
        net.init(function
            | "Linear-weight", v -> v.onesLike()
            | _, v -> v)
        let wAfter = net.parameters["Linear-weight"]
        let wAfterCorrect = dsharp.onesLike(wBefore)
        Assert.False(wAfterCorrect.allclose(wBefore))
        Assert.True(wAfterCorrect.allclose(wAfter))

    [<Test>]
    member _.TestModelCompose () =
        let net1 = ModelStyle1()
        let net2 = ModelStyle1WithParamBuffer()
        let net3 = Model.compose net1 net2
        Assert.CheckEqual(516 + 1663, net3.nparameters)

        let x = dsharp.randn([5;10])
        let y = net3.forward(x)
        Assert.CheckEqual([|5;30|], y.shape)

    [<Test>]
    member _.TestModelParametersDiff () =
        let net = ModelStyle1()

        Assert.True(net.parametersVector.isNoDiff)

        let p = net.parametersVector
        let p = p.forwardDiff(p.onesLike())
        net.parametersVector <- p
        Assert.True(net.parametersVector.isForwardDiff)

        net.noDiff()
        Assert.True(net.parametersVector.isNoDiff)

        let p = net.parametersVector
        let p = p.reverseDiff()
        net.parametersVector <- p
        Assert.True(net.parametersVector.isReverseDiff)

        net.noDiff()
        Assert.True(net.parametersVector.isNoDiff)

        let p = net.parametersVector
        let x = dsharp.randn([1;10])
        ignore <| dsharp.grad (fun p -> net.asFunction p x |> dsharp.sum) p
        Assert.True(net.parametersVector.isNoDiff)

    [<Test>]
    member _.TestModelDiff () =
        let net = ModelStyle1()

        let status = net.isNoDiff, net.isForwardDiff, net.isReverseDiff
        let statusCorrect = true, false, false
        Assert.AreEqual(statusCorrect, status)

        net.forwardDiff(net.parameters)
        let status = net.isNoDiff, net.isForwardDiff, net.isReverseDiff
        let statusCorrect = false, true, false
        Assert.AreEqual(statusCorrect, status)

        net.noDiff()
        let status = net.isNoDiff, net.isForwardDiff, net.isReverseDiff
        let statusCorrect = true, false, false
        Assert.AreEqual(statusCorrect, status)

        net.reverseDiff()
        let status = net.isNoDiff, net.isForwardDiff, net.isReverseDiff
        let statusCorrect = false, false, true
        Assert.AreEqual(statusCorrect, status)

        net.noDiff()
        let status = net.isNoDiff, net.isForwardDiff, net.isReverseDiff
        let statusCorrect = true, false, false
        Assert.AreEqual(statusCorrect, status)

    [<Test>]
    member _.TestModelForwardParameters () =
        let net = ModelStyle1()
        let f = net.asFunction
        let p = net.parametersVector
        let x = dsharp.randn([1;10])
        let y = f p x
        Assert.CheckEqual([|1;20|], y.shape)

    [<Test>]
    member _.TestModelForwardCompose () =
        let net = ModelStyle1()
        let f p x = net.asFunction p x |> dsharp.sin
        let p = net.parametersVector
        let x = dsharp.randn([1;10])
        let y = f p x
        Assert.CheckEqual([|1;20|], y.shape)

    [<Test>]
    member _.TestModelForwardLoss () =
        let net = ModelStyle1()
        let f p x t = net.asFunction p x |> dsharp.mseLoss t
        let p = net.parametersVector
        let x = dsharp.randn([1;10])
        let t = dsharp.randn([1;20])
        let y = f p x t
        Assert.CheckEqual(([| |]: int array), y.shape)

    [<Test>]
    member _.TestModelSaveLoadState () =
        let net1 = ModelStyle1WithParamBuffer()
        let p1 = net1.stateVector
        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net1.state, fileName)

        let net2 = ModelStyle1WithParamBuffer()
        let p2 = net2.stateVector
        Assert.AreNotEqual(p1, p2)

        net2.state <- dsharp.load(fileName)
        let p2 = net2.stateVector
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;20])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelSaveLoad () =
        let net1 = ModelStyle1WithParamBuffer()
        let p1 = net1.stateVector
        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net1, fileName)

        let net2:Model = dsharp.load(fileName)
        let p2 = net2.stateVector
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;20])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelSaveLoadStateDiff () =
        let net1 = ModelStyle1WithParamBuffer()
        // net1 is not differentiable
        net1.reverseDiff()
        // net1 is reverse-mode differentiable
        let b1 = net1.buffersVector
        let p1 = net1.parametersVector
        let s1 = net1.stateVector
        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net1.state, fileName)
        Assert.True(net1.isReverseDiff)
        Assert.True(b1.isNoDiff)
        Assert.True(p1.isReverseDiff)
        Assert.True(s1.isNoDiff)

        let net2 = ModelStyle1WithParamBuffer()
        // net2 is not differentiable
        let b2 = net2.buffersVector
        let p2 = net2.parametersVector
        let s2 = net2.stateVector
        Assert.AreNotEqual(b1, b2)
        Assert.AreNotEqual(p1, p2)
        Assert.AreNotEqual(s1, s2)
        Assert.True(net2.isNoDiff)
        Assert.True(b2.isNoDiff)
        Assert.True(p2.isNoDiff)
        Assert.True(s2.isNoDiff)

        net2.state <- dsharp.load(fileName)
        // net2 is still not differentiable
        // Setting a previously differentiable state does not set and derivatives
        // This is compatible with PyTorch where saving and loading a state does not preserve gradient information
        let b2 = net2.buffersVector
        let p2 = net2.parametersVector
        let s2 = net2.stateVector
        Assert.CheckEqual(b1, b2)
        Assert.CheckEqual(p1, p2)
        Assert.CheckEqual(s1, s2)
        Assert.True(net2.isNoDiff)
        Assert.True(b2.isNoDiff)
        Assert.True(p2.isNoDiff)
        Assert.True(s2.isNoDiff)

        let x = dsharp.randn([1;20])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelSaveLoadDiff () =
        let net1 = ModelStyle1WithParamBuffer()
        // net1 is not differentiable
        net1.reverseDiff()
        // net1 is reverse-mode differentiable
        let b1 = net1.buffersVector
        let p1 = net1.parametersVector
        let s1 = net1.stateVector
        let fileName = System.IO.Path.GetTempFileName()
        dsharp.save(net1, fileName)
        Assert.True(net1.isReverseDiff)
        Assert.True(b1.isNoDiff)
        Assert.True(p1.isReverseDiff)
        Assert.True(s1.isNoDiff)

        let net2:Model = dsharp.load(fileName)
        // net2 is reverse-mode differentiable
        // This is because the entire network was saved and loaded back
        let b2 = net2.buffersVector
        let p2 = net2.parametersVector
        let s2 = net2.stateVector
        Assert.CheckEqual(b1, b2)
        Assert.CheckEqual(p1, p2)
        Assert.CheckEqual(s1, s2)
        Assert.True(net2.isReverseDiff)
        Assert.True(b2.isNoDiff)
        Assert.True(p2.isReverseDiff)
        Assert.True(s2.isNoDiff)

        let x = dsharp.randn([1;20])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelMove () =
        for combo1 in Combos.FloatingPointExcept16s do
            use _holder = dsharp.useConfig(combo1.dtype, combo1.device, combo1.backend)
            let net = dsharp.view [-1; 2] --> Linear(2, 4) --> dsharp.relu --> Linear(4, 1)

            Assert.CheckEqual(combo1.device, net.parametersVector.device)
            Assert.CheckEqual(combo1.dtype, net.parametersVector.dtype)
            Assert.CheckEqual(combo1.backend, net.parametersVector.backend)

            Assert.CheckEqual(combo1.device, net.device)
            Assert.CheckEqual(combo1.dtype, net.dtype)
            Assert.CheckEqual(combo1.backend, net.backend)

            let x = combo1.randn([5; 20])
            let y = x --> net
            Assert.CheckEqual(combo1.device, y.device)
            Assert.CheckEqual(combo1.dtype, y.dtype)
            Assert.CheckEqual(combo1.backend, y.backend)

            for combo2 in Combos.FloatingPointExcept16s do
                net.move(combo2.device, combo2.dtype, combo2.backend)

                Assert.CheckEqual(combo2.device, net.parametersVector.device)
                Assert.CheckEqual(combo2.dtype, net.parametersVector.dtype)
                Assert.CheckEqual(combo2.backend, net.parametersVector.backend)

                Assert.CheckEqual(combo2.device, net.device)
                Assert.CheckEqual(combo2.dtype, net.dtype)
                Assert.CheckEqual(combo2.backend, net.backend)

                let x = combo2.randn([5; 20])
                let y = x --> net
                Assert.CheckEqual(combo2.device, y.device)
                Assert.CheckEqual(combo2.dtype, y.dtype)
                Assert.CheckEqual(combo2.backend, y.backend)

    [<Test>]
    member _.TestModelMoveWithParamBuffer () =
        for combo1 in Combos.FloatingPointExcept16s do
            use _holder = dsharp.useConfig(combo1.dtype, combo1.device, combo1.backend)
            let net = ModelStyle1WithParamBuffer()

            Assert.CheckEqual(combo1.device, net.parametersVector.device)
            Assert.CheckEqual(combo1.dtype, net.parametersVector.dtype)
            Assert.CheckEqual(combo1.backend, net.parametersVector.backend)

            Assert.CheckEqual(combo1.device, net.device)
            Assert.CheckEqual(combo1.dtype, net.dtype)
            Assert.CheckEqual(combo1.backend, net.backend)

            let x = combo1.randn([5; 20])
            let y = x --> net
            Assert.CheckEqual(combo1.device, y.device)
            Assert.CheckEqual(combo1.dtype, y.dtype)
            Assert.CheckEqual(combo1.backend, y.backend)

            for combo2 in Combos.FloatingPointExcept16s do
                net.move(combo2.device, combo2.dtype, combo2.backend)

                Assert.CheckEqual(combo2.device, net.parametersVector.device)
                Assert.CheckEqual(combo2.dtype, net.parametersVector.dtype)
                Assert.CheckEqual(combo2.backend, net.parametersVector.backend)

                Assert.CheckEqual(combo2.device, net.device)
                Assert.CheckEqual(combo2.dtype, net.dtype)
                Assert.CheckEqual(combo2.backend, net.backend)

                let x = combo2.randn([5; 20])
                let y = x --> net
                Assert.CheckEqual(combo2.device, y.device)
                Assert.CheckEqual(combo2.dtype, y.dtype)
                Assert.CheckEqual(combo2.backend, y.backend)

    [<Test>]
    member _.TestModelClone () =
        let net1 = ModelStyle1()
        let p1 = net1.stateVector

        let net2 = net1.clone()
        let p2 = net2.stateVector
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;10])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelCloneWithParamBuffer () =
        let net1 = ModelStyle1WithParamBuffer()
        let p1 = net1.stateVector

        let net2 = net1.clone()
        let p2 = net2.stateVector
        Assert.CheckEqual(p1, p2)

        let x = dsharp.randn([1;20])
        let y1 = x --> net1
        let y2 = x --> net2
        Assert.CheckEqual(y1, y2)

    [<Test>]
    member _.TestModelGeneric () =
        let g1 = GenericModelFloatFloat()
        let x1 = 1.
        let y1 = x1 --> g1
        let y1Correct = 5.
        Assert.AreEqual(y1Correct, y1, 1e-6)

        let g2 = GenericModelIntString()
        let x2 = 1
        let y2 = x2 --> g2
        let y2Correct = "5"
        Assert.AreEqual(y2Correct, y2)

    [<Test>]
    member _.TestModelTrainEval () =
        let m = Linear(1, 2) --> Linear(2, 3) --> Linear(3, 4)
        Assert.CheckEqual(Mode.Train, m.mode)
        Assert.CheckEqual(Mode.Train, m.descendants[0].mode)
        Assert.CheckEqual(Mode.Train, m.descendants[1].mode)
        Assert.CheckEqual(Mode.Train, m.descendants[2].mode)

        m.eval()
        Assert.CheckEqual(Mode.Eval, m.mode)
        Assert.CheckEqual(Mode.Eval, m.descendants[0].mode)
        Assert.CheckEqual(Mode.Eval, m.descendants[1].mode)
        Assert.CheckEqual(Mode.Eval, m.descendants[2].mode)

        m.train()
        Assert.CheckEqual(Mode.Train, m.mode)
        Assert.CheckEqual(Mode.Train, m.descendants[0].mode)
        Assert.CheckEqual(Mode.Train, m.descendants[1].mode)
        Assert.CheckEqual(Mode.Train, m.descendants[2].mode)

    [<Test>]
    member _.TestModelChildrenModels () =
        let m0 = Linear(2, 2)
        let m1 = Sequential([m0])
        let m2 = Sequential([m1])

        let m0children = m0.children
        let m1children = m1.children
        let m2children = m2.children

        let m0childrenCorrect:list<ModelBase> = []
        let m1childrenCorrect:list<ModelBase> = [m0]
        let m2childrenCorrect:list<ModelBase> = [m1]

        Assert.CheckEqual(m0childrenCorrect, m0children)
        Assert.CheckEqual(m1childrenCorrect, m1children)
        Assert.CheckEqual(m2childrenCorrect, m2children)

        let m0models = m0.descendants
        let m1models = m1.descendants
        let m2models = m2.descendants

        let m0modelsCorrect:list<ModelBase> = [m0]
        let m1modelsCorrect:list<ModelBase> = [m1;m0]
        let m2modelsCorrect:list<ModelBase> = [m2;m1;m0]

        Assert.CheckEqual(m0modelsCorrect, m0models)
        Assert.CheckEqual(m1modelsCorrect, m1models)
        Assert.CheckEqual(m2modelsCorrect, m2models)

    [<Test>]
    member _.TestModelChildrenParameters () =
        let l1 = Linear(1, 2)
        let l2 = Linear(2, 3)
        let l3 = Linear(3, 4)

        // ModelBase
        // |-ModelBase
        //   |-ModelBase
        //     |-ModelBase
        //       |- l1
        //     |-l2
        // |-l3
        let m1 = l1 --> dsharp.relu --> l2 --> dsharp.relu --> l3 --> dsharp.flatten(1)

        // ModelBase
        // |-ModelBase
        //   |-l1
        //   |-l2
        // |-l3 
        let m2 = l1 --> l2 --> l3

        // ModelBase
        // |-l1
        // |-l2
        // |-l3
        let m3 = Sequential([l1; l2; l3])

        let childrenParams (m:Model) = 
            m.children |> List.map (fun c -> c.nparameters) |> List.sum

        let m1Params = m1.nparameters
        let m2Params = m2.nparameters
        let m3Params = m3.nparameters
        let m1ChildrenParams = childrenParams m1
        let m2ChildrenParams = childrenParams m2
        let m3ChildrenParams = childrenParams m3

        Assert.CheckEqual(m1Params, m1ChildrenParams)
        Assert.CheckEqual(m2Params, m2ChildrenParams)
        Assert.CheckEqual(m3Params, m3ChildrenParams)

    [<Test>]
    member _.TestModelParameterNames () =
        let lin1 = Linear(10, 10)
        let lin1Names = lin1.parameters |> Seq.map fst |> Seq.toArray
        let lin1NamesCorrect = [|"Linear-weight"; "Linear-bias"|]

        let lin2 = lin1 --> lin1
        let lin2Names = lin2.parameters |> Seq.map fst |> Seq.toArray
        let lin2NamesCorrect = [|"Linear-weight__1"; "Linear-bias__1"; "Linear-weight__2"; "Linear-bias__2"|]

        let lin3 = lin1 --> lin1 --> lin1
        let lin3Names = lin3.parameters |> Seq.map fst |> Seq.toArray
        let lin3NamesCorrect = [|"Linear-weight__1"; "Linear-bias__1"; "Linear-weight__2"; "Linear-bias__2"; "Linear-weight__3"; "Linear-bias__3"|]

        Assert.AreEqual(lin1NamesCorrect, lin1Names)
        Assert.AreEqual(lin2NamesCorrect, lin2Names)
        Assert.AreEqual(lin3NamesCorrect, lin3Names)

    [<Test>]
    member _.TestModelFunc () =
        let f (x:Tensor) = x + 3
        let m = Model(f)
        let x = dsharp.tensor([1,2,3], dtype=Dtype.Int32)
        let fx = x |> f
        let mx = x --> m
        Assert.AreEqual(fx, mx)
