namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Model


type ModelStyle1a() =
    inherit Model()
    let fc1 = Linear(10, 32)
    let fc2 = Linear(32, 10)
    do base.add([fc1; fc2], ["fc1"; "fc2"])
    override __.forward(x) =
        x
        |> fc1.forward
        |> dsharp.relu
        |> fc2.forward

type ModelStyle1b() =
    inherit Model()
    let fc1 = Linear(10, 32)
    let fc2 = Linear(32, 20)
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
        Assert.AreEqual(682, net.nparameters())

        let net2 = ModelStyle1b()
        Assert.AreEqual(1013, net2.nparameters())

    [<Test>]
    member this.TestModelCreationStyle2 () =
        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let net = Model.create [fc1; fc2] 
                    (dsharp.view [-1; 10]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward)
        Assert.AreEqual(682, net.nparameters())

        let fc1 = Linear(10, 32)
        let fc2 = Linear(32, 10)
        let p = Parameter(dsharp.randn([]))
        let net2 = Model.create [fc1; fc2; p] 
                    (dsharp.view [-1; 28*28]
                    >> fc1.forward
                    >> dsharp.relu
                    >> fc2.forward
                    >> dsharp.mul p.value)
        Assert.AreEqual(683, net2.nparameters())

    [<Test>]
    member this.TestModelCompose () =
        let net1 = ModelStyle1a()
        let net2 = ModelStyle1b()
        let net3 = Model.compose net1 net2
        Assert.AreEqual(682 + 1013, net3.nparameters())

        let x = dsharp.randn([5;10])
        let y = net3.forward(x)
        Assert.AreEqual([5;20], y.shape)
