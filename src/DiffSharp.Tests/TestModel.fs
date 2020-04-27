namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Model

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

