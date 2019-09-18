namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDerivatives () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestDerivativeAddTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx + fwdy
        let fwdzCorrect = Tensor.Create([6.; 8.; 10.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([4.; 5.; 7.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
        let revz = revx + revy
        let revzCorrect = Tensor.Create([6.; 8.; 10.])
        revz.Reverse()
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([1.;1.;1.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([1.;1.;1.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    [<Test>]
    member this.TestDerivativeSum () =
        let fwdx = Tensor.Create([1.; 2.; 3.])
        let fwdv = Tensor.Create([2.; 3.; 4.])
        let fwdfx, fwdjv = DiffSharp.jacobianv' (fun t -> t.Sum()) fwdx fwdv
        let fwdfxCorrect = Tensor.Create(6.)
        let fwdjvCorrect = Tensor.Create(9.)

        let revx = Tensor.Create([1.; 2.; 3.])
        let revv = Tensor.Create(5.)
        let revfx, revjv = DiffSharp.jacobianTv' (fun t -> t.Sum()) revx revv
        let revfxCorrect = Tensor.Create(6.)
        let revjvCorrect = Tensor.Create([5.; 5.; 5.])

        Assert.AreEqual(fwdfx, fwdfxCorrect)
        Assert.AreEqual(fwdjv, fwdjvCorrect)
        Assert.AreEqual(revfx, revfxCorrect)
        Assert.AreEqual(revjv, revjvCorrect)