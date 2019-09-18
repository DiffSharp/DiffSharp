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
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.; 5.; 5.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([5.; 5.; 5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    [<Test>]
    member this.TestDerivativeSubTT () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdy = Tensor.Create([5.; 6.; 7.]).GetForward(Tensor.Create([2.; 2.; 3.]), 1u)
        let fwdz = fwdx - fwdy
        let fwdzCorrect = Tensor.Create([-4.; -4.; -4.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([0.; 1.; 1.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revy = Tensor.Create([5.; 6.; 7.]).GetReverse(1u)
        let revz = revx - revy
        let revzCorrect = Tensor.Create([-4.; -4.; -4.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.; 5.; 5.])
        let revyd = revy.Derivative
        let revydCorrect = Tensor.Create([-5.; -5.; -5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
        Assert.AreEqual(revyd, revydCorrect)

    [<Test>]
    member this.TestDerivativeNeg () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdz = -fwdx
        let fwdzCorrect = Tensor.Create([-1.; -2.; -3.])
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create([-2.; -3.; -4.])

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revz = -revx
        let revzCorrect = Tensor.Create([-1.; -2.; -3.])
        revz.Reverse(Tensor.Create([5.; 5.; 5.]))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([-5.; -5.; -5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)

    [<Test>]
    member this.TestDerivativeSum () =
        let fwdx = Tensor.Create([1.; 2.; 3.]).GetForward(Tensor.Create([2.; 3.; 4.]), 1u)
        let fwdz = fwdx.Sum()
        let fwdzCorrect = Tensor.Create(6.)
        let fwdzd = fwdz.Derivative
        let fwdzdCorrect = Tensor.Create(9.)

        let revx = Tensor.Create([1.; 2.; 3.]).GetReverse(1u)
        let revz = revx.Sum()
        let revzCorrect = Tensor.Create(6.)
        revz.Reverse(Tensor.Create(5.))
        let revxd = revx.Derivative
        let revxdCorrect = Tensor.Create([5.; 5.; 5.])

        Assert.AreEqual(fwdz, fwdzCorrect)
        Assert.AreEqual(fwdzd, fwdzdCorrect)
        Assert.AreEqual(revz, revzCorrect)
        Assert.AreEqual(revxd, revxdCorrect)
