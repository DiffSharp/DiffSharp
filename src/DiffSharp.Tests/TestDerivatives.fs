namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDerivatives () =

    [<SetUp>]
    member this.Setup () =
        ()

    // [<Test>]
    // member this.TestDerivativeAddTT () =
    //     let fwdx1 = Tensor.Create([1.f; 2.f; 3.f])
    //     let fwdv1 = Tensor.Create([2.f; 3.f; 4.f])
    //     let fwdx2 = Tensor.Create([5.f; 6.f; 7.f])
    //     let fwdv2 = Tensor.Create([2.f; 2.f; 3.f])
    //     let fwdx1 = fwdx1.GetForward()
    //     let fwdfx, fwdjv = DiffSharp.jacobianv' (fun t -> t.Sum()) fwdx fwdv
    //     let fwdfxCorrect = Tensor.Create(6.f)
    //     let fwdjvCorrect = Tensor.Create(9.f)

    //     let revx = Tensor.Create([1.f; 2.f; 3.f])
    //     let revv = Tensor.Create(5.f)
    //     let revfx, revjv = DiffSharp.jacobianTv' (fun t -> t.Sum()) revx revv
    //     let revfxCorrect = Tensor.Create(6.f)
    //     let revjvCorrect = Tensor.Create([5.f; 5.f; 5.f])

    //     Assert.AreEqual(fwdfx, fwdfxCorrect)
    //     Assert.AreEqual(fwdjv, fwdjvCorrect)
    //     Assert.AreEqual(revfx, revfxCorrect)
    //     Assert.AreEqual(revjv, revjvCorrect)

    [<Test>]
    member this.TestDerivativeSum () =
        let fwdx = Tensor.Create([1.f; 2.f; 3.f])
        let fwdv = Tensor.Create([2.f; 3.f; 4.f])
        let fwdfx, fwdjv = DiffSharp.jacobianv' (fun t -> t.Sum()) fwdx fwdv
        let fwdfxCorrect = Tensor.Create(6.f)
        let fwdjvCorrect = Tensor.Create(9.f)

        let revx = Tensor.Create([1.f; 2.f; 3.f])
        let revv = Tensor.Create(5.f)
        let revfx, revjv = DiffSharp.jacobianTv' (fun t -> t.Sum()) revx revv
        let revfxCorrect = Tensor.Create(6.f)
        let revjvCorrect = Tensor.Create([5.f; 5.f; 5.f])

        Assert.AreEqual(fwdfx, fwdfxCorrect)
        Assert.AreEqual(fwdjv, fwdjvCorrect)
        Assert.AreEqual(revfx, revfxCorrect)
        Assert.AreEqual(revjv, revjvCorrect)