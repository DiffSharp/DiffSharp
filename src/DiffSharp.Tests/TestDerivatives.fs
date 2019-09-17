namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDerivatives () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestDerivativeSum () =
        let x = Tensor.Create([[1.f; 2.f]; [3.f; 4.f]])
        let z, z' = DiffSharp.grad' (fun t -> t.Sum()) x
        let zCorrect = Tensor.Create(10.f)
        let z'Correct = Tensor.Create([[1.f; 1.f]; [1.f; 1.f]])

        Assert.AreEqual(z, zCorrect)
        Assert.AreEqual(z', z'Correct)