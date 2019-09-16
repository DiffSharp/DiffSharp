namespace Tests

open NUnit.Framework
open DiffSharp.Util

[<TestFixture>]
type TestUtil () =

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestArraysEqual () =
        let array1 = [|1;2;3|]
        let array2 = [|1;2;3|]
        let array3 = [|1;2;4|]
        let array4 = [|1;2;3;4|]
        Assert.True(arraysEqual array1 array2)
        Assert.False(arraysEqual array1 array3)
        Assert.False(arraysEqual array1 array4)
