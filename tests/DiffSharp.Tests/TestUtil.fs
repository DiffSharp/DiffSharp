namespace Tests

open System
open NUnit.Framework

[<AutoOpen>]
module TestUtils =
    let isException f = Assert.Throws<Exception>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isInvalidOp f = Assert.Throws<InvalidOperationException>(TestDelegate(fun () -> f() |> ignore)) |> ignore
    let isAnyException f = Assert.Catch(TestDelegate(fun () -> f() |> ignore)) |> ignore

    type Assert with 

        /// Like Assert.AreEqual bute requires theat the actual and expected are the same type
        static member CheckEqual (expected: 'T, actual: 'T) = Assert.AreEqual(box expected, box actual)
