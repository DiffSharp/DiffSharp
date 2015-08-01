
namespace Tests.AD

open Tests
open Xunit
open FsUnit.Xunit
open DiffSharp.Util
open DiffSharp.AD


module T =
    [<Fact>]
    let ``Add_D_D``() =
        let i = GlobalTagger.Next
        let a = makeForward i (D 1.2) (D 1.1)
        let b = makeForward i (D 2.2) (D 2.1)
        let c = a + b
        c |> tangent |> float |> should (equalWithin accuracy) 3.4