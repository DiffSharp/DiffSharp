
namespace Tests.AD

open Xunit
open FsUnit.Xunit
open DiffSharp.Util
open DiffSharp.AD

module P =
    [<Fact>]
    let ``Add_D_D``() =
        (D 1.) + (D 2.) |> should equal (D 3.)


module T =
    [<Fact>]
    let ``Add_D_D``() =
        let i = GlobalTagger.Next
        let a = makeForward i (D 1.2) (D 1.1)
        let b = makeForward i (D 2.2) (D 2.1)
        let c = a + b
        c |> tangent |> float |> should (equalWithin 1.e-10) 3.4

module A =
    [<Fact>]
    let ``Add_D_D``() =
        let f (x:DV) = x.[0] + x.[1]
        let g = grad f (vector [D 1.1; D 1.2])
        g.[0] |> float |> should (equalWithin 1.e-10) 1.

