module AD

open Xunit
open FsUnit.Xunit
open DiffSharp.AD

[<Fact>]
let ``AD.Fwd D Add``() =
    (D 1.) + (D 2.) |> should equal (D 3.)


