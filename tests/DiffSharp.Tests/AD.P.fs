
namespace Tests.AD

open Tests
open Xunit
open FsUnit.Xunit
open DiffSharp.Util
open DiffSharp.AD

module P =
    [<Fact>]
    let ``Add_D_D``() =
        (D 1.) + (D 2.) |> should equal (D 3.)
