
namespace Tests.AD

open Tests
open Xunit
open FsUnit.Xunit
open DiffSharp.Util
open DiffSharp.AD

module Nesting =
    [<Fact>]
    let ``Free_var``() =
        let res = diff (fun x -> diff (fun y -> x * y) (D 2.)) (D 3.)
        res |> float |> should (equalWithin accuracy) 1.