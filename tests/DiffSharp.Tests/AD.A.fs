
namespace Tests.AD

open Tests
open Xunit
open FsUnit.Xunit
open DiffSharp.Util
open DiffSharp.AD


module A =
    [<Fact>]
    let ``Add_D_D``() =
        let f (x:DV) = x.[0] + x.[1]
        let g = grad f (vector [D 1.1; D 1.2])
        g.[0] |> float |> should (equalWithin accuracy) 1.
        g.[1] |> float |> should (equalWithin accuracy) 1.
    
    [<Fact>]
    let ``Item_DV``() =
        let f (x:DV) = x.[1]
        let g = grad f (vector [D 1.1; D 1.1; D 1.1])
        g.[0] |> float |> should (equalWithin accuracy) 0.
        g.[1] |> float |> should (equalWithin accuracy) 1.
        g.[2] |> float |> should (equalWithin accuracy) 0.

//    [<Fact>]
//    let ``Item_DM``() =
//        let f ()