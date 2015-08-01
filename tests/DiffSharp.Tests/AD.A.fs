
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

    [<Fact>]
    let ``Item_DM``() =
        let f (x:DM) = x.[1, 1]
        let g = grad f (matrix [[D 1.1; D 1.1]; [D 1.1; D 1.1]])
        g.[0, 0] |> float |> should (equalWithin accuracy) 0.
        g.[1, 1] |> float |> should (equalWithin accuracy) 1.

    [<Fact>]
    let ``Split_DV``() =
        let f (x:DV) = 
            let s = x |> Vector.split [2; 3] |> Seq.toArray
            s.[0].[1] + s.[1].[2]
        let g = grad f (vector [D 1.; D 2.; D 3.; D 4.; D 5.])
        g.[1] |> float |> should (equalWithin accuracy) 1.
        g.[4] |> float |> should (equalWithin accuracy) 1.
        g.[0] |> float |> should (equalWithin accuracy) 0.

