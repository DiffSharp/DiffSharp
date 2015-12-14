namespace DiffSharp.Tests

type Util() =
    static let eps64 = 1e-4
    static let eps32 = float32 eps64

    static member (=~) (a:float32, b:float32) =
        if   System.Single.IsNaN(a) then
             System.Single.IsNaN(b)
        elif System.Single.IsPositiveInfinity(a) || (a = System.Single.MaxValue) then
             System.Single.IsPositiveInfinity(b) || (b = System.Single.MaxValue)
        elif System.Single.IsNegativeInfinity(a) || (a = System.Single.MinValue) then
             System.Single.IsNegativeInfinity(b) || (b = System.Single.MinValue)
        else 
             abs (a - b) < eps32

    static member (=~) (a:float, b:float) =
        if   System.Double.IsNaN(a) then
             System.Double.IsNaN(b)
        elif System.Double.IsPositiveInfinity(a) || (a = System.Double.MaxValue) then
             System.Double.IsPositiveInfinity(b) || (b = System.Double.MaxValue)
        elif System.Double.IsNegativeInfinity(a) || (a = System.Double.MinValue) then
             System.Double.IsNegativeInfinity(b) || (b = System.Double.MinValue)
        else 
             abs (a - b) < eps64

    static member (=~) (a:float32[], b:float32[]) =
        if a.Length <> b.Length then
            false
        else
            match Array.map2 (fun (x:float32) (y:float32) -> (Util.(=~)(x, y))) a b |> Array.tryFind not with
            | Some(_) -> false
            | _ -> true

    static member (=~) (a:float[], b:float[]) =
        if a.Length <> b.Length then
            false
        else
            match Array.map2 (fun (x:float) (y:float) -> (Util.(=~)(x, y))) a b |> Array.tryFind not with
            | Some(_) -> false
            | _ -> true