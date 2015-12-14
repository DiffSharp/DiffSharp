module DiffSharp.Tests.Util

let eps64 = 1e-4
let eps32 = float32 eps64

let inline (=~) a b =
    match box a with
    | :? float as a ->
        match box b with
        | :? float as b ->
            if   System.Double.IsNaN(a) then
                 System.Double.IsNaN(b)
            elif System.Double.IsPositiveInfinity(a) || (a = System.Double.MaxValue) then
                 System.Double.IsPositiveInfinity(b) || (b = System.Double.MaxValue)
            elif System.Double.IsNegativeInfinity(a) || (a = System.Double.MinValue) then
                 System.Double.IsNegativeInfinity(b) || (b = System.Double.MinValue)
            else 
                 abs (a - b) < eps64
        | _ -> false
    | :? float32 as a ->
        match box b with
        | :? float32 as b ->
            if  System.Single.IsNaN(a) then 
                System.Single.IsNaN(b)
            elif System.Single.IsPositiveInfinity(a) || (a = System.Single.MaxValue) then
                 System.Single.IsPositiveInfinity(b) || (b = System.Single.MaxValue)
            elif System.Single.IsNegativeInfinity(a) || (a = System.Single.MinValue) then
                 System.Single.IsNegativeInfinity(b) || (b = System.Single.MinValue)
            else 
                abs (a - b) < eps32
        | _ -> false
    | _ -> false
