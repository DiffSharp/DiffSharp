module DiffSharp.Tests

open Xunit
open FsCheck
open FsCheck.Xunit
open DiffSharp.AD.Forward

let ``is valid float`` x =
    not (System.Double.IsInfinity(x) || System.Double.IsNegativeInfinity(x) || System.Double.IsPositiveInfinity(x) || System.Double.IsNaN(x) || (x = System.Double.MinValue) || (x = System.Double.MaxValue))

let (=~) x y =
    abs (x - y) < 1e-6


let f1 = fun x -> (sin x) * (cos (exp x))
let q1 =  <@ fun x -> (sin x) * (cos (exp x)) @>


[<Property(Verbose = true)>]
let ``AD.Forward diff`` (x:float) =
    (``is valid float`` x) ==> ((DiffSharp.AD.Forward.ForwardOps.diff f1 x) =~ (DiffSharp.Symbolic.SymbolicOps.diff q1 x))

