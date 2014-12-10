(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

(**
Helmholtz Energy Function
=========================

We start by defining our neural network structure.

*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra


let helmholtz R T (b:Vector<Adj>) (A:Matrix<Adj>) (x:Vector<Adj>) =
    //Vector.su


//let testHelmholtz n =
//    let rnd = System.Random()
//
//    let r = 0.
//    let t = 0.
//    let b = Vector.init 
