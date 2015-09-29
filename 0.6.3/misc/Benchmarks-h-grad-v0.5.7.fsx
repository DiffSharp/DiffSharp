//
// Benchmark for gradient calculations with AD and numerical differentiation
//
// Atilim Gunes Baydin
// atilimgunes.baydin@nuim.ie
//
// Hamilton Institute
// National University of Ireland Maynooth
// Maynooth, Co. Kildare
// Ireland
//
// www.bcl.hamilton.ie
// 

open DiffSharp.Util.LinearAlgebra

let rnd = System.Random()

let duration n f =
    let s = new System.Diagnostics.Stopwatch()
    s.Start() |> ignore
    for i in 1..n do
        f() |> ignore
    s.Stop() |> ignore
    let dur = s.ElapsedMilliseconds
    (float dur) / (float n)

let helmholtzFloat R T (b:Vector<float>) (A:Matrix<float>) (x:Vector<float>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

// Original function
let testHelmholtzOrig n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> 0.1 * rnd.NextDouble())
    let A = Matrix.init n n (fun _ _ -> 0.1 * rnd.NextDouble())
    let x = Vector.init n (fun _ -> 0.1 * rnd.NextDouble())
    helmholtzFloat R T b A x


// Numerical differentiation
open DiffSharp.Numerical.Vector

let testHelmholtzNumerical n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    let A = Matrix.init n n (fun _ _ -> (0.1 * rnd.NextDouble()))
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtzFloat R T b A) x


// Reverse mode AD
open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector

let helmholtzAdj R T (b:Vector<Adj>) (A:Matrix<Adj>) (x:Vector<Adj>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

let testHelmholtzReverseAD n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> adj (0.1 * rnd.NextDouble()))
    let A = Matrix.init n n (fun _ _ -> adj (0.1 * rnd.NextDouble()))
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtzAdj R T b A) x


// Forward mode AD
open DiffSharp.AD.ForwardG
open DiffSharp.AD.ForwardG.Vector

let helmholtzDualG R T (b:Vector<DualG>) (A:Matrix<DualG>) (x:Vector<DualG>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

let testHelmholtzForwardAD n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> dualG (0.1 * rnd.NextDouble()) n)
    let A = Matrix.init n n (fun _ _ -> dualG (0.1 * rnd.NextDouble()) n)
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtzDualG R T b A) x


// Tests
let repetitions = 1000000
let dataori = [for n in 1..7..50->n, duration repetitions (fun _->testHelmholtzOrig n)]
let datanum = [for n in 1..7..50->n, duration repetitions (fun _->testHelmholtzNumerical n)]
let datafad = [for n in 1..7..50->n, duration repetitions (fun _->testHelmholtzForwardAD n)]
let datarad = [for n in 1..7..50->n, duration repetitions (fun _->testHelmholtzReverseAD n)]
