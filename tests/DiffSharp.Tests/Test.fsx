#load "../../src/DiffSharp/Util.General.fs"
#load "../../src/DiffSharp/Util.LinearAlgebra.fs"

open DiffSharp.Util.LinearAlgebra

let inline vector v = Vector (Array.ofSeq v)

let a = matrix [[2.; -1.; -2.]; [-4.; 6.; 3.]; [-4.; -2.; -8.]]
let b = matrix [[2.4; -3.]; [1.; 2.]]
let c = matrix [[2.4; -3.; 5.4]; [1.; 2.; -2.3]; [0.3; 1.5; -7.1]]
let d = matrix [[2.4; -3.; 5.4; 7.]; [1.; 2.; -2.3; 7.]; [0.3; 1.5; -7.1; 7.]; [7.; 7.; 7.; 7.]]
let e = matrix [[0.74099; 0.595746; 0.659522]; [0.57913; 0.725761; 0.329409]; [0.289226; 0.733011; 0.674375]]

//let test:Matrix<float> = Matrix.CreateIdentity(3)

let alu, ap, at = a.GetLUDecomposition()
let blu, bp, bt = b.GetLUDecomposition()
let clu, cp, ct = c.GetLUDecomposition()
let dlu, dp, dt = d.GetLUDecomposition()
let elu, ep, et = e.GetLUDecomposition()

let ai = a.GetInverse()
let bi = b.GetInverse()
let ci = c.GetInverse()
let di = d.GetInverse()
let ei = e.GetInverse()

let ad = a.GetDeterminant()
let bd = b.GetDeterminant()
let cd = c.GetDeterminant()
let dd = d.GetDeterminant()
let ed = e.GetDeterminant()

let aa = matrix [[3.; 7.; 2.; 5.]; [1.; 8.; 4.; 2.]; [2.; 1.; 9.; 3.]; [5.; 4.; 7.; 1.]]
let bb = vector [49.; 30.; 43.; 52.]

let xx = Matrix.Solve(aa, bb)