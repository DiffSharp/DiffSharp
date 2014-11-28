#load "../../src/DiffSharp/Util.General.fs"
#load "../../src/DiffSharp/Util.LinearAlgebra.fs"

open DiffSharp.Util.LinearAlgebra

let inline vector v = Vector (Array.ofSeq v)

let a = matrix [[2.; -1.; -2.]; [-4.; 6.; 3.]; [-4.; -2.; -8.]]
let b = matrix [[2.4; -3.]; [1.; 2.]]
let c = matrix [[2.4; -3.; 5.4]; [1.; 2.; -2.3]; [0.3; 1.5; -7.1]]
let d = matrix [[2.4; -3.; 5.4; 7.]; [1.; 2.; -2.3; 7.]; [0.3; 1.5; -7.1; 7.]; [7.; 7.; 7.; 7.]]


//let test:Matrix<float> = Matrix.CreateIdentity(3)

let alu, ap = a.GetLUDecomposition()
let blu, bp = b.GetLUDecomposition()
let clu, cp = c.GetLUDecomposition()
let dlu, dp = d.GetLUDecomposition()

let ai = a.GetInverse()
let bi = b.GetInverse()
let ci = c.GetInverse()
let di = d.GetInverse()

let ad = a.GetDeterminant()
let bd = b.GetDeterminant()
let cd = c.GetDeterminant()
let dd = d.GetDeterminant()