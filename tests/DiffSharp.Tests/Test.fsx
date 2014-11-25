
#load "../../src/DiffSharp/Util.General.fs"
#load "../../src/DiffSharp/Util.LinearAlgebra.fs"

open DiffSharp.Util.LinearAlgebra

let inline vector v = Vector (Array.ofSeq v)

let a = vector [|2.; 2.; 3.|]
let b = vector [1.; 1.; 1.]
let m = Matrix.Create(3, 7, 2.)
let i:Matrix<float> = Matrix.CreateIdentity(3)

let test = a * i