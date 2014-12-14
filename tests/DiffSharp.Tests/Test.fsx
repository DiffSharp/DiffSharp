#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

open DiffSharp.Util.LinearAlgebra


let a = matrix [[3.; 9.;]; [4.; 3.]]

let test = Matrix.eigenvalues a