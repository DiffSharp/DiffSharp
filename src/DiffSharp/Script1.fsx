#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx"


open DiffSharp.AD.Float64

let m = toDM [[1.; 2.]; [3.; 4.]]
let m2 = toDV [0.1; 0.2]

let m3 = m + (DM.createCols 2 m2)
