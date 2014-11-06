
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.Symbolic


let test = hessian <@ fun (x0:float) x1 -> x1 * sin (x0 * 7. / 2. + x1 * 5.) @> [|2.; 2.|]

