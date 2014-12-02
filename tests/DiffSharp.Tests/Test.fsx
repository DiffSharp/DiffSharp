//#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
//
//open DiffSharp.AD.ForwardGH
//
//let test = hessian (fun x -> abs (x.[0] + sin x.[1])) [|0.; 0.|]


let isHalfway (a:float) = abs (a % 1.) = 0.5

let test = isHalfway -10.51