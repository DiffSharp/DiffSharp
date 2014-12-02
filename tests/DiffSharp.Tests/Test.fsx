#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.ForwardGH

let test = hessian (fun x -> abs (x.[0] + sin x.[1])) [|0.; 0.|]
