
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.Symbolic

//let g = 
//    hessian (fun x ->
//        let theta = array x
//        let n = (Array.length theta - 1) / 2
//        let weights = theta.[0 .. n-1]
//        let lambdas = theta.[n .. 2*n-1]
//        let parameters = Array.zip weights lambdas
//        let kappa = theta.[2*n]
//        let bins = [| 0., 2500., 58. ; 2500., 7500., 61. |]
//        let binValue (low,high,count) = 
//            count * log(parameters |> Array.sumBy(fun (w,l) -> w * (exp(-l*low) - exp(-l*high))))
//        -(bins |> Array.sumBy binValue) - kappa * (1. - Array.sum weights))     
//
//let test =  g (vector [| 0.8; 0.2; 1./10000.; 1./100000.; 336. |])


let test = hessian <@ fun x y -> atan2 x y @> [|0.3; 0.2|]

//let test = diffn 2 <@ fun x -> x @> 2.3

open DiffSharp.AD.Reverse

let test2 = hessian (fun x -> atan2 x.[0] x.[1]) [|0.3; 0.2|]

//let test2 = diff2 (fun x -> x) 2.3