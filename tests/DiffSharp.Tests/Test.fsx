
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Forward


let g = 
    grad (fun theta ->
        let n = (Array.length theta - 1) / 2
        let weights = theta.[0 .. n-1]
        let lambdas = theta.[n .. 2*n-1]
        let parameters = Array.zip weights lambdas
        let kappa = theta.[2*n]
        let bins = [| 0., 2500., 58. ; 2500., 7500., 61. |]
        let binValue (low,high,count) = 
            count * log(parameters |> Array.sumBy(fun (w,l) -> w * (exp(-l*low) - exp(-l*high))))
        -(bins |> Array.sumBy binValue) - kappa * (1. - Array.sum weights))     

let test =  g [| 0.8; 0.2; 1./10000.; 1./100000.; 336. |]
