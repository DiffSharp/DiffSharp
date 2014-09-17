// Learn more about F# at http://fsharp.net. See the 'F# Tutorial' project
// for more guidance on F# programming.


#r "F:/GIT/GitHub/gbaydin/DiffSharp/src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.Numerical

//let f (x:DualT[]) = log x.[0] / sinh x.[1]

//let g = grad f [|2.; 5.|]

//let h = hessian f [|2.; 5.|]

//let gh = gradhessian' f [|2.; 5.|]


//let f (x:DualD[]) = [|sin (x.[0] * 3.) * cos x.[1]; tan (x.[0] * x.[1]); x.[0] + x.[1]|]
let f (x:float[]) = sin (x.[0] * 3.) * cos x.[1]

let test = grad f [| 2.; 3.|]
