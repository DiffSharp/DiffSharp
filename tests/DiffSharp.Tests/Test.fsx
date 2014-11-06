
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Reverse

let f (x:Adj[]) =
    [| sin (x.[0] * 5. * tan x.[2]) ; cos (x.[1] - x.[2] / x.[1]) |]

let x = [| 1.2; 3.2; 2. |]

let v = [| 3.; 4. |]

let test = jacobianTv' f x v
