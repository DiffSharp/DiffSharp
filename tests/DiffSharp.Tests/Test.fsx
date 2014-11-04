
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Forward

let f (x:Dual[]) =
    [| sin (x.[0] * 5. * tan x.[1]) ; cos (x.[1] - x.[0] / x.[1]) |]

let x = [| 1.2; 3.2 |]

let v = [| 3.; 4.|]

let test = jacobianv f x v