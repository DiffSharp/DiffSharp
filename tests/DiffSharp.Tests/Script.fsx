
#r "bin/Debug/DiffSharp.dll"

open DiffSharp.AD
open DiffSharp.Util

let f (x:DV) = 
    let s = x |> Vector.split [2; 1] |> Seq.toArray
    s.[0].[0] + s.[1].[0]
let g = grad' f (vector [D 1.; D 2.; D 3.; D 4.; D 5.])


let w = vector [D 1.; D 1.; D 1.]
let wr = makeReverse 0u w

