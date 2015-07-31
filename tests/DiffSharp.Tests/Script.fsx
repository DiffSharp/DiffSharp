
#r "bin/Debug/DiffSharp.dll"

open DiffSharp.AD
open DiffSharp.Util

let f (x:DV) = x.[0] + x.[1]
let g = grad f (vector [D 1.1; D 1.2])
g.[1] |> float