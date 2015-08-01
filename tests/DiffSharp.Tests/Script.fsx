
#r "bin/Debug/DiffSharp.dll"

open DiffSharp.AD
open DiffSharp.Util

let inline f x = sin x

let test = diff2' f (D 1.2)