#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Forward

let test = diff (fun x -> Dual.Sign x) 0.
