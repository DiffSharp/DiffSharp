
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Forward2

let test = diff' (fun x -> x ** x) 2.8