
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

open DiffSharp.AD.Forward

let df = diff (fun x -> 
                    if x = 1Q then
                        1Q
                    else
                        x - 1.)

let test = df 1.