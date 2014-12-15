#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"


open DiffSharp.AD.Forward
open DiffSharp.AD.Forward.Vector
open DiffSharp.Util.LinearAlgebra


let solve (g:Vector<Dual>->Vector<Dual>) x0 a t =
    //let f x = (g x) * (g x)
    let dseq = Seq.unfold (fun x ->
                                let v, j = jacobian' g x
                                let x' = x - a * j * v
                                if Vector.norm (x - x') < t then
                                    None
                                else

                                    Some(x, x')) x0
    (Seq.last dseq, dseq)

let inline f (x:Vector<Dual>) =
    vector [3. * x.[0] - cos (x.[1] * x.[2]) - 3. / 2.
            4. * x.[0] ** 2. - 625. * x.[1] ** 2. + 2. * x.[1] - 1.
            exp (-x.[0] * x.[1]) + 20. * x.[2]]

let xmin, dseq = solve f (vector [0.; 0.; 0.]) 0.001 0.001

let test = f (Vector.map dual xmin)