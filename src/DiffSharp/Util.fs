module DiffSharp.Util
open System

let logSqrt2Pi = log(sqrt(2. * Math.PI))

type NestingLevel =
    val mutable Current:uint32
    new() = {Current = 0u}
    member t.Next() = t.Current <- t.Current + 1u; t.Current

type GlobalNestingLevel() =
    static let tagger = NestingLevel()
    static member Current = tagger.Current
    static member Next() = tagger.Next()
    static member Reset() = tagger.Current <- 0u
    static member Set(level) = tagger.Current <- level

type Random() =
    static let mutable rnd = System.Random()
    static member Seed(seed) = rnd <- System.Random(seed)
    static member Uniform() = rnd.NextDouble()
    static member Uniform(low, high) = low + (rnd.NextDouble() * (high-low))
    static member Normal() =
        let rec normal() = 
            let x, y = (rnd.NextDouble()) * 2.0 - 1.0, (rnd.NextDouble()) * 2.0 - 1.0
            let s = x * x + y * y
            if s > 1.0 then normal() else x * sqrt (-2.0 * (log s) / s)
        normal()
    static member Normal(mean, stddev) = mean + Random.Normal() * stddev

let arrayShape (a:System.Array) =
    if a.Length = 0 then [||]
    else Array.init a.Rank (fun i -> a.GetLength(i))

let shapeLength (shape:int[]) =
    if shape.Length = 0 then 1
    else Array.reduce (*) shape

let shapeSqueeze (dim:int) (shape:int[]) =
    if dim = -1 then
        [|for s in shape do if s <> 1 then yield s|]
    elif shape.[dim] = 1 then
        [|for i=0 to shape.Length - 1 - 1 do 
            if i < dim then yield shape.[i]
            elif i > dim then yield shape.[i+1]|]
    else shape

let shapeUnsqueeze (dim:int) (shape:int[]) =
    [|for i=0 to shape.Length - 1 + 1 do 
        if i < dim then yield shape.[i]
        elif i = dim then yield 1
        else yield shape.[i-1]|]

let shapeUnsqueezeAs (shape1:int[]) (shape2:int[]) =
    if shape1.Length > shape2.Length then failwithf "Expecting shape1.Length <= shape2.Length, received %A %A" shape1.Length shape2.Length
    let ones = Array.create (shape2.Length - shape1.Length) 1
    Array.append ones shape1

let shapeContains (bigShape:int[]) (smallShape:int[]) =
    if bigShape.Length <> smallShape.Length then failwithf "Expecting shapes with same dimension, received %A %A" bigShape.Length smallShape.Length
    Array.map2 (<=) smallShape bigShape |> Array.forall id

let shapeLocationToBounds (shape:int[]) (location:int[]) =
    Array2D.init location.Length 2 (fun i j -> if j=0 then location.[i] else location.[i] + shape.[i] - 1)

let boundsToLocation (bounds:int[,]) =
    [|for i=0 to bounds.GetLength(0) do yield bounds.[i, 0]|]

let inline arraysApproximatelyEqual (tolerance:'T) (array1:'T[]) (array2:'T[]) =
    let dim1 = array1.Length
    let dim2 = array2.Length
    if dim1 <> dim2 then false
    else seq {for i in 0..dim1-1 do yield (abs(array1.[i] - array2.[i]) < tolerance) } |> Seq.forall id

let allEqual (items:seq<'a>) =
    let item0 = items |> Seq.head
    items |> Seq.forall ((=) item0)

let rec flatArrayAndShape<'T> (value:obj) =
    match value with
    | :? 'T as v -> [|v|], [||]
    | :? ('T[]) as v -> v |> Array.toSeq |> flatArrayAndShape<'T>
    | :? ('T[,]) as v ->
        seq {
            for i=0 to v.GetLength(0)-1 do
                yield seq {
                    for j=0 to v.GetLength(1)-1 do
                        yield v.[i, j]
                }
        } |> flatArrayAndShape<'T>
    | :? ('T[,,]) as v ->
        seq {
            for i=0 to v.GetLength(0)-1 do
                yield seq {
                    for j=0 to v.GetLength(1)-1 do
                        yield seq {
                            for k=0 to v.GetLength(2)-1 do
                                yield v.[i, j, k]
                        }
                }
        } |> flatArrayAndShape<'T>        
    | :? ('T[,,,]) as v ->
        seq {
            for i=0 to v.GetLength(0)-1 do
                yield seq {
                    for j=0 to v.GetLength(1)-1 do
                        yield seq {
                            for k=0 to v.GetLength(2)-1 do
                                yield seq {
                                    for l=0 to v.GetLength(3)-1 do
                                        yield v.[i, j, k, l]
                                }
                        }
                }
        } |> flatArrayAndShape<'T>    
    | :? seq<'T> as v -> Seq.toArray v, [|Seq.length v|]
    | :? seq<seq<'T>> as v ->
        let arrays, shapes = v |> Seq.map flatArrayAndShape<'T> |> Seq.toArray |> Array.unzip
        if not (allEqual shapes) then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shapes.[0]
    | :? seq<seq<seq<'T>>> as v ->
        let arrays, shapes = v |> Seq.map flatArrayAndShape<'T> |> Seq.toArray |> Array.unzip
        if not (allEqual shapes) then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shapes.[0]
    | :? seq<seq<seq<seq<'T>>>> as v ->
        let arrays, shapes = v |> Seq.map flatArrayAndShape<'T> |> Seq.toArray |> Array.unzip
        if not (allEqual shapes) then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shapes.[0]
    | _ -> null, null
    // TODO: add list of tuples parsing


let inline notNull value = not (obj.ReferenceEquals(value, null))