module DiffSharp.Util

type Tagger =
    val mutable LastTag:uint32
    new(t) = {LastTag = t}
    member t.Next() = t.LastTag <- t.LastTag + 1u; t.LastTag

type GlobalTagger() =
    static let tagger = Tagger(0u)
    static member Next = tagger.Next()
    static member Reset = tagger.LastTag <- 0u

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

let getArrayShape (a:System.Array) =
    if a.Length = 0 then [||]
    else Array.init a.Rank (fun i -> a.GetLength(i))

let getShapeLength (shape:int[]) =
    if shape.Length = 0 then 1
    else Array.reduce (*) shape

let inline arraysApproximatelyEqual (tolerance:'T) (array1:'T[]) (array2:'T[]) =
    let dim1 = array1.Length
    let dim2 = array2.Length
    if dim1 <> dim2 then false
    else seq {for i in 0..dim1-1 do yield (abs(array1.[i] - array2.[i]) < tolerance) } |> Seq.forall id

let rec toFlatArrayAndShape<'T> (value:obj) =
    match value with
    | :? 'T as v -> [|v|], [||]
    | :? ('T[]) as v -> v |> Array.toSeq |> toFlatArrayAndShape<'T>
    | :? ('T[,]) as v ->
        seq {
            for i=0 to v.GetLength(0)-1 do
                yield seq {
                    for j=0 to v.GetLength(1)-1 do
                        yield v.[i, j]
                }
        } |> toFlatArrayAndShape<'T>
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
        } |> toFlatArrayAndShape<'T>        
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
        } |> toFlatArrayAndShape<'T>    
    | :? seq<'T> as v -> Seq.toArray v, [|Seq.length v|]
    | :? seq<seq<'T>> as v ->
        let arrays, shapes = v |> Seq.map toFlatArrayAndShape<'T> |> Seq.toArray |> Array.unzip
        let shape0 = shapes.[0]
        for i=0 to shapes.Length - 1 do
            if shape0 <> shapes.[i] then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shape0
    | :? seq<seq<seq<'T>>> as v ->
        let arrays, shapes = v |> Seq.map toFlatArrayAndShape<'T> |> Seq.toArray |> Array.unzip
        let shape0 = shapes.[0]
        for i=0 to shapes.Length - 1 do
            if shape0 <> shapes.[i] then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shape0
    | :? seq<seq<seq<seq<'T>>>> as v ->
        let arrays, shapes = v |> Seq.map toFlatArrayAndShape<'T> |> Seq.toArray |> Array.unzip
        let shape0 = shapes.[0]
        for i=0 to shapes.Length - 1 do
            if shape0 <> shapes.[i] then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shape0
    | _ -> null, null
    // TODO: add list of tuples parsing


let inline notNull value = not (obj.ReferenceEquals(value, null))