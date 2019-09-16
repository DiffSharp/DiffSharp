module DiffSharp.Util

type Tagger =
    val mutable LastTag:uint32
    new(t) = {LastTag = t}
    member t.Next() = t.LastTag <- t.LastTag + 1u; t.LastTag

type GlobalTagger() =
    static let tagger = Tagger(0u)
    static member Next = tagger.Next()
    static member Reset = tagger.LastTag <- 0u


let getArrayShape (a:System.Array) =
    if a.Length = 0 then [||]
    else Array.init a.Rank (fun i -> a.GetLength(i))

let getShapeLength (shape:int[]) =
    if shape.Length = 0 then 1
    else Array.reduce (*) shape

// type System.Array with
//     static member iteri<'T> (action: int[] -> 'T -> unit) (array:System.Array) =
//         let shape = getArrayShape array
//         printfn "array %A shape %A rank %A" array shape array.Rank
//         let rec arrayforeach (shape:int[]) mapping externalCoords (masterArray:System.Array) =
//             if shape.Length = 1 then
//                 for i = 0 to shape.[0] - 1 do
//                     let globalCoords = Array.append externalCoords [|i|]
//                     let value = downcast masterArray.GetValue(globalCoords)
//                     action globalCoords value
//             else
//                 for i = 0 to shape.[0] - 1 do
//                     arrayforeach shape.[1..] action (Array.append externalCoords [|i|]) masterArray
//         arrayforeach shape action [||] array

let arraysEqual (array1:'a[]) (array2:'a[]) =
    let dim1 = array1.Length
    let dim2 = array2.Length
    if dim1 <> dim2 then false
    else seq {for i in 0..dim1-1 do yield array1.[i] = array2.[i]} |> Seq.forall id

let rec toFlatArrayAndShape<'T> (value:obj) =
    match value with
    | :? 'T as v -> [|v|], [||]
    | :? ('T[]) as v -> v |> Array.toSeq |> toFlatArrayAndShape
    | :? ('T[,]) as v ->
        seq {
            for i=0 to v.GetLength(0)-1 do
                yield seq {
                    for j=0 to v.GetLength(1)-1 do
                        yield v.[i, j]
                }
        } |> toFlatArrayAndShape
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
        } |> toFlatArrayAndShape        
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
        } |> toFlatArrayAndShape    
    | :? seq<'T> as v -> Seq.toArray v, [|Seq.length v|]
    | :? seq<seq<'T>> as v ->
        let arrays, shapes = v |> Seq.map toFlatArrayAndShape |> Seq.toArray |> Array.unzip
        let shape0 = shapes.[0]
        for i=0 to shapes.Length - 1 do
            if not (arraysEqual shape0 shapes.[i]) then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shape0
    | :? seq<seq<seq<'T>>> as v ->
        let arrays, shapes = v |> Seq.map toFlatArrayAndShape |> Seq.toArray |> Array.unzip
        let shape0 = shapes.[0]
        for i=0 to shapes.Length - 1 do
            if not (arraysEqual shape0 shapes.[i]) then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shape0
    | :? seq<seq<seq<seq<'T>>>> as v ->
        let arrays, shapes = v |> Seq.map toFlatArrayAndShape |> Seq.toArray |> Array.unzip
        let shape0 = shapes.[0]
        for i=0 to shapes.Length - 1 do
            if not (arraysEqual shape0 shapes.[i]) then invalidArg "value" "Expecting a rectangular sequence"
        Array.reduce (Array.append) arrays, Array.append [|(v |> Seq.length)|] shape0
    | _ -> invalidArg "value" "Cannot convert value to flat array and shape"

// let rec arrayShapeToString (array:'a[]) (shape:int[]) =
//     let sb = System.Text.StringBuilder()
//     for i=0 to shape.length - 1 do
        
let splitList (n:int) (l:'a list) =
    let size = l.Length / n
    seq {
        let r = ResizeArray()
        for x in l do
            r.Add(x)
            if r.Count = size then
                yield r.ToArray() |> Array.toList
                r.Clear()
        if r.Count > 0 then 
            yield r.ToArray() |> Array.toList
    } |> Seq.toList
