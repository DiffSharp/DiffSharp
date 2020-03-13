module DiffSharp.Util
open System
open System.Collections
open System.Collections.Generic
open FSharp.Reflection

let logSqrt2Pi = log(sqrt(2. * Math.PI))
let log10Val = log 10.

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

let inline cumulativeSum (a:_[]) = (Array.scan (+) LanguagePrimitives.GenericZero a).[1..]

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
    static member ChoiceIndex(probs:float[]) =
        let probsSum = probs |> Array.sum
        let cumulativeProbs = probs |> Array.map (fun v -> v / probsSum) |> cumulativeSum
        let p = rnd.NextDouble()
        cumulativeProbs |> Array.findIndex (fun v -> v >= p)
    static member Choice(array:_[]) = array.[rnd.Next(array.Length)]
    static member Choice(array:_[], probs:float[]) = 
        if array.Length <> probs.Length then failwith "Expecting array and probs of same length"
        array.[Random.ChoiceIndex(probs)]

let arrayShape (a:System.Array) =
    if a.Length = 0 then [||]
    else Array.init a.Rank (fun i -> a.GetLength(i))

let shapeLength (shape:int[]) =
    if shape.Length = 0 then 1
    else Array.reduce (*) shape

let rec shapeSqueeze (dim:int) (shape:int[]) =
    if dim = -1 then
        [|for s in shape do if s <> 1 then yield s|]
    elif shape.[dim] = 1 then
        [|for i=0 to shape.Length - 1 do 
            if i < dim then yield shape.[i]
            elif i > dim then yield shape.[i]|]
    else
        shape

let shapeUnsqueeze (dim:int) (shape:int[]) =
    if dim < 0 || dim > shape.Length then failwithf "Expecting dim in range [0, %A]" shape.Length
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
    Array2D.init location.Length 3 (fun i j -> if j=0 then location.[i] elif j=1 then location.[i] + shape.[i] - 1 else 1)

let duplicates l =
   l |> List.ofSeq
   |> List.groupBy id
   |> List.choose ( function
          | _, x::_::_ -> Some x
          | _ -> None )

let hasDuplicates l =
    (duplicates l) |> List.isEmpty |> not
        
let inline arraysApproximatelyEqual (tolerance:'T) (array1:'T[]) (array2:'T[]) =
    let dim1 = array1.Length
    let dim2 = array2.Length
    if dim1 <> dim2 then false
    else seq {for i in 0..dim1-1 do yield (abs(array1.[i] - array2.[i]) <= tolerance) } |> Seq.forall id

let allEqual (items:seq<'a>) =
    let item0 = items |> Seq.head
    items |> Seq.forall ((=) item0)

let canExpandShape (oldShape: int[]) (newShape: int[]) =
    newShape.Length >= oldShape.Length &&
    let trim = newShape.Length - oldShape.Length
    (oldShape,newShape.[trim..]) ||> Array.forall2 (fun n m -> n = 1 || n = m)

let checkCanExpandShape (oldShape: int[]) (newShape: int[]) =
    let isOK = canExpandShape oldShape newShape
    if not isOK then failwithf "can't expand from shape %A to %A - each dimension must either be equal or expand from 1" oldShape newShape

let checkCanStack (shapes:seq<int[]>) =
    if not (allEqual shapes) then failwith "Cannot stack Tensors with same shapes"

let checkCanUnstack (dim:int) =
    if dim < 1 then failwith "Cannot unstack scalar Tensor (dim < 1)"

let checkCanTranspose (dim:int) =
    if dim <> 2 then failwith "Cannot transpose Tensor when dim=2"

let checkCanFlip (dim:int) (dims:int[]) =
    if dims.Length > dim then failwithf "Expecting dims (list of dimension indices to flip) of length less than Tensor's dimensions, received %A, %A" dims.Length dim
    if hasDuplicates dims then failwithf "Expecting dims (list of dimension indices to flip) without repetition, received %A" dims
    if (Array.max dims) >= dim then failwithf "Expecting dims (list of dimension indices to flip) where all indices are less than the tensor dimension, received %A, %A" dims dim

let checkCanDilate (dim:int) (dilations:int[]) =
    if dilations.Length <> dim then failwithf "Expecting dilations (dilation to use in each dimension) of same length with Tensor's dimensions, received %A, %A" dilations.Length dim
    if (Array.min dilations) < 1 then failwithf "Expecting dilations (dilation to use in each dimension) >= 1 where 1 represents no dilation, received %A" dilations

let checkCanView (shape1:int[]) (shape2:int[]) =
    if shapeLength shape1 <> shapeLength shape2 then failwithf "Cannot view Tensor of shape %A as shape %A" shape1 shape2

let checkCanAddSlice (shape1:int[]) (location:int[]) (shape2:int[]) =
    if not (shapeContains shape1 shape2) then failwithf "Expecting shape1 to contain shape2, received %A, %A" shape1 shape2
    if location.Length <> shape1.Length then failwithf "Expecting location of the same length as shape1, received %A, %A" (location.Length) shape1

/// Find the shape into which shape1 and shape2 can be expanded
let broadcastShapes2 (shape1:int[]) (shape2:int[]) =
    if canExpandShape shape1 shape2 || canExpandShape shape2 shape1 then 
        let n1 = shape1.Length
        let n2 = shape2.Length
        let mx = max n1 n2
        let mn = mx - min n1 n2
        Array.init mx (fun i -> 
            if i < mn then (if n1 > n2 then shape1.[i] else shape2.[i])
            elif n1 > n2 then max shape1.[i] shape2.[i-mn]
            else max shape1.[i-mn] shape2.[i])
    else failwithf "shapes %A and %A are not related by broadcasting - each dimension must either be extra, equal, expand from 1" shape1 shape2

/// Find the shape into which all the shapes can be expanded
let broadcastShapes (shapes:int[][]) = Array.reduce broadcastShapes2 shapes

let boundsToLocation (bounds:int[,]) =
    [|for i=0 to bounds.GetLength(0) - 1 do yield bounds.[i, 0]|]

let boundsToShape (bounds:int[,]) =
    [|for i=0 to bounds.GetLength(0) - 1 do yield bounds.[i, 1] - bounds.[i, 0] + 1|] 

let shapeComplete (nelement:int) (shape:int[]) =
    if (shape |> Array.filter (fun x -> x < -1) |> Array.length) > 0 then failwithf "Invalid shape %A" shape
    let numUnspecified = shape |> Array.filter ((=) -1) |> Array.length
    if numUnspecified > 1 then
        failwithf "Cannot complete shape %A, expecting at most one unspecified dimension (-1)" shape
    elif numUnspecified = 0 then 
        shape
    else
        let divisor = shape |> Array.filter ((<>) -1) |> shapeLength
        if nelement % divisor <> 0 then failwithf "Cannot complete shape %A to have %A elements" shape nelement
        let missing = nelement / divisor
        [|for d in shape do if d = -1 then yield missing else yield d|]

let mirrorCoordinates (coordinates:int[]) (shape:int[]) (mirrorDims:int[]) =
    if coordinates.Length <> shape.Length then failwithf "Expecting coordinates and shape of the same dimension, received %A, %A" coordinates.Length shape.Length
    let result = Array.copy coordinates
    for d=0 to coordinates.Length-1 do
        if mirrorDims |> Array.contains d then
            result.[d] <- abs (coordinates.[d] - shape.[d] + 1)
    result

let dilatedShape (shape:int[]) (dilations:int[]) =
    Array.map2 (fun n d -> n + (n - 1) * (d - 1)) shape dilations

let undilatedShape (shape:int[]) (dilations:int[]) =
    Array.map2 (fun n d -> (n + d - 1) / d) shape dilations

let dilatedCoordinates (coordinates:int[]) (dilations:int[]) =
    Array.map2 (*) coordinates dilations

/// Create a non-jagged 3D array from jagged data
let array3D data = 
    let data = data |> Array.ofSeq |> Array.map array2D
    let r1, r2, r3 = data.Length, data.[0].GetLength(0), data.[0].GetLength(1)
    for i in 0 .. r1-1 do 
        let q2 = data.[i].GetLength(0)
        let q3 = data.[i].GetLength(1)
        if q2 <> r2 || q3 <> r3 then 
            invalidArg "data" (sprintf "jagged input at position %d: first is _ x %d x %d, later is _ x _ x %d x %d" i r2 r3 q2 q3)
    Array3D.init r1 r2 r3 (fun i j k -> data.[i].[j,k])

/// Create a non-jagged 4D array from jagged data
let array4D data = 
    let data = data |> array2D |> Array2D.map array2D
    let r1,r2,r3,r4 = (data.GetLength(0), data.GetLength(1), data.[0,0].GetLength(0),data.[0,0].GetLength(1))
    for i in 0 .. r1-1 do 
      for j in 0 .. r2-1 do 
        let q3 = data.[i,j].GetLength(0)
        let q4 = data.[i,j].GetLength(1)
        if q3 <> r3 || q4 <> r4 then 
            invalidArg "data" (sprintf "jagged input at position (%d,%d): first is _ x _ x %d x %d, later is _ x _ x %d x %d" i j r2 r3 q3 q4)
    Array4D.init r1 r2 r3 r4 (fun i j k m -> data.[i,j].[k,m])

let arrayND (shape: int[]) f =
    match shape with 
    | [| |] -> f [| |] |> box
    | [| d0 |] -> Array.init d0 (fun i -> f [| i |]) |> box
    | [| d0; d1 |] -> Array2D.init d0 d1 (fun i1 i2 -> f [| i1; i2 |]) |> box
    | [| d0; d1; d2 |] -> Array3D.init d0 d1 d2 (fun i1 i2 i3 -> f [| i1; i2; i3 |]) |> box
    | [| d0; d1; d2; d3 |] -> Array4D.init d0 d1 d2 d3 (fun i1 i2 i3 i4 -> f [| i1; i2; i3; i4 |]) |> box
    | _ -> failwith "arrayND - nyi for dim > 4"

/// Get the elements of an arbitrary IEnumerble
let private seqElements (ie: obj) = 
    let e = (ie :?> IEnumerable).GetEnumerator()
    [| while e.MoveNext() do yield e.Current |]

/// Match an array type of arbitrary rank
let private (|ArrayTy|_|) (ty: Type) = 
    if ty.IsArray && ty.GetArrayRank() <= 4 then
        Some(ty.GetArrayRank(), ty.GetElementType())
    else 
       None

/// Match an tuple type
let private (|TupleTy|_|) (ty: Type) = 
    if FSharpType.IsTuple ty then 
        Some(FSharpType.GetTupleElements ty)
    else 
       None

let rec private  (|ListTy|_|) (ty: Type) = 
    if ty.IsGenericType && ty.GetGenericTypeDefinition().Equals(typedefof<list<int>>) then
       Some (ty.GetGenericArguments().[0])
    else   
        None

/// Match a 1D sequence type (seq<_>) or a subclass
let rec private  (|SeqTy|_|) (ty: Type) = 
    if ty.IsGenericType && ty.GetGenericTypeDefinition().Equals(typedefof<seq<int>>) then
       Some (ty.GetGenericArguments().[0])
    else   
        match ty.BaseType with 
        | null -> None 
        | _ -> 
            match ty.BaseType with 
            | SeqTy ety -> Some ety
            | _ -> 
                ty.GetInterfaces() |> Array.tryPick (|SeqTy|_|)

let rec formatType (ty: Type) = 
    match ty with 
    | ListTy ety -> sprintf "list<%s>" (formatType ety)
    | ArrayTy (_,ety) -> sprintf "%s[]" (formatType ety)
    | SeqTy ety -> sprintf "seq<%s>" (formatType ety)
    | TupleTy etys -> String.concat "*" (Array.map formatType etys)
    | ty when ty = typeof<int64> -> "int64"
    | ty when ty = typeof<int> -> "int"
    | ty when ty = typeof<double> -> "double"
    | ty when ty = typeof<float32> -> "float32"
    | _ -> ty.ToString()

let private (|SeqTupleTy|_|) (ty: Type) = 
    match ty with 
    | SeqTy (TupleTy etys) -> 
        match etys |> Array.tryFind (fun ety -> ety <> etys.[0]) with
        | None -> ()
        | Some ety2 -> failwithf "jagged input: unexpected mixed types in tuple being used as sequence notation, %s and %s" (formatType etys.[0]) (formatType ety2)
        Some (etys.[0])
    | _ -> None

let private (|TupleLeafTy|_|) (tgt: Type) (ty: Type) = 
    match ty with 
    | TupleTy etys when etys |> Array.forall (fun ety -> ety = tgt) -> Some ()
    | _ -> None

let private (|SeqTupleLeafTy|_|) (tgt: Type) (ty: Type) = 
    match ty with 
    | SeqTy (TupleLeafTy tgt) -> Some ()
    | _ -> None

let private flatArrayAndShape1D<'T> (v: 'T[]) =
    v, [|Array.length v|]

let private flatArrayAndShape2D<'T> (v: 'T[,]) =
    let n1 = Array2D.length1 v
    let n2 = Array2D.length2 v
    let arr =
        [|  for i=0 to n1-1 do
                for j=0 to n2-1 do
                   yield v.[i, j] |]
    arr, [| n1;n2|]

let private flatArrayAndShape3D<'T> (v: 'T[,,]) =
    let n1 = Array3D.length1 v
    let n2 = Array3D.length2 v
    let n3 = Array3D.length3 v
    let arr =
        [|  for i=0 to n1-1 do
                for j=0 to n2-1 do
                    for k=0 to n3-1 do
                        yield v.[i, j, k] |]
    arr, [| n1;n2;n3 |]

let private flatArrayAndShape4D<'T> (v: 'T[,,,]) =
    let n1 = Array4D.length1 v
    let n2 = Array4D.length2 v
    let n3 = Array4D.length3 v
    let n4 = Array4D.length4 v
    let arr =
        [|  for i=0 to n1-1 do
                for j=0 to n2-1 do
                    for k=0 to n3-1 do
                        for m=0 to n4-1 do
                            yield v.[i, j, k, m] |]
    arr, [| n1;n2;n3;n4 |]

let private seqTupleElements (els: obj) =
    match seqElements els with 
    | [| el |] -> FSharpValue.GetTupleFields(el) 
    | tup -> failwithf "unexpected multiple values in tuple list input: %A" (Array.toList tup)

let private arrayCast<'T> (els: obj[]) = els |> Array.map (fun v -> v :?> 'T)

let private (|SeqOrSeqTupleTy|_|) ty =
    match ty with 
    | SeqTupleTy ety -> Some (seqTupleElements, ety)
    | SeqTy ety -> Some (seqElements, ety)
    | _ -> None

let private (|SeqOrSeqTupleLeafTy|_|) tgt ty =
    match ty with 
    | SeqTupleLeafTy tgt -> Some (seqTupleElements)
    | SeqTy ety when ety = tgt -> Some (seqElements)
    | _ -> None

let rec flatArrayAndShape<'T> (value:obj) =

    match value with
    | :? 'T as v -> [|v|], [||]
    | :? ('T[]) as v -> flatArrayAndShape1D v
    | :? ('T[,]) as v -> flatArrayAndShape2D<'T> v
    | :? ('T[,,]) as v -> flatArrayAndShape3D<'T> v
    | :? ('T[,,,]) as v -> flatArrayAndShape4D<'T> v
    | :? seq<'T> as v -> flatArrayAndShape1D (Seq.toArray v)
    | :? seq<seq<'T>> as v -> flatArrayAndShape2D (array2D v)
    | :? seq<seq<seq<'T>>> as v -> flatArrayAndShape3D (array3D v)
    | :? seq<seq<seq<seq<'T>>>> as v -> flatArrayAndShape4D (array4D v)
    | _ -> 
    let vty = value.GetType()
    let tgt = (typeof<'T>)
    match vty with
    // list<int * int> -> dim 1
    | SeqTupleLeafTy tgt -> 
        let arr = value |> seqTupleElements |> arrayCast<'T>
        arr, [| arr.Length |]
    // list<list<int * int>> etc. -> dim 2
    | SeqOrSeqTupleTy (fetcher, (SeqOrSeqTupleLeafTy tgt fetcher2)) -> 
        let els = value |> fetcher |> Array.map (fetcher2 >> arrayCast<'T>) |> array2D
        flatArrayAndShape2D<'T> els
    // ... -> dim 3
    | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleLeafTy tgt fetcher3)) -> 
        let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> arrayCast<'T>)) |> array3D
        flatArrayAndShape3D<'T> els
    // ... -> dim 4
    | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleTy (fetcher3, SeqOrSeqTupleLeafTy tgt fetcher4))) -> 
        let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> Array.map (fetcher4 >> arrayCast<'T>))) |> array4D
        flatArrayAndShape4D<'T> els
    | _ -> null, null

let toInt a =
    match box a with
    | :? float as a -> a |> int
    | :? float32 as a -> a |> int
    | :? int as a -> a
    | _ -> failwith "Cannot convert to int"

let inline notNull value = not (obj.ReferenceEquals(value, null))

let maxIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.maxBy snd |> fst
let minIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.minBy snd |> fst

let memoize fn =
  let cache = new Dictionary<_,_>()
  (fun x ->
    match cache.TryGetValue x with
    | true, v -> v
    | false, _ -> let v = fn (x)
                  cache.Add(x,v)
                  v)

let getKeys (dictionary:Dictionary<string, 'a>) =
    let keys = Array.create dictionary.Count ""
    dictionary.Keys.CopyTo(keys, 0)
    keys