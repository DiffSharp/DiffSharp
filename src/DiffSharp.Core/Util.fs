// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

/// Contains utilities related to the DiffSharp programming model.
namespace DiffSharp.Util

open System
open System.Collections
open System.Diagnostics.CodeAnalysis
open FSharp.Reflection
open DiffSharp

/// Represents a differentiation nesting level.
type NestingLevel =
    val mutable Current:uint32
    new() = {Current = 0u}
    member t.Next() = t.Current <- t.Current + 1u; t.Current

/// Contains operations to get, set or reset the global nesting level for differentiation operations.
type GlobalNestingLevel() =
    static let tagger = NestingLevel()
    static member Current = tagger.Current
    static member Next() = tagger.Next()
    static member Reset() = tagger.Current <- 0u
    static member Set(level) = tagger.Current <- level

/// Contains operations relating to pseudo-random number generation.
type Random() =
    static let mutable rnd = System.Random()

    /// Sets the random seed.
    static member Seed(seed) = rnd <- System.Random(seed)

    /// Samples a random value from the standard uniform distribution over the interval [0,1).
    static member Uniform() = rnd.NextDouble()

    /// Samples a random value from the uniform distribution with the given parameters [low, high).
    static member Uniform(low, high) = low + (rnd.NextDouble() * (high-low))

    /// Samples a random value from the standard normal distribution with mean 0 and standard deviation 1.
    static member Normal() =
        let rec normal() = 
            let x, y = (rnd.NextDouble()) * 2.0 - 1.0, (rnd.NextDouble()) * 2.0 - 1.0
            let s = x * x + y * y
            if s > 1.0 then normal() else x * sqrt (-2.0 * (log s) / s)
        normal()

    /// Samples a random value from the normal distribution with the given mean and standard deviation.
    static member Normal(mean, stddev) = mean + Random.Normal() * stddev

    /// Samples a random integer in the given range [low, high).
    static member Integer(low, high) = rnd.Next(low, high)

    /// Samples an index at random with the given categorical probabilities.
    static member ChoiceIndex(probs:float[]) =
        let probsSum = probs |> Array.sum
        let cumulativeProbs = probs |> Array.map (fun v -> v / probsSum) |> Array.cumulativeSum
        let p = rnd.NextDouble()
        cumulativeProbs |> Array.findIndex (fun v -> v >= p)

    /// Samples a value at random from the given array.
    static member Choice(array:_[]) = array.[rnd.Next(array.Length)]

    /// Samples a value at random from the given array using the given categorical probabilities.
    static member Choice(array:_[], probs:float[]) = 
        if array.Length <> probs.Length then failwith "Expecting array and probs of same length"
        array.[Random.ChoiceIndex(probs)]

    /// Samples a number of random values  array of random values for the given weighted distribution
    static member Multinomial(probs:float[], numSamples:int) =
        Array.init numSamples (fun _ -> Random.ChoiceIndex(probs)) // Samples with replacement

    /// Returns a 2D array where each row contains `numSamples` indices sampled from the multinomial probability distribution defined by the probabilities in the corresponding row of the `probs` array.
    static member Multinomial(probs:float[,], numSamples:int) =
        Array2D.init (probs.GetLength(0)) numSamples (fun i _ -> Random.ChoiceIndex(probs.[i,*])) // Samples with replacement

    /// Samples a random value from the Bernoulli distribution with the given probability.
    static member Bernoulli(prob:float) = if rnd.NextDouble() < prob then 1. else 0.

    /// Samples a random value from the Bernoulli distribution.
    static member Bernoulli() = Random.Bernoulli(0.5)

    /// Returns a universally unique identifier (UUID) string
    static member UUID() = System.Guid.NewGuid().ToString()

    /// Returns an array that is a randomly-shuffled version of the given array, using the Durstenfeld/Knuth shuffle.
    static member Shuffle(array:_[]) =
        // Durstenfeld/Knuth shuffle
        let a = array |> Array.copy
        let mutable n = array.Length
        while n > 1 do
            n <- n - 1
            let i = rnd.Next(n+1)
            let temp = a.[i]
            a.[i] <- a.[n]
            a.[n] <- temp
        a

/// Contains operations relating to pseudo-random number generation.
module Random = 

    /// Returns a function that maps a given index to a shuffled version of the indexes up to the given `length`
    let shuffledIndices (length: int) =
        let indices = Array.init length id
        let indicesShuffled = Random.Shuffle(indices)
        fun (i: int) -> indicesShuffled.[i]

/// Contains operations relating to converting .NET data to tensor data.
module DataConverter =

    /// Gets the elements of an arbitrary IEnumerble.
    let private seqElements (ie: obj) = 
        let e = (ie :?> IEnumerable).GetEnumerator()
        [| while e.MoveNext() do yield e.Current |]

    /// Matches an array type of arbitrary rank.
    let private (|ArrayTy|_|) (ty: Type) = 
        if ty.IsArray && ty.GetArrayRank() <= 4 then
            Some(ty.GetArrayRank(), ty.GetElementType())
        else 
           None

    /// Matches a tuple type.
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

    /// Matches a 1D sequence type (seq<_>) or a subclass.
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

    let rec tryFlatArrayAndShape<'T> (value:obj) : ('T[] * int[]) option =

        match value with
        | :? 'T as v -> Some ([|v|], [||])
        | :? ('T[]) as v -> Some (flatArrayAndShape1D v)
        | :? ('T[,]) as v -> Some (flatArrayAndShape2D<'T> v)
        | :? ('T[,,]) as v -> Some (flatArrayAndShape3D<'T> v)
        | :? ('T[,,,]) as v -> Some (flatArrayAndShape4D<'T> v)
        | :? seq<'T> as v -> Some (flatArrayAndShape1D (Seq.toArray v))
        | :? seq<seq<'T>> as v -> Some (flatArrayAndShape2D (array2D v))
        | :? seq<seq<seq<'T>>> as v -> Some (flatArrayAndShape3D (array3D v))
        | :? seq<seq<seq<seq<'T>>>> as v -> Some (flatArrayAndShape4D (array4D v))
        | _ -> 
        let vty = value.GetType()
        let tgt = (typeof<'T>)
        match vty with
        // list<int * int> -> dim 1
        | SeqTupleLeafTy tgt -> 
            let arr = value |> seqTupleElements |> arrayCast<'T>
            Some (arr, [| arr.Length |])
        // list<list<int * int>> etc. -> dim 2
        | SeqOrSeqTupleTy (fetcher, (SeqOrSeqTupleLeafTy tgt fetcher2)) -> 
            let els = value |> fetcher |> Array.map (fetcher2 >> arrayCast<'T>) |> array2D
            Some (flatArrayAndShape2D<'T> els)
        // ... -> dim 3
        | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleLeafTy tgt fetcher3)) -> 
            let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> arrayCast<'T>)) |> array3D
            Some (flatArrayAndShape3D<'T> els)
        // ... -> dim 4
        | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleTy (fetcher3, SeqOrSeqTupleLeafTy tgt fetcher4))) -> 
            let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> Array.map (fetcher4 >> arrayCast<'T>))) |> array4D
            Some (flatArrayAndShape4D<'T> els)
        | _ -> None

    [<ExcludeFromCodeCoverage>]
    let inline dataOfValues ofFloat32 ofFloat64 ofInt8 ofInt16 ofInt32 ofInt64 ofBool ofByte (value:obj) : (^T[] * int[]) = 
        match value |> tryFlatArrayAndShape<float32> with
        | Some (values, shape) -> (values |> Array.map ofFloat32, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<double> with
        | Some (values, shape) -> (values |> Array.map ofFloat64, shape) 
        | None -> 
        match value |> tryFlatArrayAndShape<int32> with
        | Some (values, shape) -> (values |> Array.map ofInt32, shape) 
        | None -> 
        match value |> tryFlatArrayAndShape<int64> with
        | Some (values, shape) -> (values |> Array.map ofInt64, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<int8>  with
        | Some (values, shape) -> (values |> Array.map ofInt8, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<byte>  with
        | Some (values, shape) -> (values |> Array.map ofByte, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<int16>  with
        | Some (values, shape) -> (values |> Array.map ofInt16, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<bool> with
        | Some (values, shape) -> (values |> Array.map ofBool, shape) 
        | None -> 
        match value with
        | :? (obj[]) as v when Array.isEmpty v -> ([||] |> Array.map ofFloat32, [|0|])
        | :? (seq<obj>) as v when Seq.isEmpty v -> ([||] |> Array.map ofFloat32, [|0|])
        | _ -> failwithf "Cannot convert value of type %A to RawTensorCPU" (value.GetType())

    let dataOfValuesForFloat32 (value:obj) =
        dataOfValues float32 float32 float32 float32 float32 float32 (fun x -> if x then 1.0f else 0.0f) float32 value 

    let dataOfValuesForFloat64 (value:obj) =
        dataOfValues double double double double double double (fun x -> if x then 1.0 else 0.0) double value 

    let dataOfValuesForByte (value:obj) =
        dataOfValues byte byte byte byte byte byte (fun x -> if x then 1uy else 0uy) id value 

    let dataOfValuesForInt8 (value:obj) =
        dataOfValues int8 int8 int8 int8 int8 int8 (fun x -> if x then 1y else 0y) int8 value 

    let dataOfValuesForInt16 (value:obj) =
        dataOfValues int16 int16 int16 int16 int16 int16 (fun x -> if x then 1s else 0s) int16 value 

    let dataOfValuesForInt32 (value:obj) =
        dataOfValues int32 int32 int32 int32 int32 int32 (fun x -> if x then 1 else 0) int32 value

    let dataOfValuesForInt64 (value:obj) =
        dataOfValues int64 int64 int64 int64 int64 int64 (fun x -> if x then 1L else 0L) int64 value

    let dataOfValuesForBool (value:obj) =
        dataOfValues (fun i -> abs i >= 1.0f) (fun i -> abs i >= 1.0) (fun i -> abs i > 0y) (fun i -> abs i > 0s) (fun i -> abs i > 0) (fun i -> abs i > 0L) id (fun i -> i > 0uy) value 
