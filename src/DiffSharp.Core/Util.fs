// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

/// Contains utilities related to the DiffSharp programming model.
namespace DiffSharp.Util

open System
open System.Collections
open System.Collections.Generic
open System.Diagnostics.CodeAnalysis
open FSharp.Reflection
open System.IO
open System.IO.Compression
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary


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
        // Marsaglia polar method
        // TODO: this is discarding one of the two samples that can be generated. For efficiency, we can keep the second sample around to return it in the next call.
        let rec normal() = 
            let x, y = (rnd.NextDouble()) * 2.0 - 1.0, (rnd.NextDouble()) * 2.0 - 1.0
            let s = x * x + y * y
            if s > 1.0 then normal() else x * sqrt (-2.0 * (log s) / s)
        normal()

    /// Samples a random value from the normal distribution with the given mean and standard deviation.
    static member Normal(mean, stddev) = mean + Random.Normal() * stddev

    /// Samples a double value in the range [0, 1)
    static member Double() = rnd.NextDouble()

    /// Samples a double value in the given range [low, high)
    static member Double(low, high) = 
        if high < low then failwithf "Expecting high >= low"
        low + rnd.NextDouble() * (high-low)

    /// Samples a non-negative random integer
    static member Integer() = rnd.Next()

    /// Samples a random integer in the given range [low, high).
    static member Integer(low, high) = rnd.Next(low, high)

    /// Samples an index at random with the given categorical probabilities.
    static member ChoiceIndex(probs:float[]) =
        let probsSum = probs |> Array.sum
        let cumulativeProbs = probs |> Array.map (fun v -> v / probsSum) |> Array.cumulativeSum
        let p = rnd.NextDouble()
        cumulativeProbs |> Array.findIndex (fun v -> v >= p)

    /// Samples a value at random from the given array.
    static member Choice(array:_[]) = array[rnd.Next(array.Length)]

    /// Samples a value at random from the given array using the given categorical probabilities.
    static member Choice(array:_[], probs:float[]) = 
        if array.Length <> probs.Length then failwith "Expecting array and probs of same length"
        array[Random.ChoiceIndex(probs)]

    /// Samples a number of random values  array of random values for the given weighted distribution
    static member Multinomial(probs:float[], numSamples:int) =
        Array.init numSamples (fun _ -> Random.ChoiceIndex(probs)) // Samples with replacement

    /// Returns a 2D array where each row contains `numSamples` indices sampled from the multinomial probability distribution defined by the probabilities in the corresponding row of the `probs` array.
    static member Multinomial(probs:float[,], numSamples:int) =
        Array2D.init (probs.GetLength(0)) numSamples (fun i _ -> Random.ChoiceIndex(probs[i,*])) // Samples with replacement

    /// Samples a random value from the Bernoulli distribution with the given probability.
    static member Bernoulli(prob:float) = if rnd.NextDouble() < prob then 1. else 0.

    /// Samples a random value from the Bernoulli distribution.
    static member Bernoulli() = Random.Bernoulli(0.5)

    /// Returns a universally unique identifier (UUID) string
    // https://en.wikipedia.org/wiki/Universally_unique_identifier
    static member UUID() = 
        // We don't use System.Guid.NewGuid().ToString() because it relies on a separate randomness source whose seed we cannot control through System.Random(seed)
        let bytes = Array.zeroCreate (sizeof<Guid>)
        rnd.NextBytes(bytes)
        let guid = new Guid(bytes)
        guid.ToString()

    /// Returns an array that is a randomly-shuffled version of the given array, using the Durstenfeld/Knuth shuffle.
    static member Shuffle(array:_[]) =
        // Durstenfeld/Knuth shuffle
        let a = array |> Array.copy
        let mutable n = array.Length
        while n > 1 do
            n <- n - 1
            let i = rnd.Next(n+1)
            let temp = a[i]
            a[i] <- a[n]
            a[n] <- temp
        a

/// Contains operations relating to pseudo-random number generation.
module Random = 

    /// Returns a function that maps a given index to a shuffled version of the indexes up to the given `length`
    let shuffledIndices (length: int) =
        let indices = Array.init length id
        let indicesShuffled = Random.Shuffle(indices)
        fun (i: int) -> indicesShuffled[i]

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
           Some (ty.GetGenericArguments()[0])
        else   
            None

    /// Matches a 1D sequence type (seq<_>) or a subclass.
    let rec private  (|SeqTy|_|) (ty: Type) = 
        if ty.IsGenericType && ty.GetGenericTypeDefinition().Equals(typedefof<seq<int>>) then
           Some (ty.GetGenericArguments()[0])
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
            match etys |> Array.tryFind (fun ety -> ety <> etys[0]) with
            | None -> ()
            | Some ety2 -> failwithf "jagged input: unexpected mixed types in tuple being used as sequence notation, %s and %s" (formatType etys[0]) (formatType ety2)
            Some (etys[0])
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
                       yield v[i, j] |]
        arr, [| n1;n2|]

    let private flatArrayAndShape3D<'T> (v: 'T[,,]) =
        let n1 = Array3D.length1 v
        let n2 = Array3D.length2 v
        let n3 = Array3D.length3 v
        let arr =
            [|  for i=0 to n1-1 do
                    for j=0 to n2-1 do
                        for k=0 to n3-1 do
                            yield v[i, j, k] |]
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
                                yield v[i, j, k, m] |]
        arr, [| n1;n2;n3;n4 |]

    let private flatArrayAndShape5D<'T> (v: Array) =
        let n1 = Array5D.length1 v
        let n2 = Array5D.length2 v
        let n3 = Array5D.length3 v
        let n4 = Array5D.length4 v
        let n5 = Array5D.length5 v
        let arr =
            [|  for i1=0 to n1-1 do
                    for i2=0 to n2-1 do
                        for i3=0 to n3-1 do
                            for i4=0 to n4-1 do
                                for i5=0 to n5-1 do
                                    yield Array5D.get v i1 i2 i3 i4 i5 :?> 'T|]
        arr, [| n1;n2;n3;n4;n5 |]

    let private flatArrayAndShape6D<'T> (v: Array) =
        let n1 = Array6D.length1 v
        let n2 = Array6D.length2 v
        let n3 = Array6D.length3 v
        let n4 = Array6D.length4 v
        let n5 = Array6D.length5 v
        let n6 = Array6D.length6 v
        let arr =
            [|  for i1=0 to n1-1 do
                    for i2=0 to n2-1 do
                        for i3=0 to n3-1 do
                            for i4=0 to n4-1 do
                                for i5=0 to n5-1 do
                                    for i6=0 to n6-1 do
                                        yield Array6D.get v i1 i2 i3 i4 i5 i6 :?> 'T|]
        arr, [| n1;n2;n3;n4;n5;n6 |]

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

    // An exact type-match test is needed because of https://github.com/DiffSharp/DiffSharp/issues/203 and https://github.com/dotnet/fsharp/issues/10202
    // That is in .NET and F#, a boxed "byte[]" can be unboxed to "int8[]" and vice-versa.
    // This also affects pattern matches of the element types of sequences as well
    let typesMatch<'T> (array: System.Array) = (array.GetType().GetElementType() = typeof<'T>)

    let rec tryFlatArrayAndShape<'T> (value:obj) : ('T[] * int[]) option =
        match value with
        | :? 'T as v -> Some ([|v|], [||])
        | :? ('T[]) as v when typesMatch<'T> v -> Some (flatArrayAndShape1D v)
        | :? ('T[,]) as v when typesMatch<'T> v -> Some (flatArrayAndShape2D<'T> v)
        | :? ('T[,,]) as v when typesMatch<'T> v -> Some (flatArrayAndShape3D<'T> v)
        | :? ('T[,,,]) as v when typesMatch<'T> v -> Some (flatArrayAndShape4D<'T> v)
        | :? System.Array as v when v.Rank = 5 && typesMatch<'T> v -> Some (flatArrayAndShape5D<'T> v)
        | :? System.Array as v when v.Rank = 6 && typesMatch<'T> v -> Some (flatArrayAndShape6D<'T> v)
        | :? seq<'T> as v when typesMatch<'T> (Seq.toArray v) -> Some (flatArrayAndShape1D (Seq.toArray v))
        | :? seq<seq<'T>> as v when typesMatch<'T> (array2D v) -> Some (flatArrayAndShape2D (array2D v))
        | :? seq<seq<seq<'T>>> as v when typesMatch<'T> (array3D v) -> Some (flatArrayAndShape3D (array3D v))
        | :? seq<seq<seq<seq<'T>>>> as v when typesMatch<'T> (array4D v) -> Some (flatArrayAndShape4D (array4D v))
        | :? seq<seq<seq<seq<seq<'T>>>>> as v when typesMatch<'T> (array5D v) -> Some (flatArrayAndShape5D (array5D v))
        | :? seq<seq<seq<seq<seq<seq<'T>>>>>> as v when typesMatch<'T> (array6D v) -> Some (flatArrayAndShape6D (array6D v))
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
        // ... -> dim 5
        | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleTy (fetcher3, SeqOrSeqTupleTy (fetcher4, SeqOrSeqTupleLeafTy tgt fetcher5)))) -> 
            let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> Array.map (fetcher4 >> Array.map (fetcher5 >> arrayCast<'T>)))) |> array5D
            Some (flatArrayAndShape5D<'T> els)
        // ... -> dim 6
        | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleTy (fetcher3, SeqOrSeqTupleTy (fetcher4, SeqOrSeqTupleTy (fetcher5, SeqOrSeqTupleLeafTy tgt fetcher6))))) -> 
            let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> Array.map (fetcher4 >> Array.map (fetcher5 >> Array.map (fetcher6 >> arrayCast<'T>))))) |> array6D
            Some (flatArrayAndShape6D<'T> els)
        | _ -> None


    [<ExcludeFromCodeCoverage>]
    let inline dataOfValues ofFloat32 ofFloat64 ofInt8 ofInt16 ofInt32 ofInt64 ofBool ofByte (value:obj) : (^T[] * int[]) = 
        match value |> tryFlatArrayAndShape<float32> with
        | Some (values, shape) -> (values |> Array.map ofFloat32, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<double> with
        | Some (values, shape) -> (values |> Array.map ofFloat64, shape) 
        | None -> 
        match value |> tryFlatArrayAndShape<int64> with
        | Some (values, shape) -> (values |> Array.map ofInt64, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<int32> with
        | Some (values, shape) -> (values |> Array.map ofInt32, shape) 
        | None -> 
        match value |> tryFlatArrayAndShape<int16>  with
        | Some (values, shape) -> (values |> Array.map ofInt16, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<bool> with
        | Some (values, shape) -> (values |> Array.map ofBool, shape) 
        | None -> 
        match value |> tryFlatArrayAndShape<byte>  with
        | Some (values, shape) -> (values |> Array.map ofByte, shape)
        | None -> 
        match value |> tryFlatArrayAndShape<int8>  with
        | Some (values, shape) -> (values |> Array.map ofInt8, shape)
        | None -> 
        // Empty tensor (no data, shape: [0])
        match value with
        | :? (seq<obj>) as v when Seq.isEmpty v -> ([||] |> Array.map ofFloat32, [|0|])
        | _ ->
        failwithf "Cannot convert from value of type %A" (value.GetType())

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
        dataOfValues System.Convert.ToBoolean System.Convert.ToBoolean System.Convert.ToBoolean System.Convert.ToBoolean System.Convert.ToBoolean System.Convert.ToBoolean id System.Convert.ToBoolean value 


/// Contains auto-opened utilities related to the DiffSharp programming model.
[<AutoOpen>]
module UtilAutoOpens =

    /// Returns a function that memoizes the given function using a lookaside table.
    let memoize fn =
        let cache = new Dictionary<_,_>()
        fun x ->
            match cache.TryGetValue x with
            | true, v -> v
            | false, _ ->
                let v = fn x
                cache.Add(x,v)
                v

    /// Saves the given value to the given local file using binary serialization.
    let saveBinary (object: 'T) (fileName:string) =
        let formatter = BinaryFormatter()
        let fs = new FileStream(fileName, FileMode.Create)
        let cs = new GZipStream(fs, CompressionMode.Compress)
        try
            formatter.Serialize(cs, object)
            cs.Flush()
            cs.Close()
            fs.Close()
        with
        | :? SerializationException as e -> failwithf "Cannot save to file. %A" e.Message

    /// Loads the given value from the given local file using binary serialization.
    let loadBinary (fileName:string):'T =
        let formatter = BinaryFormatter()
        let fs = new FileStream(fileName, FileMode.Open)
        let cs = new GZipStream(fs, CompressionMode.Decompress)
        try
            let object = formatter.Deserialize(cs) :?> 'T
            cs.Close()
            fs.Close()
            object
        with
        | :? SerializationException as e -> failwithf "Cannot load from file. %A" e.Message

    /// Value of log(sqrt(2*Math.PI)).
    let logSqrt2Pi = log(sqrt(2. * Math.PI))

    /// Value of log(10).
    let log10Val = log 10.

    /// Indents all lines of the given string by the given number of spaces.
    let indentNewLines (str:String) numSpaces =
        let mutable ret = ""
        let spaces = String.replicate numSpaces " "
        str |> Seq.toList |> List.iter (fun c -> 
                            if c = '\n' then 
                                ret <- ret + "\n" + spaces
                            else ret <- ret + string c)
        ret

    /// Left-pads a string up to the given length.
    let stringPad (s:string) (width:int) =
        if s.Length > width then s
        else String.replicate (width - s.Length) " " + s

    /// Left-pads a string to match the length of another string.
    let stringPadAs (s1:string) (s2:string) = stringPad s1 s2.Length

    /// Formats an integer as a string with comma as thousands separator
    let thousandsInt(x:int) = String.Format("{0:#,0}", x)

    /// Formats an integer as a string with comma as thousands separator
    let thousandsFloat(x:float) = String.Format("{0:N}", x)

    /// Returns the file contents as Base64 encoded string
    let fileToBase64String fileName =
        let bytes = System.IO.File.ReadAllBytes(fileName)
        System.Convert.ToBase64String(bytes)

    /// Given a PNG image file name, returns an HTML image element with the image content included as a Base64 encoded string
    let pngToHtml fileName widthPixels =
        sprintf """<img src="data:image/png;base64,%s" style="width: %dpx; height: auto"/>""" (fileName |> fileToBase64String) widthPixels

    /// Return a human-readable string representation of the given value in Bytes.
    let bytesReadable (i:int64) =
        // Based on https://www.somacon.com/p576.php
        let absolute_i = abs i
        let suffix, readable = 
            // https://en.wikipedia.org/wiki/Binary_prefix
            if absolute_i >= 0x1000000000000000L then // exbibyte
                "EiB", (i >>> 50)
            elif absolute_i >= 0x4000000000000L then // pebibyte
                "PiB", (i >>> 40)
            elif absolute_i >= 0x10000000000L then // tebibyte
                "TiB", (i >>> 30)
            elif absolute_i >= 0x40000000L then // gibibyte
                "GiB", (i >>> 20)
            elif absolute_i >= 0x100000L then // mebibyte
                "MiB", (i >>> 10)
            elif absolute_i >= 0x400L then // kibibyte
                "KiB", i
            else
                "B", i // Byte
        if suffix = "B" then i.ToString("0 B") else
        let readable = (double readable / 1024.)
        readable.ToString("0.### ") + suffix

    // Avoids warning FS3370 in F# 6
    let (!) (r: 'T ref)  = r.Value

    // Avoids warning FS3370 in F# 6
    let (:=) (r: 'T ref) (v: 'T)  = r.Value <- v
