// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Util

open System
open System.Collections.Generic
open System.Collections.Specialized
open System.Diagnostics.CodeAnalysis


/// <summary>
///   Contains extensions to the F# Array module. 
/// </summary>
///
/// <namespacedoc>
///   <summary>Contains utilities and library extensions related to the DiffSharp programming model.</summary>
/// </namespacedoc>
module Array =

    /// Determines if all values of the first array lie within the given tolerances of the second array.
    [<ExcludeFromCodeCoverage>]
    let inline allClose (relativeTolerance:'T) (absoluteTolerance:'T) (array1:'T[]) (array2:'T[]) =
        let dim1 = array1.Length
        let dim2 = array2.Length
        if dim1 <> dim2 then false
        else (array1,array2) ||> Array.forall2 (fun a b -> abs(a-b) <= absoluteTolerance + relativeTolerance*abs(b)) 

    /// Gets the cumulative sum of the input array.
    [<ExcludeFromCodeCoverage>]
    let inline cumulativeSum (a:_[]) = (Array.scan (+) LanguagePrimitives.GenericZero a)[1..]

    /// Gets the unique counts of the input array.
    let getUniqueCounts (sorted:bool) (values:'T[]) =
        let counts = Dictionary<'T, int>()
        for v in values do
            if counts.ContainsKey(v) then counts[v] <- counts[v] + 1 else counts[v] <- 1
        if sorted then
            counts |> Array.ofSeq |> Array.sortByDescending (fun (KeyValue(_, v)) -> v) |> Array.map (fun (KeyValue(k, v)) -> k, v) |> Array.unzip
        else
            counts |> Array.ofSeq |> Array.map (fun (KeyValue(k, v)) -> k, v) |> Array.unzip

    // Create a 2D array using a flat representation
    let initFlat2D i j f = Array.init (i*j) (fun ij -> f (ij/j) (ij%j))

    // Create a 3D array using a flat representation
    let initFlat3D i j k f = Array.init (i*j*k) (fun ijk -> f (ijk/j/k) ((ijk/k)%j) (ijk%k))

    let foralli f (arr: 'T[]) =
        let mutable i = 0
        let n = arr.Length
        while i < n && f i arr[i] do
            i <- i + 1
        (i = n)

    // Copied from https://github.com/dotnet/fsharp/pull/11888 contributed by Jan Dryk (uxsoft)
    let insertManyAt (index: int) (values: seq<'T>) (source: 'T[]) : 'T[] =
        if index < 0 || index > source.Length then invalidArg "index" "index must be within bounds of the array"

        let valuesArray = Seq.toArray values
        if valuesArray.Length = 0 then source
        else
            let length = source.Length + valuesArray.Length
            let result = Array.zeroCreate length
            if index > 0 then
                Array.Copy(source, result, index)
            Array.Copy(valuesArray, 0, result, index, valuesArray.Length)
            if source.Length - index > 0 then
                Array.Copy(source, index, result, index + valuesArray.Length, source.Length - index)
            result

    // Copied from https://github.com/dotnet/fsharp/pull/11888 contributed by Jan Dryk (uxsoft)
    let removeAt (index: int) (source: 'T[]) : 'T[] =
        if index < 0 || index >= source.Length then invalidArg "index" "index must be within bounds of the array"
        let length = source.Length - 1
        let result = Array.zeroCreate length
        if index > 0 then 
            Array.Copy(source, result, index)
        if length - index > 0 then
            Array.Copy(source, index + 1, result, index, length - index)
        result

module Array4D =
    /// Builds a new array whose elements are the results of applying the given function to each of the elements of the array.
    let map mapping (array:'a[,,,]) =
        Array4D.init (array.GetLength(0)) (array.GetLength(1)) (array.GetLength(2)) (array.GetLength(3)) (fun i j k l -> mapping array[i, j, k, l])

// See https://github.com/dotnet/fsharp/issues/12013
//type 'T array5d = 'T ``[,,,,]``
//type 'T array6d = 'T ``[,,,,,]``

module Array5D =
    /// <summary></summary> <exclude />
    let zeroCreate<'T> (length1:int) length2 length3 length4 length5 : Array =
        System.Array.CreateInstance(typeof<'T>, [|length1;length2;length3;length4;length5|])

    let get (array:Array) (index1:int) index2 index3 index4 index5 =
        array.GetValue([|index1;index2;index3;index4;index5|])

    let set (array:Array) (index1:int) index2 index3 index4 index5 value =
        array.SetValue(value, [|index1;index2;index3;index4;index5|])
   
    let length1 (array: Array) = array.GetLength(0)
    let length2 (array: Array) = array.GetLength(1)
    let length3 (array: Array) = array.GetLength(2)
    let length4 (array: Array) = array.GetLength(3)
    let length5 (array: Array) = array.GetLength(4)

    let init<'T> (length1:int) length2 length3 length4 length5 (initializer:int->int->int->int->int->'T) : Array =
        let arr = zeroCreate<'T> length1 length2 length3 length4 length5
        for i1=0 to length1-1 do
            for i2=0 to length2-1 do
                for i3=0 to length3-1 do
                    for i4=0 to length4-1 do
                        for i5=0 to length5-1 do
                            set arr i1 i2 i3 i4 i5 (initializer i1 i2 i3 i4 i5)
        arr

    let create (length1:int) length2 length3 length4 length5 (initial:'T) = init length1 length2 length3 length4 length5 (fun _ _ _ _ _ -> initial)

    let map mapping (array: Array) =
        init (length1 array) (length2 array) (length3 array) (length4 array) (length5 array) (fun i1 i2 i3 i4 i5 -> mapping (get array i1 i2 i3 i4 i5))

module Array6D =
    let zeroCreate<'T> (length1:int) length2 length3 length4 length5 length6 : Array =
        System.Array.CreateInstance(typeof<'T>, [|length1;length2;length3;length4;length5;length6|])

    let get (array: Array) (index1: int) index2 index3 index4 index5 index6 =
        array.GetValue([|index1;index2;index3;index4;index5;index6|])

    let set (array: Array) (index1: int) index2 index3 index4 index5 index6 value =
        array.SetValue(value, [|index1;index2;index3;index4;index5;index6|])

    let length1 (array: Array) = array.GetLength(0)
    let length2 (array: Array) = array.GetLength(1)
    let length3 (array: Array) = array.GetLength(2)
    let length4 (array: Array) = array.GetLength(3)
    let length5 (array: Array) = array.GetLength(4)
    let length6 (array: Array) = array.GetLength(5)

    let init<'T> (length1: int) length2 length3 length4 length5 length6 (initializer: int->int->int->int->int->int->'T) =
        let arr = zeroCreate<'T> length1 length2 length3 length4 length5 length6
        for i1=0 to length1-1 do
            for i2=0 to length2-1 do
                for i3=0 to length3-1 do
                    for i4=0 to length4-1 do
                        for i5=0 to length5-1 do
                            for i6=0 to length6-1 do
                                set arr i1 i2 i3 i4 i5 i6 (initializer i1 i2 i3 i4 i5 i6)
        arr

    let create (length1: int) length2 length3 length4 length5 length6 (initial:'T) =
        init length1 length2 length3 length4 length5 length6 (fun _ _ _ _ _ _ -> initial)

    let map mapping (array: Array) =
        init (length1 array) (length2 array) (length3 array) (length4 array) (length5 array) (length6 array) (fun i1 i2 i3 i4 i5 i6 -> mapping (get array i1 i2 i3 i4 i5 i6))


// Notes about slicing 5d and 6d arrays if needed
// #if SLICING
// [<AutoOpen>]
// module Array5DExtensions =
//     type ``[,,,,]``<'T> with
//         member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) : ``[,,,,]``<'T> =
//             failwith "tbd"
//         member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) : 'T[,,,] =
//             failwith "tbd"
//
// let d = Array5D.zeroCreate<int> 2 2 2 2 2
// d[0..0,0..0,0..0,0..0,0..0]
// d[0,0..0,0..0,0..0,0..0]
// #endif


module ArrayND = 
    /// Initializes an array with a given shape and initializer function.
    let init (shape: int[]) (f: int[] -> 'T) : obj =
        match shape with 
        | [| |] -> f [| |]  :> _
        | [| d1 |] -> Array.init d1 (fun i -> f [| i |]) :> _
        | [| d1; d2 |] -> Array2D.init d1 d2 (fun i1 i2 -> f [| i1; i2 |]) :> _
        | [| d1; d2; d3 |] -> Array3D.init d1 d2 d3 (fun i1 i2 i3 -> f [| i1; i2; i3 |]) :> _
        | [| d1; d2; d3; d4 |] -> Array4D.init d1 d2 d3 d4 (fun i1 i2 i3 i4 -> f [| i1; i2; i3; i4 |]) :> _
        | [| d1; d2; d3; d4; d5 |] -> Array5D.init d1 d2 d3 d4 d5 (fun i1 i2 i3 i4 i5 -> f [| i1; i2; i3; i4; i5 |]) :> _
        | [| d1; d2; d3; d4; d5; d6 |] -> Array6D.init d1 d2 d3 d4 d5 d6 (fun i1 i2 i3 i4 i5 i6 -> f [| i1; i2; i3; i4; i5; i6 |]) :> _
        | _ -> failwith "ArrayND.init not supported for dim > 6"

    /// Initializes an array with a given shape and initializer function.
    let zeroCreate (shape: int[]) : Array =
        match shape with 
        | [| |] -> [| |] :> _
        | [| d1 |] -> Array.zeroCreate d1 :> _
        | [| d1; d2 |] -> Array2D.zeroCreate d1 d2 :> _
        | [| d1; d2; d3 |] -> Array3D.zeroCreate d1 d2 d3 :> _
        | [| d1; d2; d3; d4 |] -> Array4D.zeroCreate d1 d2 d3 d4 :> _
        | [| d1; d2; d3; d4; d5 |] -> Array5D.zeroCreate d1 d2 d3 d4 d5
        | [| d1; d2; d3; d4; d5; d6 |] -> Array6D.zeroCreate d1 d2 d3 d4 d5 d6
        | _ -> failwith "ArrayND.zeroCreate not supported for dim > 6"

/// Contains extensions to the F# Seq module. 
module Seq =

    /// Gets the index of the maximum element of the sequence.
    let maxIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.maxBy snd |> fst

    /// Gets the index of the minimum element of the sequence.
    let minIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.minBy snd |> fst

    /// Indicates if all elements of the sequence are equal.
    let allEqual (items:seq<'T>) =
        let item0 = items |> Seq.head
        items |> Seq.forall ((=) item0)

    /// Gets the duplicate elements in the sequence.
    let duplicates l =
       l |> List.ofSeq
       |> List.groupBy id
       |> List.choose ( function
              | _, x::_::_ -> Some x
              | _ -> None )

    /// Indicates if a sequence has duplicate elements.
    let hasDuplicates l =
        duplicates l |> List.isEmpty |> not

    /// Like Seq.toArray but does not clone the array if the input is already an array
    let inline toArrayQuick (xs: seq<'T>) =
        match xs with
        | :? ('T[]) as arr -> arr
        | _ -> Seq.toArray xs

/// Contains extensions related to .NET OrderedDictionary. 
module OrderedDictionary =

    /// Gets a fresh array containing the keys of the dictionary.
    let copyKeys (dictionary:OrderedDictionary) =
        let keys = Array.zeroCreate dictionary.Count
        dictionary.Keys.CopyTo(keys, 0)
        keys

/// Contains extensions related to .NET Dictionary. 
module Dictionary =

    /// Gets a fresh array containing the keys of the dictionary.
    let copyKeys (dictionary:Dictionary<'Key, 'Value>) =
        let keys = Array.zeroCreate dictionary.Count
        dictionary.Keys.CopyTo(keys, 0)
        keys

    /// Gets a fresh array containing the values of the dictionary.
    let copyValues (dictionary:Dictionary<'Key, 'Value>) =
        let values = Array.zeroCreate dictionary.Count
        dictionary.Values.CopyTo(values, 0)
        values

/// Contains auto-opened extensions to the F# programming model.
[<AutoOpen>]
module ExtensionAutoOpens =

    /// Indicates if a value is not null.
    [<ExcludeFromCodeCoverage>]
    let inline notNull value = not (obj.ReferenceEquals(value, null))

    /// Creates a non-jagged 3D array from jagged data.
    let array3D data = 
        let data = data |> Array.ofSeq |> Array.map array2D
        let r1, r2, r3 = data.Length, data[0].GetLength(0), data[0].GetLength(1)
        for i in 0 .. r1-1 do 
            let q2 = data[i].GetLength(0)
            let q3 = data[i].GetLength(1)
            if q2 <> r2 || q3 <> r3 then 
                invalidArg "data" (sprintf "jagged input at position %d: first is _ x %d x %d, later is _ x %d x %d" i r2 r3 q2 q3)
        Array3D.init r1 r2 r3 (fun i j k -> data[i][j,k])

    /// Creates a non-jagged 4D array from jagged data.
    let array4D data = 
        let data = data |> array2D |> Array2D.map array2D
        let r1,r2,r3,r4 = data.GetLength(0), data.GetLength(1), data[0,0].GetLength(0), data[0,0].GetLength(1)
        for i in 0 .. r1-1 do 
          for j in 0 .. r2-1 do 
            let q3 = data[i,j].GetLength(0)
            let q4 = data[i,j].GetLength(1)
            if q3 <> r3 || q4 <> r4 then 
                invalidArg "data" (sprintf "jagged input at position (%d,%d): first is _ x _ x %d x %d, later is _ x _ x %d x %d" i j r2 r3 q3 q4)
        Array4D.init r1 r2 r3 r4 (fun i j k m -> data[i,j][k,m])

    let array5D data =
        let data = data |> Array.ofSeq |> Array.map array4D
        let r1,r2,r3,r4,r5 = data.Length, data[0].GetLength(0), data[0].GetLength(1), data[0].GetLength(2), data[0].GetLength(3)
        for i in 0 .. r1-1 do
            let q2 = data[i].GetLength(0)
            let q3 = data[i].GetLength(1)
            let q4 = data[i].GetLength(2)
            let q5 = data[i].GetLength(3)
            if q2 <> r2 || q3 <> r3 || q4 <> r4 || q5 <> r5 then
                invalidArg "data" (sprintf "jagged input at position %d: first is _ x %d x %d x %d x %d, later is _ x %d x %d x %d x %d" i r2 r3 r4 r5 q2 q3 q4 q5)
        Array5D.init r1 r2 r3 r4 r5 (fun i1 i2 i3 i4 i5 -> data[i1][i2,i3,i4,i5])

    let array6D data =
        let data = data |> array2D |> Array2D.map array4D
        let r1,r2,r3,r4,r5,r6 = data.GetLength(0), data.GetLength(1), data[0,0].GetLength(0), data[0,0].GetLength(1), data[0,0].GetLength(2), data[0,0].GetLength(3)
        for i in 0 .. r1-1 do
            for j in 0 .. r2-2 do
                let q3 = data[i,j].GetLength(0)
                let q4 = data[i,j].GetLength(1)
                let q5 = data[i,j].GetLength(2)
                let q6 = data[i,j].GetLength(3)
                if q3 <> r3 || q4 <> r4 || q5 <> r5 || q6 <> r6 then
                    invalidArg "data" (sprintf "jagged input at position (%d,%d): first is _ x _ x %d x %d x %d x %d, later is _ x _ x %d x %d x %d x %d" i j r3 r4 r5 r6 q3 q4 q5 q6)
        Array6D.init r1 r2 r3 r4 r5 r6 (fun i1 i2 i3 i4 i5 i6 -> data[i1,i2][i3,i4,i5,i6])

    /// Print the given value to the console using the '%A' printf format specifier
    let print x = printfn "%A" x 


[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("DiffSharp.Tests")>]
do()
