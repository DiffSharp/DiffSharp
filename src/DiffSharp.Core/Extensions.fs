// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Util

open System
open System.Collections.Generic
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
    let inline cumulativeSum (a:_[]) = (Array.scan (+) LanguagePrimitives.GenericZero a).[1..]

    /// Gets the unique counts of the input array.
    let getUniqueCounts (sorted:bool) (values:'T[]) =
        let counts = Dictionary<'T, int>()
        for v in values do
            if counts.ContainsKey(v) then counts.[v] <- counts.[v] + 1 else counts.[v] <- 1
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
        while i < n && f i arr.[i] do
            i <- i + 1
        (i = n)

module Array4D =
    /// Builds a new array whose elements are the results of applying the given function to each of the elements of the array.
    let map mapping (array:'a[,,,]) =
        Array4D.init (array.GetLength(0)) (array.GetLength(1)) (array.GetLength(2)) (array.GetLength(3)) (fun i j k l -> mapping array.[i, j, k, l])

module ArrayND = 
    /// Initializes an array with a given shape and initializer function.
    let init (shape: int[]) (f: int[] -> 'T) : obj =
        match shape with 
        | [| |] -> f [| |]  :> _
        | [| d0 |] -> Array.init d0 (fun i -> f [| i |]) :> _
        | [| d0; d1 |] -> Array2D.init d0 d1 (fun i1 i2 -> f [| i1; i2 |]) :> _
        | [| d0; d1; d2 |] -> Array3D.init d0 d1 d2 (fun i1 i2 i3 -> f [| i1; i2; i3 |]) :> _
        | [| d0; d1; d2; d3 |] -> Array4D.init d0 d1 d2 d3 (fun i1 i2 i3 i4 -> f [| i1; i2; i3; i4 |]) :> _
        | _ -> failwith "ArrayND.init - nyi for dim > 4"

    /// Initializes an array with a given shape and initializer function.
    let zeroCreate (shape: int[]) : Array =
        match shape with 
        | [| |] -> [| |] :> _
        | [| d0 |] -> Array.zeroCreate d0 :> _
        | [| d0; d1 |] -> Array2D.zeroCreate d0 d1 :> _
        | [| d0; d1; d2 |] -> Array3D.zeroCreate d0 d1 d2 :> _
        | [| d0; d1; d2; d3 |] -> Array4D.zeroCreate d0 d1 d2 d3 :> _
        | _ -> failwith "ArrayND.zeroCreate - nyi for dim > 4"

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

/// Contains extensions related to .NET dictionaries. 
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
        let r1, r2, r3 = data.Length, data.[0].GetLength(0), data.[0].GetLength(1)
        for i in 0 .. r1-1 do 
            let q2 = data.[i].GetLength(0)
            let q3 = data.[i].GetLength(1)
            if q2 <> r2 || q3 <> r3 then 
                invalidArg "data" (sprintf "jagged input at position %d: first is _ x %d x %d, later is _ x _ x %d x %d" i r2 r3 q2 q3)
        Array3D.init r1 r2 r3 (fun i j k -> data.[i].[j,k])

    /// Creates a non-jagged 4D array from jagged data.
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

    /// Print the given value to the console using the '%A' printf format specifier
    let print x = printfn "%A" x 


[<assembly: System.Runtime.CompilerServices.InternalsVisibleTo("DiffSharp.Tests")>]
do()
