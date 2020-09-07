namespace DiffSharp.Util

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
    [<ExcludeFromCodeCoverage>]
    let inline allClose (relativeTolerance:'T) (absoluteTolerance:'T) (array1:'T[]) (array2:'T[]) =
        let dim1 = array1.Length
        let dim2 = array2.Length
        if dim1 <> dim2 then false
        else (array1,array2) ||> Array.forall2 (fun a b -> abs(a-b) <= absoluteTolerance + relativeTolerance*abs(b)) 

    [<ExcludeFromCodeCoverage>]
    let inline cumulativeSum (a:_[]) = (Array.scan (+) LanguagePrimitives.GenericZero a).[1..]

/// Contains extensions to the F# Seq module. 
module Seq =
    let maxIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.maxBy snd |> fst

    let minIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.minBy snd |> fst

    let allEqual (items:seq<'a>) =
        let item0 = items |> Seq.head
        items |> Seq.forall ((=) item0)

    let duplicates l =
       l |> List.ofSeq
       |> List.groupBy id
       |> List.choose ( function
              | _, x::_::_ -> Some x
              | _ -> None )

    let hasDuplicates l =
        duplicates l |> List.isEmpty |> not

/// Contains extensions related to .NET dictionaries. 
module Dictionary =
    let copyKeys (dictionary:Dictionary<'a, 'b>) =
        let keys = Array.zeroCreate dictionary.Count
        dictionary.Keys.CopyTo(keys, 0)
        keys

    let copyValues (dictionary:Dictionary<'a, 'b>) =
        let values = Array.zeroCreate dictionary.Count
        dictionary.Values.CopyTo(values, 0)
        values


/// Contains auto-opened extensions to the F# programming model
[<AutoOpen>]
module ExtensionAutoOpens =
    [<ExcludeFromCodeCoverage>]
    let inline notNull value = not (obj.ReferenceEquals(value, null))

    let memoize fn =
        let cache = new Dictionary<_,_>()
        fun x ->
            match cache.TryGetValue x with
            | true, v -> v
            | false, _ ->
                let v = fn x
                cache.Add(x,v)
                v

