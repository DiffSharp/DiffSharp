namespace DiffSharp

/// Represents a backend for DiffSharp tensors
[<RequireQualifiedAccess>]
type Backend =
    /// The reference backend 
    | Reference
    /// The LibTorch backend 
    | Torch
    /// Reserved for future use
    | Other of name: string * code: int

    member internal x.Code = 
        match x with 
        | Reference -> 0x000
        | Torch -> 0x0100
        | Other (_name, code) -> (code + 3) <<< 8

    /// Get the name of the backend
    member x.Name = 
        match x with 
        | Reference -> "Reference"
        | Torch -> "Torch"
        | Other (name, _) -> name

/// Contains functions and settings related to backend specifications.
module Backend = 
    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Backend>()

    /// Register a new backend
    let Register name = codes.GetOrAdd(name, (fun _ -> incr count; Backend.Other(name, count.Value)))

    /// Get or set the default backend used when creating tensors. Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default = Backend.Reference

