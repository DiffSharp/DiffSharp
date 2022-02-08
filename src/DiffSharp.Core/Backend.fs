// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

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

    override x.ToString() = x.Name

/// Contains functions and settings related to backend specifications.
module Backend = 
    let mutable internal count = 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Backend>()

    /// Register a new backend
    let Register name =
        codes.GetOrAdd(name, (fun _ ->
            count <- count + 1
            Backend.Other(name, count)))

    /// Get or set the default backend used when creating tensors. Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default = Backend.Reference

type BackendFunctionality<'T>() =
    let mutable last = None
    let backends = System.Collections.Concurrent.ConcurrentDictionary<int, 'T>()

    member _.Get(?backend: Backend) =
        let backend = defaultArg backend Backend.Default
        let code = backend.Code
        match last with 
        | Some (code2, v) when code = code2 -> v
        | _ ->
        match backends.TryGetValue(code) with 
        | true, v -> v
        | false, _ -> 
            let res =
                backends.GetOrAdd(code, fun _ -> 
                    let name = "DiffSharp.Backends." + backend.Name
                    let fullName = System.Reflection.Assembly.GetExecutingAssembly().FullName.Replace("DiffSharp.Core", name)
                    let asm = 
                        try System.Reflection.Assembly.Load(fullName)
                        with e ->  failwithf "Couldn't find assembly '%s', error = %s" fullName (e.ToString())
                    let typeName = sprintf "DiffSharp.Backends.%s.%s%s" backend.Name backend.Name typeof<'T>.Name
                    let theType = asm.GetType(typeName)
                    if isNull theType then failwithf "Couldn't find type '%s' in assembly '%s'" typeName fullName
                    let b = 
                        match System.Activator.CreateInstance(theType) with
                        | :? 'T as b -> b
                        | _ -> failwith "activation failed to return correct type"
                    b
                    ) 
            last <- Some (code, res)
            res

    member _.Backends = backends
