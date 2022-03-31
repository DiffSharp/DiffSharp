// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model


type Sequential(models: seq<Model>) =
    inherit Model()
    do base.addModel(models |> Seq.toArray)
    override _.forward(value) = 
        models |> Seq.fold (fun v m -> m.forward v) value
    override m.ToString() = sprintf "Sequential(%A)" m.children