// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

open BenchmarkDotNet.Running

[<EntryPoint>]
let main args = 
    let _summary = BenchmarkSwitcher.FromAssembly(System.Reflection.Assembly.GetExecutingAssembly()).Run(args)
    0
