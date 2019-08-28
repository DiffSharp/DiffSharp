// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

//
// Script for generating library documentation
//

#r "../packages/FSharp.Compiler.Service/lib/net45/FSharp.Compiler.Service.dll"
#r "../packages/FSharp.Formatting/lib/net40/FSharp.Markdown.dll"
#r "../packages/FSharpVSPowerTools.Core/lib/net45/FSharpVSPowerTools.Core.dll"
#r "../packages/FSharp.Formatting/lib/net40/RazorEngine.dll"
#r "../packages/FSharp.Formatting/lib/net40/CSharpFormat.dll"
#r "../packages/FSharp.Formatting/lib/net40/FSharp.CodeFormat.dll"
#r "../packages/FSharp.Formatting/lib/net40/FSharp.Literate.dll"
#r "../packages/FSharp.Formatting/lib/net40/FSharp.MetadataFormat.dll"


open System.IO
open FSharp.Literate
open FSharp.MetadataFormat

//
// Setup output directory structure and copy static files
//

let source = __SOURCE_DIRECTORY__
let docs = Path.Combine(source, "")
let relative subdir = Path.Combine(docs, subdir)

let test = Directory.Exists (relative "output")

if not (Directory.Exists(relative "output")) then
    Directory.CreateDirectory(relative "output") |> ignore
if not (Directory.Exists(relative "output/img")) then
    Directory.CreateDirectory (relative "output/img") |> ignore
if not (Directory.Exists(relative "output/misc")) then
    Directory.CreateDirectory (relative "output/misc") |> ignore
if not (Directory.Exists(relative "output/reference")) then
    Directory.CreateDirectory (relative "output/reference") |> ignore

for fileInfo in DirectoryInfo(relative "input/files/misc").EnumerateFiles() do
    fileInfo.CopyTo(Path.Combine(relative "output/misc", fileInfo.Name), true) |> ignore

for fileInfo in DirectoryInfo(relative "input/files/img").EnumerateFiles() do
    fileInfo.CopyTo(Path.Combine(relative "output/img", fileInfo.Name), true) |> ignore


//
// Generate documentation
//

let tags = ["project-name", "DiffSharp"; "project-author", "Atılım Güneş Baydin"; "project-github", "https://github.com/DiffSharp/DiffSharp"; "project-nuget", "https://www.nuget.org/packages/diffsharp"; "root", ""]

Literate.ProcessScriptFile(relative "input/index.fsx", relative "input/templates/template.html", relative "output/index.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/download.fsx", relative "input/templates/template.html", relative "output/download.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/api-overview.fsx", relative "input/templates/template.html", relative "output/api-overview.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/gettingstarted-typeinference.fsx", relative "input/templates/template.html", relative "output/gettingstarted-typeinference.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/gettingstarted-nestedad.fsx", relative "input/templates/template.html", relative "output/gettingstarted-nestedad.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/gettingstarted-symbolicdifferentiation.fsx", relative "input/templates/template.html", relative "output/gettingstarted-symbolicdifferentiation.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/gettingstarted-numericaldifferentiation.fsx", relative "input/templates/template.html", relative "output/gettingstarted-numericaldifferentiation.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/benchmarks.fsx", relative "input/templates/template.html", relative "output/benchmarks.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/csharp.fsx", relative "input/templates/template.html", relative "output/csharp.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/examples-gradientdescent.fsx", relative "input/templates/template.html", relative "output/examples-gradientdescent.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/examples-inversekinematics.fsx", relative "input/templates/template.html", relative "output/examples-inversekinematics.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/examples-hamiltonianmontecarlo.fsx", relative "input/templates/template.html", relative "output/examples-hamiltonianmontecarlo.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/examples-helmholtzenergyfunction.fsx", relative "input/templates/template.html", relative "output/examples-helmholtzenergyfunction.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/examples-kinematics.fsx", relative "input/templates/template.html", relative "output/examples-kinematics.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/examples-kmeansclustering.fsx", relative "input/templates/template.html", relative "output/examples-kmeansclustering.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/examples-lhopitalsrule.fsx", relative "input/templates/template.html", relative "output/examples-lhopitalsrule.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/examples-neuralnetworks.fsx", relative "input/templates/template.html", relative "output/examples-neuralnetworks.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/examples-newtonsmethod.fsx", relative "input/templates/template.html", relative "output/examples-newtonsmethod.html", replacements = tags, fsiEvaluator = FsiEvaluator())
Literate.ProcessScriptFile(relative "input/examples-stochasticgradientdescent.fsx", relative "input/templates/template.html", relative "output/examples-stochasticgradientdescent.html", replacements = tags, fsiEvaluator = FsiEvaluator())

//
// Generate API reference
//

let library = relative "../src/DiffSharp/bin/Debug/DiffSharp.dll"
let layoutRoots = [relative "input/templates"; relative "input/templates/reference" ]

MetadataFormat.Generate(library, relative "output/reference", layoutRoots, tags, markDownComments = true)
