//
// This file is part of
// DiffSharp: Differentiable Functional Programming
//
// Copyright (c) 2014--2016, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under the LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Script for generating library documentation
//

#I "../packages/FSharp.Compiler.Service.0.0.90/lib/net45/"
#r "FSharp.Compiler.Service.dll"
#I "../packages/FSharpVSPowerTools.Core.1.9.0/lib/net45"
#r "FSharpVSPowerTools.Core.dll"
#I "../packages/FSharp.Formatting.2.10.3/lib/net40/"
#r "CSharpFormat.dll"
#r "FSharp.CodeFormat.dll"
#r "FSharp.Literate.dll"
#r "FSharp.MetadataFormat.dll"
#r "FSharp.Markdown.dll"


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

let tags = ["project-name", "DiffSharp"; "project-author", "Atılım Güneş Baydin"; "project-github", "http://github.com/DiffSharp/DiffSharp"; "project-nuget", "https://www.nuget.org/packages/diffsharp"; "root", ""]

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
