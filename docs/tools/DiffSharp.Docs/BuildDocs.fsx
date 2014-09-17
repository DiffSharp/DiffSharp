//
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright 2014 National University of Ireland Maynooth.
// All rights reserved.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Script for generating library documentation
//


#I "../../../packages/FSharp.Compiler.Service.0.0.59/lib/net40/"
#r "FSharp.Compiler.Service.dll"
#I "../../../packages/RazorEngine.3.3.0/lib/net40/"
#r "RazorEngine.dll"
#I "../../../packages/Microsoft.AspNet.Razor.2.0.30506.0/lib/net40/"
#r "System.Web.Razor.dll"
#I "../../../packages/FSharp.Formatting.2.4.21/lib/net40/"
#r "FSharp.MetadataFormat.dll"
#r "FSharp.CodeFormat.dll"
#r "FSharp.Literate.dll"
#r "FSharp.Markdown.dll"
#r "CSharpFormat.dll"

open System.IO
open FSharp.Literate
open FSharp.MetadataFormat

//
// Setup output directory structure and copy static files
//

let source = __SOURCE_DIRECTORY__ 
let docs = Path.Combine(source, "../../")
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
let tags = ["project-name", "DiffSharp"; "project-author", "Atılım Güneş Baydin"; "project-github", "http://github.com/gbaydin/DiffSharp"; "project-nuget", ""; "root", ""]

Literate.ProcessScriptFile(relative "input/index.fsx", relative "input/templates/template.html", relative "output/index.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/api-overview.fsx", relative "input/templates/template.html", relative "output/api-overview.html", replacements = tags)
Literate.ProcessScriptFile(relative "input/howto-typeinference.fsx", relative "input/templates/template.html", relative "output/howto-typeinference.html", replacements = tags)

//
// Generate API reference
//

let library = relative "../src/DiffSharp/bin/Debug/DiffSharp.dll"
let layoutRoots = [relative "input/templates"; relative "input/templates/reference" ]

MetadataFormat.Generate(library, relative "output/reference", layoutRoots, tags, markDownComments = true)
