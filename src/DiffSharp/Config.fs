//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
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

namespace DiffSharp.Config

open DiffSharp.Backend

/// Record type holding configuration parameters
type Config =
    {Float32Backend : Backend<float32>
     Float64Backend : Backend<float>
     Float32Epsilon : float32
     Float64Epsilon : float
     Float32EpsilonRec : float32
     Float64EpsilonRec : float
     Float32EpsilonRec2 : float32
     Float64EpsilonRec2 : float
     Float32VisualizationContrast : float32
     Float64VisualizationContrast : float
     GrayscalePalette : string[]}

/// Global configuration
type GlobalConfig() =
    static let GrayscalePaletteUnicode = [|" "; "·"; "-"; "▴"; "▪"; "●"; "♦"; "■"; "█"|]
    static let GrayscalePaletteASCII = [|" "; "."; ":"; "x"; "T"; "Y"; "V"; "X"; "H"; "N"; "M"|]
    static let mutable C =
        let eps = 0.00001
        {Float32Backend = OpenBLAS.Float32Backend()
         Float64Backend = OpenBLAS.Float64Backend()
         Float32Epsilon = (float32 eps)
         Float64Epsilon = eps
         Float32EpsilonRec = 1.f / (float32 eps)
         Float64EpsilonRec = 1. / eps
         Float32EpsilonRec2 = 0.5f / (float32 eps)
         Float64EpsilonRec2 = 0.5 / eps
         Float32VisualizationContrast = 1.2f
         Float64VisualizationContrast = 1.2
         GrayscalePalette = GrayscalePaletteUnicode}

    static member Float32Backend = C.Float32Backend
    static member Float64Backend = C.Float64Backend
    static member Float32Epsilon = C.Float32Epsilon
    static member Float64Epsilon = C.Float64Epsilon
    static member Float32EpsilonRec = C.Float32EpsilonRec
    static member Float64EpsilonRec = C.Float64EpsilonRec
    static member Float32EpsilonRec2 = C.Float32EpsilonRec2
    static member Float64EpsilonRec2 = C.Float64EpsilonRec2
    static member Float32VisualizationContrast = C.Float32VisualizationContrast
    static member Float64VisualizationContrast = C.Float64VisualizationContrast
    static member GrayscalePalette = C.GrayscalePalette
    static member SetBackend(backend:string) =
        match backend with
        | "OpenBLAS" ->
            C <- {C with
                    Float32Backend = OpenBLAS.Float32Backend()
                    Float64Backend = OpenBLAS.Float64Backend()}
        | _ -> invalidArg "" "Unsupported backend. Try: OpenBLAS"
    static member SetEpsilon(e:float32) = 
        C <- {C with
                Float32Epsilon = e
                Float64Epsilon = float e
                Float32EpsilonRec = 1.f / e
                Float64EpsilonRec = 1. / (float e)
                Float32EpsilonRec2 = 0.5f / e
                Float64EpsilonRec2 = 0.5 / (float e)}
    static member SetEpsilon(e:float) = 
        C <- {C with
                Float32Epsilon = float32 e
                Float64Epsilon = e;
                Float32EpsilonRec = 1.f / (float32 e)
                Float64EpsilonRec = 1. / e
                Float32EpsilonRec2 = 0.5f / (float32 e)
                Float64EpsilonRec2 = 0.5 / e}
    static member SetVisualizationContrast(c:float32) =
        C <- {C with
                Float32VisualizationContrast = c
                Float64VisualizationContrast = float c}
    static member SetVisualizationContrast(c:float) =
        C <- {C with
                Float32VisualizationContrast = float32 c
                Float64VisualizationContrast = c}
    static member SetVisualizationPalette(palette:string) =
        match palette with
        | "ASCII" ->
            C <- {C with
                    GrayscalePalette = GrayscalePaletteASCII}
        | "Unicode" ->
            C <- {C with
                    GrayscalePalette = GrayscalePaletteUnicode}
        | _ -> invalidArg "" "Unsupported palette. Try: ASCII or Unicode"