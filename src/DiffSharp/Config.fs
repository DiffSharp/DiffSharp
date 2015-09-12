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

open DiffSharp.BackEnd

type Config =
    {Float32BackEnd : BackEnd<float32>
     Float64BackEnd : BackEnd<float>
     Float32Epsilon : float32
     Float64Epsilon : float
     Float32EpsilonRec : float32
     Float64EpsilonRec : float
     Float32EpsilonRec2 : float32
     Float64EpsilonRec2 : float}


type GlobalConfig() =
    static let mutable C =
        let eps = 0.00001
        {Float32BackEnd = OpenBLAS.Float32BackEnd(); 
         Float64BackEnd = OpenBLAS.Float64BackEnd(); 
         Float32Epsilon = (float32 eps);
         Float64Epsilon = eps;
         Float32EpsilonRec = 1.f / (float32 eps);
         Float64EpsilonRec = 1. / eps
         Float32EpsilonRec2 = 0.5f / (float32 eps);
         Float64EpsilonRec2 = 0.5 / eps}

    static member Float32BackEnd = C.Float32BackEnd
    static member Float64BackEnd = C.Float64BackEnd
    static member Float32Epsilon = C.Float32Epsilon
    static member Float64Epsilon = C.Float64Epsilon
    static member Float32EpsilonRec = C.Float32EpsilonRec
    static member Float64EpsilonRec = C.Float64EpsilonRec
    static member Float32EpsilonRec2 = C.Float32EpsilonRec2
    static member Float64EpsilonRec2 = C.Float64EpsilonRec2
    static member SetBackEnd(backend:string) =
        match backend with
        | "OpenBLAS" ->
            C <- {Float32BackEnd = OpenBLAS.Float32BackEnd(); 
                  Float64BackEnd = OpenBLAS.Float64BackEnd(); 
                  Float32Epsilon = C.Float32Epsilon;
                  Float64Epsilon = C.Float64Epsilon;
                  Float32EpsilonRec = C.Float32EpsilonRec;
                  Float64EpsilonRec = C.Float64EpsilonRec;
                  Float32EpsilonRec2 = C.Float32EpsilonRec2;
                  Float64EpsilonRec2 = C.Float64EpsilonRec2}
        | _ -> invalidArg "" "Unsupported back end."
    static member SetEpsilon(e:float32) = 
        C <- {Float32BackEnd = C.Float32BackEnd; 
              Float64BackEnd = C.Float64BackEnd; 
              Float32Epsilon = e;
              Float64Epsilon = (float e);
              Float32EpsilonRec = 1.f / e;
              Float64EpsilonRec = 1. / (float e);
              Float32EpsilonRec2 = 0.5f / e;
              Float64EpsilonRec2 = 0.5 / (float e)}
    static member SetEpsilon(e:float) = 
        C <- {Float32BackEnd = C.Float32BackEnd; 
              Float64BackEnd = C.Float64BackEnd; 
              Float32Epsilon = (float32 e);
              Float64Epsilon = e;
              Float32EpsilonRec = 1.f / (float32 e);
              Float64EpsilonRec = 1. / e;
              Float32EpsilonRec2 = 0.5f / (float32 e);
              Float64EpsilonRec2 = 0.5 / e}
