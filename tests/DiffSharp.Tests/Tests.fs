// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

namespace DiffSharp.Tests

open DiffSharp.Util
open FsCheck


type Util() =
    static let eps64 = 1e-4
    static let eps32 = float32 eps64

    static member IsNice(a:float32) = not (System.Single.IsInfinity(a) || System.Single.IsNaN(a) || (a = System.Single.MinValue) || (a = System.Single.MaxValue))
    static member IsNice(a:float) = not (System.Double.IsInfinity(a) || System.Double.IsNaN(a) || (a = System.Double.MinValue) || (a = System.Double.MaxValue))
    static member IsNice(a:float32[]) =
        match a |> Array.map Util.IsNice |> Array.tryFind not with
        | Some(_) -> false
        | _ -> true
    static member IsNice(a:float[]) =
        match a |> Array.map Util.IsNice |> Array.tryFind not with
        | Some(_) -> false
        | _ -> true
    static member IsNice(a:float32[,]) =
        match a |> Array2D.map Util.IsNice |> Array2D.tryFind not with
        | Some(_) -> false
        | _ -> true
    static member IsNice(a:float[,]) =
        match a |> Array2D.map Util.IsNice |> Array2D.tryFind not with
        | Some(_) -> false
        | _ -> true

    static member (=~) (a:float32, b:float32) =
        if   System.Single.IsNaN(a) then
             System.Single.IsNaN(b)
        elif System.Single.IsPositiveInfinity(a) || (a = System.Single.MaxValue) then
             System.Single.IsPositiveInfinity(b) || (b = System.Single.MaxValue)
        elif System.Single.IsNegativeInfinity(a) || (a = System.Single.MinValue) then
             System.Single.IsNegativeInfinity(b) || (b = System.Single.MinValue)
        else
             abs (a - b) < eps32

    static member (=~) (a:float, b:float) =
        if   System.Double.IsNaN(a) then
             System.Double.IsNaN(b)
        elif System.Double.IsPositiveInfinity(a) || (a = System.Double.MaxValue) then
             System.Double.IsPositiveInfinity(b) || (b = System.Double.MaxValue)
        elif System.Double.IsNegativeInfinity(a) || (a = System.Double.MinValue) then
             System.Double.IsNegativeInfinity(b) || (b = System.Double.MinValue)
        else
             abs (a - b) < eps64

    static member (=~) (a:DiffSharp.AD.Float32.D, b:DiffSharp.AD.Float32.D) =
        Util.(=~)(float32 a, float32 b)

    static member (=~) (a:DiffSharp.AD.Float64.D, b:DiffSharp.AD.Float64.D) =
        Util.(=~)(float a, float b)

    static member (=~) (a:float32[], b:float32[]) =
        if (a.Length = 0) && (b.Length = 0) then
            true
        elif a.Length <> b.Length then
            false
        else
            match Array.map2 (fun (x:float32) (y:float32) -> (Util.(=~)(x, y))) a b |> Array.tryFind not with
            | Some(_) -> false
            | _ -> true

    static member (=~) (a:float[], b:float[]) =
        if (a.Length = 0) && (b.Length = 0) then
            true
        elif a.Length <> b.Length then
            false
        else
            match Array.map2 (fun (x:float) (y:float) -> (Util.(=~)(x, y))) a b |> Array.tryFind not with
            | Some(_) -> false
            | _ -> true

    static member (=~) (a:float32[,], b:float32[,]) =
        if (a.Length = 0) && (b.Length = 0) then
            true
        elif ((Array2D.length1 a) <> (Array2D.length1 b)) || ((Array2D.length2 a) <> (Array2D.length2 b))then
            false
        else
            match Array2D.map2 (fun (x:float32) (y:float32) -> (Util.(=~)(x, y))) a b |> Array2D.tryFind not with
            | Some(_) -> false
            | _ -> true

    static member (=~) (a:float[,], b:float[,]) =
        if (a.Length = 0) && (b.Length = 0) then
            true
        elif ((Array2D.length1 a) <> (Array2D.length1 b)) || ((Array2D.length2 a) <> (Array2D.length2 b))then
            false
        else
            match Array2D.map2 (fun (x:float) (y:float) -> (Util.(=~)(x, y))) a b |> Array2D.tryFind not with
            | Some(_) -> false
            | _ -> true
