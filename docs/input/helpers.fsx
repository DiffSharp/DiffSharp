(*** hide ***)
namespace global
#r "../../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"


(**
API Helpers
============
*)

open DiffSharp

type Scalar = Tensor
type Vec = Tensor
type Mat = Tensor
[<AutoOpen>]
module Globals =
    let v (x: 'T) : Scalar = dsharp.tensor x
    let vec (x: 'T seq) : Vec = dsharp.tensor x
    let mat (x: seq< #seq<'T> >) : Mat = dsharp.tensor x
    let value (x: Tensor) : 'T = x.toScalar() :?> 'T
    let values (x: Tensor) : 'T = x.toArray() :?> 'T

module Vec =
    let zeros n : Vec = dsharp.zeros [n]
    let full n x : Vec = dsharp.full([n],x)
    let init n f : Vec = dsharp.full([n],[| for i in 0 ..n-1 -> f i|])
    let length (v: Vec) = v.shape.[0]
    let min (v: Vec) = v.min()
    let max (v: Vec) = v.max()
    let sum (v: Vec) = v.sum()
module Mat =
    let zeros n m : Mat = dsharp.zeros [n; m]
    let full n m x : Mat = dsharp.full([n;m],x)
    let init n m f : Mat = dsharp.full([n;m],[| for i in 0 ..n-1 do for j in 0..m-1 do yield f i j|])
    let initRows n (f: int -> Vec) : Mat = dsharp.tensor([| for i in 0 ..n-1 do yield f n |])
    let initCols n (f: int -> Vec) : Mat = dsharp.tensor([| for i in 0 ..n-1 do yield f n |]).transpose()
    let nrows (v: Mat) = v.shape.[0]
    let ncols (v: Mat) = v.shape.[1]

