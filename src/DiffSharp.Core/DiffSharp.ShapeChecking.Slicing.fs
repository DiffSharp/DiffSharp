namespace DiffSharp.ShapeChecking

open DiffSharp
open System.Diagnostics.CodeAnalysis

[<AutoOpen>]
module SlicingExtensions =
  type Tensor with
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option) =
        // Dims: 1
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int) =
        // Dims: 1
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option) =
        // Dims: 2
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int) =
        // Dims: 2
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option) =
        // Dims: 2
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int) =
        // Dims: 2
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option) =
        // Dims: 3
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int) =
        // Dims: 3
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option) =
        // Dims: 3
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int) =
        // Dims: 3
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int) =
        // Dims: 4
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int, i4min:Int option, i4max:Int option) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int, i4:Int) =
        // Dims: 5
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:Int option, i0max:Int option, i1:Int, i2:Int, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1I else 0I
        let i0min   = defaultArg i0min 0I
        let i0max   = defaultArg i0max (t.shapex.[0] - 1)
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1min:Int option, i1max:Int option, i2:Int, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1I else 0I
        let i1min   = defaultArg i1min 0I
        let i1max   = defaultArg i1max (t.shapex.[1] - 1)
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2min:Int option, i2max:Int option, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1I else 0I
        let i2min   = defaultArg i2min 0I
        let i2max   = defaultArg i2max (t.shapex.[2] - 1)
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3min:Int option, i3max:Int option, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1I else 0I
        let i3min   = defaultArg i3min 0I
        let i3max   = defaultArg i3max (t.shapex.[3] - 1)
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int, i4min:Int option, i4max:Int option, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1I else 0I
        let i4min   = defaultArg i4min 0I
        let i4max   = defaultArg i4max (t.shapex.[4] - 1)
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int, i4:Int, i5min:Int option, i5max:Int option) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1I else 0I
        let i5min   = defaultArg i5min 0I
        let i5max   = defaultArg i5max (t.shapex.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:Int, i1:Int, i2:Int, i3:Int, i4:Int, i5:Int) =
        // Dims: 6
        let i0given = 1I
        let i0min   = i0
        let i0max   = i0
        let i1given = 1I
        let i1min   = i1
        let i1max   = i1
        let i2given = 1I
        let i2min   = i2
        let i2max   = i2
        let i3given = 1I
        let i3min   = i3
        let i3max   = i3
        let i4given = 1I
        let i4min   = i4
        let i4max   = i4
        let i5given = 1I
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
