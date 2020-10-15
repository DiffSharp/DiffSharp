namespace DiffSharp

open DiffSharp
open System.Diagnostics.CodeAnalysis

[<AutoOpen>]
module SlicingExtensions =
  type Tensor with
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option) =
        // Dims: 1
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int) =
        // Dims: 1
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let bounds = array2D [[i0min; i0max; i0given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option) =
        // Dims: 2
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int) =
        // Dims: 2
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option) =
        // Dims: 2
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int) =
        // Dims: 2
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int) =
        // Dims: 3
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int) =
        // Dims: 3
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int) =
        // Dims: 4
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int) =
        // Dims: 4
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int) =
        // Dims: 5
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0min:int option, i0max:int option, i1:int, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = if i0min.IsSome || i0max.IsSome then 1 else 0
        let i0min   = defaultArg i0min 0
        let i0max   = defaultArg i0max (t.shape.[0] - 1)
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1min:int option, i1max:int option, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = if i1min.IsSome || i1max.IsSome then 1 else 0
        let i1min   = defaultArg i1min 0
        let i1max   = defaultArg i1max (t.shape.[1] - 1)
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2min:int option, i2max:int option, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = if i2min.IsSome || i2max.IsSome then 1 else 0
        let i2min   = defaultArg i2min 0
        let i2max   = defaultArg i2max (t.shape.[2] - 1)
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3min:int option, i3max:int option, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = if i3min.IsSome || i3max.IsSome then 1 else 0
        let i3min   = defaultArg i3min 0
        let i3max   = defaultArg i3max (t.shape.[3] - 1)
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4min:int option, i4max:int option, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = if i4min.IsSome || i4max.IsSome then 1 else 0
        let i4min   = defaultArg i4min 0
        let i4max   = defaultArg i4max (t.shape.[4] - 1)
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int, i5min:int option, i5max:int option) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = if i5min.IsSome || i5max.IsSome then 1 else 0
        let i5min   = defaultArg i5min 0
        let i5max   = defaultArg i5max (t.shape.[5] - 1)
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
    [<ExcludeFromCodeCoverage>]
    /// <summary></summary> <exclude />
    member t.GetSlice(i0:int, i1:int, i2:int, i3:int, i4:int, i5:int) =
        // Dims: 6
        let i0given = 1
        let i0min   = i0
        let i0max   = i0
        let i1given = 1
        let i1min   = i1
        let i1max   = i1
        let i2given = 1
        let i2min   = i2
        let i2max   = i2
        let i3given = 1
        let i3min   = i3
        let i3max   = i3
        let i4given = 1
        let i4min   = i4
        let i4max   = i4
        let i5given = 1
        let i5min   = i5
        let i5max   = i5
        let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]; [i4min; i4max; i4given]; [i5min; i5max; i5given]]
        t.GetSlice(bounds)
