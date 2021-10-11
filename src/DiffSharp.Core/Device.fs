// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

/// <summary>
///   Represents the type of a device. 
/// </summary>
///
/// <remarks>
///   The numeric values used are as for LibTorch.
/// </remarks>
///
/// <namespacedoc>
///   <summary>Contains fundamental types for the tensor programming model, including Tensor, Shape and dsharp.</summary>
/// </namespacedoc>
type DeviceType =
    | CPU = 0
    | CUDA = 1 // CUDA.
    | MKLDNN = 2 // Reserved for explicit MKLDNN
    | OPENGL = 3 // OpenGL
    | OPENCL = 4 // OpenCL
    | IDEEP = 5 // IDEEP.
    | HIP = 6 // AMD HIP
    | FPGA = 7 // FPGA
    | MSNPU = 8 // MSNPU
    | XLA = 9 // XLA / TPU

/// Represents a device specification.
[<Struct>]
type Device =
    | Device of DeviceType * int
    member x.DeviceType = (let (Device(a,_)) = x in a)
    member x.DeviceIndex = (let (Device(_,b)) = x in b)
    static member CPU = Device(DeviceType.CPU, -1)
    static member GPU = Device(DeviceType.CUDA, 0)

    member internal x.Code = (int x.DeviceType <<< 4) + x.DeviceIndex

    member internal x.Name =
       (match x.DeviceType with
        | DeviceType.CPU -> "cpu"
        | DeviceType.CUDA -> "cuda"
        | DeviceType.MKLDNN -> "mkldnn"
        | DeviceType.OPENGL -> "opengl"
        | DeviceType.OPENCL -> "opencl"
        | DeviceType.IDEEP -> "ideep"
        | DeviceType.HIP -> "hip"
        | DeviceType.FPGA -> "fpga"
        | DeviceType.MSNPU -> "msnpu"
        | DeviceType.XLA -> "xla"
        | _ -> failwith "unknown device type") + string x.DeviceIndex

    override x.ToString() = x.Name

/// Contains functions and settings related to device specifications.
module Device = 

    /// Get or set the default device used when creating tensors. Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default : Device = Device.CPU
