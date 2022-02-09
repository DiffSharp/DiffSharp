/// Defines extensions to the DiffSharp programming model when the Torch backend can be assumed
namespace DiffSharp.Torch

open DiffSharp
open DiffSharp.Backends.Torch
open TorchSharp

[<AutoOpen>]
module Extensions =

    type dsharp with

        /// <summary>
        /// Creates a new tensor from the torch tensor.
        /// </summary>
        /// <param name="tt">The given TorchSharp tensor.</param>
        static member ofTorchTensor(tt: torch.Tensor) =
            Tensor.ofRawTensor(TorchRawTensor(tt))

        /// <summary>
        /// Converts the primal of a tensor to a torch tensor.
        /// </summary>
        /// <remarks>If the tensor does not use the Torch backend an exception is raised</remarks>
        static member primalTorch(t: Tensor) =
            match t.primalRaw with
            | :? TorchRawTensor as trt -> trt.TorchTensor
            | _ -> failwith $"primalTorch: the input is not a DiffSharp.Backends.Torch tensor, its backend is {t.backend}"

    type Tensor with
        /// <summary>
        /// Converts the primal of a tensor to a torch tensor.
        /// </summary>
        /// <remarks>If the tensor does not use the Torch backend an exception is raised</remarks>
        member t.primalTorch() = dsharp.primalTorch t

