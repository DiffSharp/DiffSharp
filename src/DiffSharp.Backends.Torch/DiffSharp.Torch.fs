/// Defines extensions to the DiffSharp programming model when the Torch backend can be assumed
namespace DiffSharp.Torch

open DiffSharp
open DiffSharp.Backends.Torch
open TorchSharp

[<AutoOpen>]
module Extensions =

    type torch.Tensor with

        /// <summary>
        /// Creates a new DiffSharp tensor from the torch tensor.
        /// </summary>
        member tt.toTensor() : Tensor =
            Tensor.ofRawTensor(TorchRawTensor(tt))

    type Tensor with
        /// <summary>
        /// Converts the primal of a tensor to a torch tensor.
        /// </summary>
        /// <remarks>If the tensor does not use the Torch backend an exception is raised</remarks>
        member t.primalRawTorch() =
            match t.primalRaw with
            | :? TorchRawTensor as trt -> trt.TorchTensor
            | _ -> failwith $"primalRawTorch: the input is not a DiffSharp.Backends.Torch tensor, its backend is {t.backend}"

