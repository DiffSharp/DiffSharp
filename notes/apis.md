
//FM usage model:

open FM

[<Model>]  // = ReflectedDefinition = quotation
module NeuralStyles = 
    let PretrainedFFStyleVGG ... = ...


FM.TrainUsingTensorFlow <@ NeuralStyles.PretrainedFFStyleVGG @>
FM.TrainUsingDiffSharp <@ NeuralStyles.PretrainedFFStyleVGG @>
FM.TrainUsingSourceToSource <@ NeuralStyles.PretrainedFFStyleVGG @>
FM.ConvertToONNX <@ NeuralStyles.PretrainedFFStyleVGG @>
FM.CompileToC <@ NeuralStyles.PretrainedFFStyleVGG @>
FM.AddGradientsBySourceToSource <@ NeuralStyles.PretrainedFFStyleVGG @>
FM.AdvancedFunkyCompilerToC <@ NeuralStyles.PretrainedFFStyleVGG @>



// F# for AI Models API and capabilities:

namespace FM = 

   type public DType = 
       | DSingle
       | DDouble
       | DFloat16
       | DInferred // note types are inferred, currently using F# type inference

    type Dim = 
       | Inferred
       | Known // note shapes and dimensions are inferred

    type Shape = Dim[] * Inferred // note shapes and dimensions are inferred

    type Device = 
       | Cpu
       | Gpu
       | Inferred // note devices are inferred

    type DT<'T> = 
        member Shape: Shape
        //member DType: DType
        member Device: Device 

   DT.Zero  : DT  // (Shape = [..], DType=?, Device=?  )
   DT.Add  : v1: DT * v2: DT  // Shape.Compatible(v1.Shape, v2.Shape)
   // etc.

namespace DiffSharp

   type public DType = 
       | DSingle
       | DDouble
       | DFloat16

   type internal TensorValue =
       | TorchTensor of libTorchHandle: IntPtr
       | ScalarFloat32 of float32
       | ScalarFloat64 of float64
       member DType: DType
       member Shape: long[]
       member Device: string
       member OnGpu: TensorValue 
       member OnCpu: TensorValue 

   type DTensor =
       | Tensor of TensorValue
       | TensorF of primal: DTensor * derivative: DTensor * tag: uint32
       | TensorR of primal: DTensor * derivative: (DTensor ref) * parentOp: TensorOp * fanOut: (uint32 ref) * tag: uint32

       member DType: DType
       member Shape: long[]
       member Device: long[]
       member OnGpu: DTensor
       member OnCpu: DTensor 

// TensorFlow.FSharp and TensorFlow.NET today
namespace TensorFlow.FSharp

   type TFTensor
       member DType: TFDType // Float32, Float64, ...
       member Shape: TFShape 
       member Device: ...

   type TFOutput
       member DType: TFDType // Float32, Float64, ...
       member Shape: TFShape 

// Torch.Sharp today
namespace Torch.Sharp

   type ITorchTensor<'T>
       //does not have: member DType: TFDType // Float32, Float64, ...
       member Shape: TFShape 
       member Device: ...

   type FloatTensor etc.

