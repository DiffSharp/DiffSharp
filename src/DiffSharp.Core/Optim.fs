namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model
open System.Collections.Generic

[<AbstractClass>]
type Optimizer(model:Model) =
    member val Model = model
    abstract member ParameterUpdate: string -> Tensor -> Tensor
    member o.Step() = o.Model.UpdateParameters(o.Model.Parameters.Map(o.ParameterUpdate))


type SGD(model, ?learningRate:Tensor, ?momentum:Tensor, ?nesterov:bool) =
    inherit Optimizer(model)
    let lr = defaultArg learningRate (Tensor.Create(0.001))
    let mom = momentum
    let nesterov = defaultArg nesterov true
    let momentumBuffer = TensorDict()
    override o.ParameterUpdate name t = 
        match mom with
        | Some mom -> 
            if nesterov then t.Primal - lr * t.Derivative
            else t.Primal - lr * t.Derivative
        | None -> t.Primal - lr * t.Derivative