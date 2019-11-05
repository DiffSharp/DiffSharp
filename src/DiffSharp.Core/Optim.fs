namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model

[<AbstractClass>]
type Optimizer(model:Model) =
    member val Model = model
    abstract member ParameterUpdate : Parameter -> unit
    member o.Step() = o.Model.IterParameters(fun _ p -> o.ParameterUpdate(p))

type SGD(model, ?learningRate:Tensor, ?momentum:Tensor, ?nesterov:bool) =
    inherit Optimizer(model)
    let lr = defaultArg learningRate (Tensor.Create(0.001))
    let mom = momentum
    let nesterov = defaultArg nesterov true
    override o.ParameterUpdate(p) = 
        match mom with
        | Some mom -> 
            if nesterov then p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative
            else p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative
        | None -> p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative