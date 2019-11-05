namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model

[<AbstractClass>]
type Optimizer(model:Model) =
    member val Model = model
    abstract member ParameterUpdate : Parameter -> unit
    member o.Step() = o.Model.IterParameters(fun _ p -> o.ParameterUpdate(p))

type SGD(model, lr:float) =
    inherit Optimizer(model)
    override o.ParameterUpdate(p) = p.Tensor <- p.Tensor.Primal - lr * p.Tensor.Derivative