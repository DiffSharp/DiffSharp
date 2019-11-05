namespace DiffSharp.Optim
open DiffSharp
open DiffSharp.Model

[<AbstractClass>]
type Optimizer() =
    abstract member Step : unit -> unit
