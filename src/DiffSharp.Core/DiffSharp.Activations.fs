namespace DiffSharp

    [<AutoOpen>]
    module ActivationsExtensions =

        /// Defines an extension as an element-wise operation using the function and its derivative
        let inline elementwiseOp nm f deriv =
            Tensor.Op
                { new UnaryOpElementwise(nm ) with 
                    member _.fRaw(a) = f a
                    member t.dfda(a, f) = deriv f a
                    }

        type Tensor with

            /// <summary>Applies the exponential linear unit function element-wise.</summary>
            /// <param name="alpha">The alpha parameter to the elu function. Default: 1.0.</param>
            member a.elu(?alpha: double) =
                let alpha = defaultArg alpha 1.0
                elementwiseOp "elu"
                    (fun a -> a.EluT(alpha, 1.0, 1.0)) 
                    (fun f a -> failwith "deriv of elu NYI") a

            /// <summary>Applies the sigmoid linear unit function element-wise.</summary>
            member a.silu() =
                elementwiseOp "silu"
                    (fun a -> a.SiluT())
                    (fun f a -> failwith "deriv of silu NYI") a

            /// <summary>Applies the gaussian error linear unit function element-wise.</summary>
            member a.gelu() =
                elementwiseOp  "gelu"
                    (fun a -> a.GeluT())
                    (fun f a -> failwith "deriv of gelu NYI") a

            /// <summary>Applies the hardswish function element-wise.</summary>
            member a.hardswish() = 
                elementwiseOp "hardswish"
                    (fun a -> a.HardswishT())
                    (fun f a -> failwith "deriv of hardswish NYI") a

            /// <summary>Applies the rectified linear unit function (6 max) element-wise.</summary>
            member a.relu6() =
                elementwiseOp "relu6"
                    (fun a -> a.Relu6T())
                    (fun f a -> failwith "deriv of relu6 NYI") a

            /// <summary>Applies the hardsigmoid function element-wise.</summary>
            member a.hardsigmoid() =
                elementwiseOp "hardsigmoid"
                    (fun a -> a.HardsigmoidT())
                    (fun f a -> failwith "deriv of hardsigmoid NYI") a

        type dsharp with

            /// <summary>Applies the exponential linear unit function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            /// <param name="alpha">The alpha parameter to the elu function. Default: 1.0.</param>
            static member elu(input:Tensor, ?alpha: double) =
                input.elu(?alpha=alpha)

            /// <summary>Applies the sigmoid linear unit function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member silu(input:Tensor) = input.silu()

            /// <summary>Applies the gaussian error linear unit function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member gelu(input:Tensor) = input.gelu()

            /// <summary>Applies the hardswish function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member hardswish(input:Tensor) = input.hardswish()

            /// <summary>Applies the rectified linear unit function (6 max) element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member relu6(input:Tensor) = input.relu6()

            /// <summary>Applies the hardsigmoid function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member hardsigmoid(input:Tensor) = input.hardsigmoid()

