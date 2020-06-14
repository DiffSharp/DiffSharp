namespace DiffSharp.Model
open DiffSharp
open DiffSharp.Util
open System.Collections.Generic


type Parameter =
    val mutable value:Tensor
    new(value) = {value=value}
    member p.forwardDiff(derivative:Tensor, ?tag:uint32) = p.value <- p.value.forwardDiff(derivative, ?tag=tag)
    member p.reverseDiff(?tag:uint32) = p.value <- p.value.reverseDiff(?tag=tag)
    member p.noDiff() = p.value <- p.value.noDiff()
    member p.move(?dtype, ?device, ?backend) = p.value <- p.value.move(?dtype=dtype, ?device=device, ?backend=backend)
    override p.ToString() = sprintf "Parameter(shape:%A, value:%A)" p.value.shape p.value


type ParameterDict() =
    member val values = Dictionary<string, Parameter>()
    member d.Item
        with get key = d.values.[key].value
        and set key v = d.values.[key].value <- v
    member d.add(name, parameter) = d.values.Add(name, parameter)
    member d.add(parameters:list<string*Parameter>) = for (n, p) in parameters do d.add(n, p)
    member d.add(parameters:ParameterDict) = for KeyValue(n, p) in parameters.values do d.add(n, p)
    member d.copy() = d.map(fun (t:Tensor) -> t)
    member d.map(f:string*Parameter->string*Parameter) =
        let ret = ParameterDict()
        for KeyValue(n, p) in d.values do ret.values.Add(f(n,p))
        ret
    member d.map(f:string*Tensor->string*Tensor) = d.map(fun (n, p:Parameter) -> let nn, tt = f(n, p.value) in nn, Parameter(tt))
    member d.map(f:Tensor->Tensor) = d.map(fun (n,t) -> n, f t)
    member d.set(parameters:ParameterDict) = d.iter(fun (n, p) -> p.value <- parameters.[n])
    member d.iter(f:string*Parameter->unit) = for KeyValue(n, p) in d.values do f(n,p)
    member d.forwarddiff(derivatives:ParameterDict, ?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        d.iter(fun (n, p) -> p.forwardDiff(derivatives.[n], tag))
    member d.reverseDiff(?tag:uint32) = 
        let tag = defaultArg tag GlobalNestingLevel.Current
        d.iter(fun (_, p) -> p.reverseDiff(tag))
    member d.noDiff() = d.iter(fun (_, p) -> p.noDiff())
    member d.move(?dtype, ?device, ?backend) = d.iter (fun (_, p) -> p.move(?dtype=dtype, ?device=device, ?backend=backend))
    member d.primal with get() = d.map(fun (t:Tensor)->t.primal)
    member d.derivative with get() = d.map(fun (t:Tensor)->t.derivative)
    member d.nelement with get() = [|for t in d.values.Values do t.value.nelement|] |> Array.sum
    member d.flatten() =
        let ts = [for t in d.values.Values do t.value.view(-1)]
        dsharp.cat(ts)
    member d.unflatten(tensors:Tensor) =
        if tensors.dim <> 1 then failwithf "Expecting 1d tensors but received tensors with shape %A" tensors.shape
        if tensors.nelement <> d.nelement then failwithf "Expecting tensors.nelement (%A) and ParameterDict.nelement (%A) to be the same" tensors.nelement d.nelement
        let shapes = [|for t in d.values.Values do t.value.shape|]
        let sizes = [|for s in shapes do shapeLength s|]
        let ts = Array.map2 (fun (t:Tensor) (s:int[]) -> t.view(s)) (tensors.split(sizes)) shapes
        let mutable i = 0
        let keys = copyKeys d.values
        for n in keys do
            d.[n] <- ts.[i]
            i <- i+1
    member d.unflattenToNew(tensors:Tensor) = 
        let dd = d.copy()
        dd.unflatten(tensors)
        dd
    override d.ToString() =
        let sb = System.Text.StringBuilder()
        sb.Append("ParameterDict(") |> ignore
        let mutable prefix = ""
        for KeyValue(n, p) in d.values do 
            sb.Append(sprintf "%s%A:%A" prefix n p) |> ignore
            prefix <- ", "
        sb.Append(")") |> ignore
        sb.ToString()


type Mode =
    | Train = 0
    | Eval = 1


[<AbstractClass>]
type Model() =
    [<DefaultValue>]
    val mutable mode: Mode
    member val ParametersDict = ParameterDict()
    member val SubModelsDict = Dictionary<string, Model>()
    member m.train() = 
        m.mode <- Mode.Train
        for model:Model in m.allModels do model.mode <- Mode.Train
    member m.eval() = 
        m.mode <- Mode.Eval
        for model:Model in m.allModels do model.mode <- Mode.Eval
    member m.parametersDict
        with get () = m.ParametersDict
        and set parameters = m.ParametersDict.set(parameters)
    member m.parameters
        with get () = m.parametersDict.flatten()
        and set parameters = m.parametersDict.unflatten(parameters)
    member m.allModels
        with get () =
            if m.SubModelsDict.Count = 0 then [m]
            else [for sm in m.SubModelsDict.Values do yield! sm.allModels]
    member m.add(parameters:seq<obj>, ?names:seq<string>) =
        let parameters = parameters |> Seq.toArray
        let names = defaultArg names (Seq.init (parameters.Length) (fun i -> sprintf "m__%d" i)) |> Seq.toArray
        if parameters.Length <> names.Length then failwithf "Expecting parameters.Length (%A) and names.Length (%A) to be same" parameters.Length names.Length
        for p, n in Array.zip parameters names do
            match (box p) with
            | :? Parameter as p -> 
                m.parametersDict.add(n, p)
            | :? Model as mm ->
                m.SubModelsDict.Add(n, mm)
                m.parametersDict.add(mm.parametersDict.map(fun (nn, pp:Parameter) -> (n + "__" + nn, pp)))
            | _ -> failwithf "Unsupported type. Expecting a Parameter or Model"
    member m.forwardDiff(derivatives:ParameterDict) = m.parametersDict.forwarddiff(derivatives)
    member m.reverseDiff() = m.parametersDict.reverseDiff()
    member m.noDiff() = m.parametersDict.noDiff()
    member m.move(?dtype, ?device, ?backend) = m.parametersDict.move(?dtype=dtype, ?device=device, ?backend=backend)
    member m.nparameters = m.parametersDict.nelement
    abstract member forward: Tensor -> Tensor
    member m.forwardParameters (input:Tensor) (parameters:Tensor) =
        m.parameters <- parameters
        let f = m.forward(input) in m.noDiff(); f
    member m.forwardCompose (f:Tensor->Tensor) (input:Tensor) (parameters:Tensor) =
        m.forwardParameters input parameters |> f
    member m.forwardLoss (f:Tensor->Tensor->Tensor) (input:Tensor) (target:Tensor) (parameters:Tensor) =
        m.forwardCompose (f target) input parameters
    static member create ps f =
        let model = { new Model() with override __.forward(x) = f x}
        model.add(ps)
        model
    override m.ToString() =
        let sb = System.Text.StringBuilder()
        sb.Append("Model(\n") |> ignore
        for model in m.allModels do sb.Append(sprintf "%A\n" model) |> ignore
        sb.Append(")") |> ignore
        sb.ToString()
    static member compose (m1:Model) (m2:Model) = Model.create [m1; m2] (m1.forward >> m2.forward)
    static member (-->) (m1:Model, m2:Model) = Model.compose m1 m2
    static member (-->) (m:Model, f:Tensor->Tensor) = Model.create [m] (m.forward >> f)
    static member (-->) (f:Tensor->Tensor, m:Model) = Model.create [m] (f >> m.forward)
    static member (-->) (t:Tensor, m:Model) = m.forward t
    member m.saveParameters(fileName) = m.parameters.save(fileName)
    member m.loadParameters(fileName) = m.parameters <- Tensor.load(fileName)
    member m.save(fileName) = saveBinary m fileName
    static member load(fileName):Model = loadBinary fileName
    member m.clone() = 
        let fileName = System.IO.Path.GetTempFileName()
        m.save(fileName)
        Model.load(fileName)


type Weight() =
    static member kaiming(fanIn, fanOut, ?a:float) = 
        // He et al. 2015. https://arxiv.org/abs/1502.01852
        let a = defaultArg a (sqrt 5.)
        let w = dsharp.randn([fanIn; fanOut])
        let s = sqrt (2. / ((1. + a*a) * (float fanIn)))
        w * s
    static member standard(shape:int[], k:float) =
        -k + dsharp.rand(shape) * 2*k


type Linear(inFeatures, outFeatures, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let w = Parameter(Weight.kaiming(inFeatures, outFeatures))
    let k = 1./sqrt (float outFeatures)
    let b = Parameter(if bias then Weight.standard([|outFeatures|], k) else dsharp.zero())
    do base.add([w;b],["Linear__weight";"Linear__bias"])
    override l.ToString() = sprintf "Linear(%A, %A)" inFeatures outFeatures
    override l.forward(value) =
        let f = dsharp.matmul(value, w.value)
        if bias then f + b.value else f


type Conv1d(inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize))
    let w = Parameter <| Weight.standard([|outChannels; inChannels; kernelSize|], k)
    let b = Parameter <| if bias then Weight.standard([|outChannels|], k) else dsharp.zero()
    do base.add([w;b],["Conv1d__weight";"Conv1d__bias"])
    override c.ToString() = sprintf "Conv1d(%A, %A, %A)" inChannels outChannels kernelSize
    override c.forward(value) =
        let f = dsharp.conv1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation)
        if bias then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1]) else f


type Conv2d(inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = 
        match kernelSize, kernelSizes with
        | Some _ , Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
        | Some k, None -> [|k; k|]
        | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 2 then failwithf "Expecting kernelSizes to have length two" else k
        | _ -> [|1; 1|]
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]))
    let w = Parameter <| Weight.standard([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]|], k)
    let b = Parameter <| if bias then Weight.standard([|outChannels|], k) else dsharp.zero()
    do base.add([w;b],["Conv2d__weight";"Conv2d__bias"])
    override c.ToString() = sprintf "Conv2d(%A, %A, %A)" inChannels outChannels kernelSizes
    override c.forward(value) =
        let f = dsharp.conv2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1; 1]) else f


type Conv3d(inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = 
        match kernelSize, kernelSizes with
        | Some _ , Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
        | Some k, None -> [|k; k; k|]
        | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 3 then failwithf "Expecting kernelSizes to have length three" else k
        | _ -> [|1; 1; 1|]
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]*kernelSizes.[2]))
    let w = Parameter <| Weight.standard([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]; kernelSizes.[2]|], k)
    let b = Parameter <| if bias then Weight.standard([|outChannels|], k) else dsharp.zero()
    do base.add([w;b],["Conv3d__weight";"Conv3d__bias"])
    override c.ToString() = sprintf "Conv3d(%A, %A, %A)" inChannels outChannels kernelSizes
    override c.forward(value) =
        let f = dsharp.conv3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1; 1; 1]) else f


type Dropout(?p:double) =
    inherit Model()
    override d.ToString() = sprintf "Dropout()"
    override d.forward(value) =
        if d.mode = Mode.Train then value.dropout(?p=p) else value


type Dropout2d(?p:double) =
    inherit Model()
    override d.ToString() = sprintf "Dropout2d()"
    override d.forward(value) =
        if d.mode = Mode.Train then value.dropout2d(?p=p) else value


type Dropout3d(?p:double) =
    inherit Model()
    override d.ToString() = sprintf "Dropout3d()"
    override d.forward(value) =
        if d.mode = Mode.Train then value.dropout3d(?p=p) else value