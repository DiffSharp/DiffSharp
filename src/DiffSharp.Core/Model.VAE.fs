// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp
open DiffSharp.Util

/// <summary>Variational auto-encoder base</summary>
[<AbstractClass>]
type VAEBase(zDim:int) =
    inherit Model()

    let sampleLatent mu (logVar:Tensor) =
        let std = dsharp.exp(0.5*logVar)
        let eps = dsharp.randnLike(std)
        eps.mul(std).add(mu)

    abstract member encode: Tensor -> Tensor * Tensor
    abstract member decode: Tensor -> Tensor

    member m.encodeDecode(x:Tensor) =
        let mu, logVar = m.encode x
        let z = sampleLatent mu logVar
        m.decode z, mu, logVar

    override m.forward(x) =
        let x, _, _ = m.encodeDecode(x) in x

    static member loss(xRecon:Tensor, x:Tensor, mu:Tensor, logVar:Tensor) =
        let bce = dsharp.bceLoss(xRecon, x.viewAs(xRecon), reduction="sum")
        let kl = -0.5 * dsharp.sum(1. + logVar - mu.pow(2.) - logVar.exp())
        bce + kl

    member m.loss(x, ?normalize:bool) =
        let normalize = defaultArg normalize true
        let xRecon, mu, logVar = m.encodeDecode x
        let loss = VAEBase.loss(xRecon, x, mu, logVar)
        if normalize then loss / x.shape[0] else loss

    member m.sample(?numSamples:int) = 
        let numSamples = defaultArg numSamples 1
        dsharp.randn([|numSamples; zDim|]) |> m.decode


/// <summary>Variational auto-encoder</summary>
type VAE(xShape:seq<int>, zDim:int, encoder:Model, decoder:Model) =
    inherit VAEBase(zDim)
    // TODO: check if encoder can accept input with xShape
    let encoderOutputDim = encoder.forward(dsharp.zeros(xShape).unsqueeze(0)).flatten().nelement
    let prez = Linear(encoderOutputDim, zDim*2)
    let postz = Linear(zDim, encoderOutputDim)
    do
        // TODO: check if decoder can accept input with (-1, zDim)
        // let decodedExample = xExample --> encoder --> decoder
        // if decodedExample.shape <> xShape then failwithf "Expecting decoder's output shape (%A) to be xShape (%A)" decodedExample.shape xShape
        base.addModel(encoder,decoder,prez,postz)

    override _.encode x =
        let mulogvar = x --> encoder --> prez
        let h = mulogvar.split([zDim; zDim], dim=1)
        let mu, logVar = h[0], h[1]
        mu, logVar

    override _.decode z =
        z --> postz -->decoder

    override _.ToString() = sprintf "VAE(%A, %A, %A, %A)" xShape zDim encoder decoder


/// <summary>Variational auto-encoder with multilayer perceptron (MLP) encoder and decoder.</summary>
type VAEMLP(xDim:int, zDim:int, ?hDims:seq<int>, ?nonlinearity:Tensor->Tensor, ?nonlinearityLast:Tensor->Tensor) =
    inherit VAEBase(zDim)
    let hDims = defaultArg hDims (let d = (xDim+zDim)/2 in seq [d; d]) |> Array.ofSeq
    let nonlinearity = defaultArg nonlinearity dsharp.relu
    let nonlinearityLast = defaultArg nonlinearityLast dsharp.sigmoid
    let dims =
        if hDims.Length = 0 then
            [|xDim; zDim|]
        else
            Array.append (Array.append [|xDim|] hDims) [|zDim|]
            
    let enc:Model[] = Array.append [|for i in 0..dims.Length-2 -> Linear(dims[i], dims[i+1])|] [|Linear(dims[dims.Length-2], dims[dims.Length-1])|]
    let dec:Model[] = Array.rev [|for i in 0..dims.Length-2 -> Linear(dims[i+1], dims[i])|]
    do 
        base.addModel(enc)
        base.addModel(dec)

    override _.encode (x:Tensor) =
        let batchSize = x.shape[0]
        let mutable x = x.view([batchSize; xDim])
        for i in 0..enc.Length-3 do
            x <- nonlinearity <| enc[i].forward(x)
        let mu = enc[enc.Length-2].forward(x)
        let logVar = enc[enc.Length-1].forward(x)
        mu, logVar

    override _.decode z =
        let mutable h = z
        for i in 0..dec.Length-2 do
            h <- nonlinearity <| dec[i].forward(h)
        nonlinearityLast <| dec[dec.Length-1].forward(h)

    override _.ToString() = sprintf "VAEMLP(%A, %A, %A)" xDim hDims zDim
