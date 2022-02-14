// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model


/// <summary>A model which during training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.</summary>
type Dropout(?p:double) =
    inherit Model()

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "Dropout()"

    /// <summary>TBD</summary>
    override m.forward(value) =
        if m.mode = Mode.Train then value.dropout(?p=p) else value


/// <summary>A model which during training, randomly zero out entire channels. Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.</summary>
type Dropout2d(?p:double) =
    inherit Model()

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "Dropout2d()"

    /// <summary>TBD</summary>
    override m.forward(value) =
        if m.mode = Mode.Train then value.dropout2d(?p=p) else value


/// <summary>A model which during training, randomly zero out entire channels. Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.</summary>
type Dropout3d(?p:double) =
    inherit Model()

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "Dropout3d()"

    /// <summary>TBD</summary>
    override m.forward(value) =
        if m.mode = Mode.Train then value.dropout3d(?p=p) else value
