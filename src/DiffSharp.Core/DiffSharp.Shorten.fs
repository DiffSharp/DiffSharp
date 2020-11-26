// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

module DiffSharp.Shorten

// Functional automatic differentiation API shorthand names
type dsharp with

    /// <summary>TBD</summary>
    static member gvp f x v = dsharp.gradv f x v

    /// <summary>TBD</summary>
    static member g f x = dsharp.grad f x

    /// <summary>TBD</summary>
    static member hvp f x v = dsharp.hessianv f x v

    /// <summary>TBD</summary>
    static member h f x = dsharp.hessian f x

    /// <summary>TBD</summary>
    static member gh f x = dsharp.gradhessian f x

    /// <summary>TBD</summary>
    static member ghvp f x v = dsharp.gradhessianv f x v

    /// <summary>TBD</summary>
    static member jvp f x v = dsharp.jacobianv f x v

    /// <summary>TBD</summary>
    static member vjp f x v = dsharp.jacobianTv f x v

    /// <summary>TBD</summary>
    static member j f x = dsharp.jacobian f x

    /// <summary>TBD</summary>
    static member fgvp f x v = dsharp.fgradv f x v

    /// <summary>TBD</summary>
    static member fg f x = dsharp.fgrad f x

    /// <summary>TBD</summary>
    static member fgh f x = dsharp.fgradhessian f x

    /// <summary>TBD</summary>
    static member fhvp f x v = dsharp.fhessianv f x v

    /// <summary>TBD</summary>
    static member fh f x = dsharp.fhessian f x

    /// <summary>TBD</summary>
    static member fghvp f x v = dsharp.fgradhessianv f x v

    /// <summary>TBD</summary>
    static member fjvp f x v = dsharp.fjacobianv f x v

    /// <summary>TBD</summary>
    static member fvjp f x v = dsharp.fjacobianTv f x v

    /// <summary>TBD</summary>
    static member fj f x = dsharp.fjacobian f x    

