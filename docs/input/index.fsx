(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
DiffSharp: Automatic Differentiation Library
============================================

DiffSharp (∂#) is an _automatic differentiation_ (AD) library implemented in the F# language.

AD allows exact and efficient calculation of derivatives, by systematically applying the chain rule of calculus at the elementary operator level. AD is different from _numerical differentiation_, which is prone to truncation and round-off errors, and _symbolic differentiation_, which is exact but not efficient for run-time calculations and can only handle closed-form mathematical expressions.

Using the DiffSharp library, derivative calculations can be incorporated with minimal change into existing algorithms.

The library is under active development by [Atılım Güneş Baydin](http://www.cs.nuim.ie/~gunes/) and [Barak A. Pearlmutter](http://bcl.hamilton.ie/~barak/) mainly for research applications in machine learning, as part of their work at the [Brain and Computation Lab](http://www.bcl.hamilton.ie/), Hamilton Institute, National University of Ireland Maynooth.

Installation
------------
You can download the [compiled library binaries](https://github.com/gbaydin/DiffSharp/) as a zip file or install the library package via NuGet.

<div class="row">
    <div class="span1"></div>
    <div class="span6">
    <div class="well well-small" id="nuget">
        The DiffSharp library can be <a href="https://nuget.org/packages/DiffSharp">installed from NuGet</a>. To install, run the following command in the <a href="http://docs.nuget.org/docs/start-here/using-the-package-manager-console">Package Manager Console</a>:
        <pre>PM> Install-Package DiffSharp</pre>
    </div>
    </div>
    <div class="span1"></div>
</div>

Implemented Techniques
----------------------

The main focus of the DiffSharp library is AD, but we also implement symbolic and numerical differentiation.

We are working on the following features for the next release:

- Handling of nested AD operations
- AD via source code transformation
- Use of code quotations for AD

Quick Usage Example
-------------------
*)

// Use forward mode AD
open DiffSharp.AD.Forward

// A scalar-to-scalar function
let f x = 
    sin (sqrt x)

// Derivative of f
let f' = diff f

// Value of the derivative of f at a point
let y' = f' 2.

(**
More Info
---------

For a quick overview of AD and other differentiation methods, you can refer to our [recent poster](http://www.cs.nuim.ie/~gunes/files/AGBaydinICML2014Poster.pdf) for the AutoML workshop at the International Conference on Machine Learning 2014. For in-depth material, you can check our [publications page](http://www.bcl.hamilton.ie/publications/) and the [autodiff.org](http://www.autodiff.org/) website.
*)
