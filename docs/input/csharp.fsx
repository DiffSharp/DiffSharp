
(**
Interoperability with Other Languages
=====================================

As F# can interoperate seamlessly with C# and other [CLI languages](http://en.wikipedia.org/wiki/List_of_CLI_languages), DiffSharp can be used with these languages as well. Your project should reference the **DiffSharp.dll** assembly, its dependencies, and also the **FSharp.Core.dll** assembly.

For C# and other languages, the **DiffSharp.Interop** namespace provides a simple way of accessing main DiffSharp functionality. (Without **DiffSharp.Interop**, you can still use the regular DiffSharp namespaces, but you will need to take care of issues such as converting to and from [**FSharp.Core.FSharpFunc**] objects (https://msdn.microsoft.com/en-us/library/ee340302.aspx).)

Using DiffSharp with C#
=======================

Nested Automatic Differentiation
--------------------------------

For using the nested forward and reverse AD capability, you need to write the part of your numeric code where you need deriatives (e.g. for optimization) using the **DiffSharp.Interop.D** numeric type, the results of which you may convert later to an integral type such as **double** (also see [Type Inference](gettingstarted-typeinference.html) and [Nested AD](gettingstarted-nestedad.html)).

C# versions of the differentiation operations (see [API Overview](api-overview.html)) are provided through the **DiffSharp.Interop.AD** wrapper class, which internally handles conversions to and from C# functions. The names of differentiation operations (e.g. **diff**, **grad**, **hessian** ) remain the same, only their first letters are capitalized (e.g. **AD.Diff**, **AD.Grad**, **AD.Hessian** ).

The **DiffSharp.Interop.AD** wrapper class also provides common mathematical functions (e.g. **AD.Exp**, **AD.Sin**, **AD.Pow** ) for the **D** type, similar to the use of [**System.Math**](https://msdn.microsoft.com/en-us/library/System.Math(v=vs.110).aspx) class with the **double** type and other types.

    [lang=csharp]
    using DiffSharp.Interop;

Currently the following operations are supported by **DiffSharp.Interop.AD**:

- **AD.Diff**: First derivative of a scalar-to-scalar function
- **AD.Diff2**: Second derivative of a scalar-to-scalar function
- **AD.Diffn**: N-th derivative of a scalar-to-scalar function
- **AD.Grad**: Gradient of a vector-to-scalar function
- **AD.Gradv**: Gradient-vector product (directional derivative)
- **AD.Laplacian**: Laplacian of a vector-to-scalar function
- **AD.Jacobian**: Jacobian of a vector-to-vector function
- **AD.JacobianT**: Transposed Jacobian of a vector-to-vector function
- **AD.Jacobianv**: Jacobian-vector product
- **AD.JacobianTv**: Transposed Jacobian-vector product
- **AD.Hessian**: Hessian of a vector-to-scalar function
- **AD.Hessianv**: Hessian-vector product
- **AD.Curl**: Curl of a vector-to-vector function
- **AD.Div**: Divergence of a vector-to-vector function

Numerical Differentiation
-------------------------

**DiffSharp.Interop** also provides 
*)

