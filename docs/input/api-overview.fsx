(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
API Overview
============

The following table gives an overview of the differentiation API provided by the DiffSharp library.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-eugx{background-color:#e4ffb3;text-align:center}
.tg .tg-5y5n{background-color:#ecf4ff}
.tg .tg-e3zv{font-weight:bold}
.tg .tg-tfw5{background-color:#ffffc7;text-align:center}
.tg .tg-a0td{font-size:100%}
.tg .tg-lgsi{font-size:100%;background-color:#ffffc7}
.tg .tg-u986{font-size:100%;background-color:#e4ffb3}
.tg .tg-2sn5{background-color:#e4ffb3}
.tg .tg-40di{font-size:100%;background-color:#ecf4ff;color:#000000}
.tg .tg-dyge{font-weight:bold;font-size:100%}
.tg .tg-gkzk{font-size:100%;background-color:#ffffc7;text-align:center}
.tg .tg-v6es{font-size:100%;background-color:#e4ffb3;text-align:center}
.tg .tg-uy90{font-size:100%;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-ci37{background-color:#ecf4ff;text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-a0td"></th>
    <th class="tg-lgsi">diff</th>
    <th class="tg-lgsi">diff2</th>
    <th class="tg-lgsi">diffn</th>
    <th class="tg-u986">grad</th>
    <th class="tg-2sn5">gradv</th>
    <th class="tg-u986">hessian</th>
    <th class="tg-u986">gradhessian</th>
    <th class="tg-u986">laplacian</th>
    <th class="tg-40di">jacobian</th>
    <th class="tg-5y5n">jacobianv</th>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Forward</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-eugx">X</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-uy90">X</td>
    <td class="tg-ci37">X</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Forward2</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-eugx">X</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-uy90">X</td>
    <td class="tg-ci37">X</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.ForwardG</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-uy90">X</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.ForwardGH</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-v6es">X</td>
    <td class="tg-v6es">X</td>
    <td class="tg-uy90">X</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardN</td>
    <td class="tg-tfw5">X</td>
    <td class="tg-tfw5">X</td>
    <td class="tg-tfw5">X</td>
    <td class="tg-eugx">X</td>
    <td class="tg-eugx">X</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">X</td>
    <td class="tg-ci37">X</td>
    <td class="tg-ci37">X</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Reverse</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es">A</td>
    <td class="tg-v6es">XA</td>
    <td class="tg-v6es">A</td>
    <td class="tg-uy90">X</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.Numerical</td>
    <td class="tg-gkzk">A</td>
    <td class="tg-gkzk">A</td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">A</td>
    <td class="tg-eugx">A</td>
    <td class="tg-v6es">A</td>
    <td class="tg-v6es">A</td>
    <td class="tg-v6es">A</td>
    <td class="tg-uy90">A</td>
    <td class="tg-ci37">A</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.Symbolic</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-gkzk">X</td>
    <td class="tg-v6es">X</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es">X</td>
    <td class="tg-v6es">X</td>
    <td class="tg-v6es">X</td>
    <td class="tg-uy90">X</td>
    <td class="tg-ci37"></td>
  </tr>
</table>

**Yellow**: For scalar-to-scalar functions; **Green**: For vector-to-scalar functions; **Blue**: For vector-to-vector functions

**X**: Exact value; **A**: Numerical approximation; **XA**: Exact gradient, approximated Hessian

Implemented Techniques
----------------------

The main focus of the DiffSharp library is AD, but we also implement symbolic and numerical differentiation.

Currently, the library provides the following implementations in different modules:

- DiffSharp.AD.Forward
- DiffSharp.AD.Forward2
- DiffSharp.AD.ForwardG
- DiffSharp.AD.ForwardGH
- DiffSharp.AD.ForwardN
- DiffSharp.AD.Reverse
- DiffSharp.Numerical
- DiffSharp.Symbolic

For brief explanations of these implementations, please refer to the [Forward AD](gettingstarted-forwardad.html), [Reverse AD](gettingstarted-reversead.html), [Numerical Differentiation](gettingstarted-numericaldifferentiation.html), and [Symbolic Differentiation](gettingstarted-symbolicdifferentiation.html) pages.

Operations and Variants
-----------------------

The operations summarized in the above table have _prime-suffixed_ variants (e.g. **diff** and **diff'** ) that return tuples of (_the value of the original function_, _the value of the desired operation_). This is advantageous in the majority of AD operations, since the original function value would have been already computed as a by-product of AD computations, providing a performance advantage. 
*)

// Use forward AD
open DiffSharp.AD.Forward

// Derivative of Sin(Sqrt(x)) at x = 2
let a = diff (fun x -> sin (sqrt x)) 2.

// (Original value, derivative) of Sin(Sqrt(x)) at x = 2
let b, c = diff' (fun x -> sin (sqrt x)) 2.

(**
In addition to these, **jacobian** operations have _T-suffixed_ variants returning the transposed version of the Jacobian matrix.

Currently, the library provides the following operations:

- **diff**: First derivative of a scalar-to-scalar function
- **diff'**: Original value and first derivative of a scalar-to-scalar function
- **diff2**: Second derivative of a scalar-to-scalar function
- **diff2'**: Original value and second derivative of a scalar-to-scalar function
- **diff2''**: Original value, first derivative, and second derivative of a scalar-to-scalar function
- **diffn**: N-th derivative of a scalar-to-scalar function
- **diffn'**: Original value and n-th derivative of a scalar-to-scalar function
- **grad**: Gradient of a vector-to-scalar function
- **grad'**: Original value and gradient of a vector-to-scalar function
- **gradv**: Gradient-vector product (directional derivative) of a vector-to-scalar function
- **gradv'**: Original value and gradient-vector product (directional derivative) of a vector-to-scalar function
- **hessian**: Hessian of a vector-to-scalar function
- **hessian'**: Original value and Hessian of a vector-to-scalar function
- **gradhessian**: Gradient and Hessian of a vector-to-scalar function
- **gradhessian'**: Original value, gradient, and Hessian of a vector-to-scalar function
- **laplacian**: Laplacian of a vector-to-scalar function
- **laplacian'**: Original value and Laplacian of a vector-to-scalar function
- **jacobian**: Jacobian of a vector-to-vector function
- **jacobian'**: Original value and Jacobian of a vector-to-vector function
- **jacobianT**: Transposed Jacobian of a vector-to-vector function
- **jacobianT'**: Original value and transposed Jacobian of a vector-to-vector function
- **jacobianv**: Jacobian-vector product of a vector-to-vector function
- **jacobianv'**: Original value and Jacobian-vector product of a vector-to-vector function

Vector and Matrix versus float[] and float[,]
---------------------------------------------

When a differentiation module such as **DiffSharp.AD.Forward** is opened, the default operations involving vectors or matrices handle these via **float[]** and **float[,]** arrays.

*)

open DiffSharp.AD.Forward

// Gradient of a vector-to-scalar function
// g1: float[] -> float[]
// i.e. take the function arguments as a float[] and return the gradient as a float[]
let g1 = grad (fun x -> sin (x.[0] * x.[1]))

(**

In addition to this, every module provides a **Vector** submodule containing versions of the same differentiation operators using the **Vector** and **Matrix** types instead of **float[]** and **float[,]**. 

*)

open DiffSharp.AD.Forward.Vector

// Gradient of a vector-to-scalar function
// g2: Vector -> Vector
// i.e. take the function arguments as a Vector and return the gradient as a Vector
let g2 = grad (fun x -> sin (x.[0] * x.[1]))
