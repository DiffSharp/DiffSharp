(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
API Overview
===============

The following table gives an overview of the differentiation API provided by the DiffSharp library.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-kr94{font-size:12px;text-align:center}
.tg .tg-k6pi{font-size:12px}
</style>
<table class="tg">
  <tr>
    <th class="tg-k6pi"></th>
    <th class="tg-k6pi">diff</th>
    <th class="tg-k6pi">diff2</th>
    <th class="tg-k6pi">diffn</th>
    <th class="tg-k6pi">diffdir</th>
    <th class="tg-k6pi">grad</th>
    <th class="tg-k6pi">hessian</th>
    <th class="tg-k6pi">gradhessian</th>
    <th class="tg-k6pi">jacobian</th>
    <th class="tg-k6pi">laplacian</th>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.AD.Forward</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.AD.ForwardDoublet</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.AD.ForwardLazy</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.AD.ForwardTriplet</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.AD.ForwardTwice</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.AD.Reverse</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">XA</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94"></td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.Numerical</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94"></td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">A</td>
    <td class="tg-kr94">A</td>
  </tr>
  <tr>
    <td class="tg-k6pi">DiffSharp.Symbolic</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
    <td class="tg-kr94">X</td>
  </tr>
</table>

**X**: Exact value

**A**: Numerical approximation

**XA**: Exact gradient, approximated Hessian

Operations and Variations
--------------------

The operations summarized above have _'-suffixed_ varieties that return a 2-tuple of (_value of original function_, _value of desired operation_). This is advantageous in the majority of AD computations, since the original function value has been already computed during AD computations, providing a performance advantage. 
*)

// Use forward mode AD
open DiffSharp.AD.Forward

// A scalar-to-scalar function
let f x = 
    sin (sqrt x)

// Derivative of f at a point
let a = diff f 2.

// (Original value, derivative) of f at a point
let (y, y') = diff' f 2.

(**
In addition to these, **jacobian** operations have _T-suffixed varieties_ returning the transposed version of the Jacobian matrix.

Currently, the library provides the following operations:

- **diff**: First derivative of a scalar-to-scalar function
- **diff'**: Original value and first derivative of a scalar-to-scalar function
- **diff2**: Second derivative of a scalar-to-scalar function
- **diff2'**: Original value and second derivative of a scalar-to-scalar function
- **diffn**: N-th derivative of a scalar-to-scalar function
- **diffn'**: Original value and n-th derivative of a scalar-to-scalar function
- **diffdir**: Directional derivative of a vector-to-scalar function
- **diffdir'**: Original value and directional derivative of a vector-to-scalar function
- **grad**: Gradient of a vector-to-scalar function
- **grad'**: Original value and gradient of a vector-to-scalar function
- **hessian**: Hessian of a vector-to-scalar function
- **hessian'**: Original value and Hessian of a vector-to-scalar function
- **gradhessian**: Gradient and Hessian of a vector-to-scalar function
- **gradhessian'**: Original value, gradient, and Hessian of a vector-to-scalar function
- **jacobian**: Jacobian of a vector-to-vector function
- **jacobian'**: Original value and Jacobian of a vector-to-vector function
- **jacobianT**: Transposed Jacobian of a vector-to-vector function
- **jacobianT'**: Original value and transposed Jacobian of a vector-to-vector function
- **laplacian**: Laplacian of a vector-to-scalar function
- **laplacian'**: Original value and Laplacian of a vector-to-scalar function
*)
