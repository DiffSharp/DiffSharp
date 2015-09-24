
(**
Benchmarks
==========

The following tables present the running times of differentiation operations for a set of benchmark cases. For descriptions of the differentiation operations, please refer to [API Overview](api-overview.html).

The values are normalized with respect to the running time of the original function corresponding to each column. Operations **diffn** and **diffn'** are used with $n=2$, for enabling comparisons with **diff2** and **diff2'**. 

The benchmark functions are:

$$$
 f_{ss}(x) = l_{100}\; ,

where $l_{n + 1} = 4 l_{n} (1 - l_n)$ and $l_1 = x$ (the [logistic map](https://en.wikipedia.org/wiki/Logistic_map)), for the scalar-to-scalar case;

$$$
 f_{vs}(\mathbf{x}) = \mathbf{x}^{\textrm{T}} \left(\log \frac{\textbf{x}}{2}\right)\; ,

where $\log$ and division by scalar is applied element-wise to vector $\mathbf{x}$, for the vector-to-scalar case; and

$$$
 f_{vv}(\mathbf{x}) = \left( \mathbf{x}^{\textrm{T}}\left(\log \frac{\textbf{x}}{2}\right),\, \textrm{sum}\{\exp\left(\sin\textbf{x}\right)\},\, \textrm{sum}\{\exp\left(\cos\textbf{x}\right)\}\right)\;

 for the vector-to-vector case.

The running times are averaged over half a million calls to each operation, on a PC with an Intel Core i7-4785T 2.2 GHz CPU and 16 GB RAM, running Windows 10 Build 10240 and .NET Framework 4.6.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-gik2{font-size:10px;font-family:serif !important;;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-ieqq{font-size:10px;font-family:serif !important;;background-color:#e4ffb3;text-align:center}
.tg .tg-habg{font-weight:bold;font-size:10px;font-family:serif !important;}
.tg .tg-g4y3{font-size:10px;font-family:serif !important;}
.tg .tg-x05y{font-size:10px;font-family:serif !important;;background-color:#ffffc7;text-align:center}
.tg .tg-8xte{font-size:10px;font-family:serif !important;;background-color:#ecf4ff;text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-g4y3"></th>
    <th class="tg-x05y">diff </th>
    <th class="tg-x05y">diff2 </th>
    <th class="tg-x05y">diffn </th>
    <th class="tg-ieqq">grad </th>
    <th class="tg-ieqq">gradv</th>
    <th class="tg-ieqq">hessian </th>
    <th class="tg-ieqq">hessianv</th>
    <th class="tg-ieqq">gradhessian </th>
    <th class="tg-ieqq">gradhessianv</th>
    <th class="tg-ieqq">laplacian </th>
    <th class="tg-gik2">jacobian </th>
    <th class="tg-8xte">jacobianv</th>
    <th class="tg-8xte">jacobianT</th>
    <th class="tg-8xte">jacobianTv</th>
  </tr>
  <tr>
    <td class="tg-habg">AD</td>
    <td class="tg-x05y">3.83</td>
    <td class="tg-x05y">9.71</td>
    <td class="tg-x05y">9.33</td>
    <td class="tg-ieqq">12.23</td>
    <td class="tg-ieqq">2.19</td>
    <td class="tg-ieqq">63.84</td>
    <td class="tg-ieqq">7.33</td>
    <td class="tg-ieqq">63.15</td>
    <td class="tg-ieqq">7.11</td>
    <td class="tg-ieqq">64.47</td>
    <td class="tg-8xte">10.97</td>
    <td class="tg-8xte">2.47</td>
    <td class="tg-8xte">11.97</td>
    <td class="tg-8xte">3.17</td>
  </tr>
  <tr>
    <td class="tg-habg">Numerical</td>
    <td class="tg-x05y">3.24</td>
    <td class="tg-x05y">3.78</td>
    <td class="tg-x05y"></td>
    <td class="tg-ieqq">51.21</td>
    <td class="tg-ieqq">10.30</td>
    <td class="tg-ieqq">528.24</td>
    <td class="tg-ieqq">96.71</td>
    <td class="tg-ieqq">518.38</td>
    <td class="tg-ieqq">108.08</td>
    <td class="tg-ieqq">581.75</td>
    <td class="tg-gik2">38.31</td>
    <td class="tg-8xte">3.57</td>
    <td class="tg-8xte">20.26</td>
    <td class="tg-8xte"></td>
  </tr>
  <tr style="visibility:hidden">
    <td class="tg-g4y3"></td>
    <td class="tg-x05y"></td>
    <td class="tg-x05y"></td>
    <td class="tg-x05y"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-ieqq"></td>
    <td class="tg-8xte"></td>
    <td class="tg-8xte"></td>
    <td class="tg-8xte"></td>
    <td class="tg-8xte"></td>
  </tr>
  <tr>
    <td class="tg-g4y3"></td>
    <td class="tg-x05y">diff'</td>
    <td class="tg-x05y">diff2'</td>
    <td class="tg-x05y">diffn'</td>
    <td class="tg-ieqq">grad'</td>
    <td class="tg-ieqq">gradv'</td>
    <td class="tg-ieqq">hessian'</td>
    <td class="tg-ieqq">hessianv'</td>
    <td class="tg-ieqq">gradhessian'</td>
    <td class="tg-ieqq">gradhessianv'</td>
    <td class="tg-ieqq">laplacian'</td>
    <td class="tg-8xte">jacobian'</td>
    <td class="tg-8xte">jacobianv'</td>
    <td class="tg-8xte">jacobianT'</td>
    <td class="tg-8xte">jacobianTv'</td>
  </tr>
  <tr>
    <td class="tg-habg">AD</td>
    <td class="tg-x05y">3.44</td>
    <td class="tg-x05y">12.56</td>
    <td class="tg-x05y">10.12</td>
    <td class="tg-ieqq">3.36</td>
    <td class="tg-ieqq">2.22</td>
    <td class="tg-ieqq">64.97</td>
    <td class="tg-ieqq">6.79</td>
    <td class="tg-ieqq">63.22</td>
    <td class="tg-ieqq">7.33</td>
    <td class="tg-ieqq">64.41</td>
    <td class="tg-8xte">10.96</td>
    <td class="tg-8xte">2.39</td>
    <td class="tg-8xte">11.78</td>
    <td class="tg-8xte">3.14</td>
  </tr>
  <tr>
    <td class="tg-habg">Numerical</td>
    <td class="tg-x05y">4.09</td>
    <td class="tg-x05y">5.01</td>
    <td class="tg-x05y"></td>
    <td class="tg-ieqq">45.13</td>
    <td class="tg-ieqq">12.11</td>
    <td class="tg-ieqq">521.20</td>
    <td class="tg-ieqq">96.19</td>
    <td class="tg-ieqq">504.64</td>
    <td class="tg-ieqq">107.55</td>
    <td class="tg-ieqq">579.96</td>
    <td class="tg-8xte">38.17</td>
    <td class="tg-8xte">4.60</td>
    <td class="tg-8xte">20.08</td>
    <td class="tg-8xte"></td>
  </tr>
</table>

<br>

Running Benchmarks on Your Machine
----------------------------------

If you would like to run the benchmarks on your own machine, you can use the **dsbench** command line tool distributed together with the latest release <a href="https://github.com/DiffSharp/DiffSharp/releases">on GitHub</a>.

<div class="row">
    <div class="span6">
        <img src="img/benchmarks.png" alt="Chart" style="width:569px"/>
    </div>
</div>
*)

