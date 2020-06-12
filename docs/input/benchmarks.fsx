
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

The running times are averaged over 10000 calls to each operation, with vector size 100, on a PC with an Intel Core i7-4785T 2.2 GHz CPU and 16 GB RAM, running Windows 10 Build 10240 and .NET Framework 4.6. 

Please note that the numbers for multivariate functions are highly dependent on the selected size of the input vector (i.e., independent variables). The [Helmholtz Energy Function](examples-helmholtzenergyfunction.html) page demonstrates how the overhead factors scale as a function of the number of independent variables.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-mk1h{font-size:10px;background-color:#ecf4ff;color:#000000;text-align:center;vertical-align:top}
.tg .tg-xswp{font-size:10px;background-color:#ecf4ff;text-align:center;vertical-align:top}
.tg .tg-klyj{font-weight:bold;font-size:10px;vertical-align:top}
.tg .tg-5sg4{font-size:10px;vertical-align:top}
.tg .tg-9t4f{font-size:10px;background-color:#ffffc7;text-align:center;vertical-align:top}
.tg .tg-h8aj{font-size:10px;background-color:#e4ffb3;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-5sg4"></th>
    <th class="tg-9t4f">diff </th>
    <th class="tg-9t4f">diff2 </th>
    <th class="tg-9t4f">diffn </th>
    <th class="tg-h8aj">grad </th>
    <th class="tg-h8aj">gradv</th>
    <th class="tg-h8aj">hessian </th>
    <th class="tg-h8aj">hessianv</th>
    <th class="tg-h8aj">gradhessian </th>
    <th class="tg-h8aj">gradhessianv</th>
    <th class="tg-h8aj">laplacian </th>
    <th class="tg-mk1h">jacobian </th>
    <th class="tg-xswp">jacobianv</th>
    <th class="tg-xswp">jacobianT</th>
    <th class="tg-xswp">jacobianTv</th>
  </tr>
  <tr>
    <td class="tg-klyj">AD</td>
    <td class="tg-9t4f">3.33</td>
    <td class="tg-9t4f">9.30</td>
    <td class="tg-9t4f">9.21</td>
    <td class="tg-h8aj">4.79</td>
    <td class="tg-h8aj">2.11</td>
    <td class="tg-h8aj">810.22</td>
    <td class="tg-h8aj">9.62</td>
    <td class="tg-h8aj">823.86</td>
    <td class="tg-h8aj">9.55</td>
    <td class="tg-h8aj">814.19</td>
    <td class="tg-xswp">15.10</td>
    <td class="tg-xswp">2.55</td>
    <td class="tg-xswp">15.41</td>
    <td class="tg-xswp">4.53</td>
  </tr>
  <tr>
    <td class="tg-klyj">Numerical</td>
    <td class="tg-9t4f">1.93</td>
    <td class="tg-9t4f">2.78</td>
    <td class="tg-9t4f"></td>
    <td class="tg-h8aj">161.36</td>
    <td class="tg-h8aj">3.49</td>
    <td class="tg-h8aj">16.75k</td>
    <td class="tg-h8aj">322.88</td>
    <td class="tg-h8aj">16.82k</td>
    <td class="tg-h8aj">326.97</td>
    <td class="tg-h8aj">16.75k</td>
    <td class="tg-mk1h">112.54</td>
    <td class="tg-xswp">2.21</td>
    <td class="tg-xswp">111.38</td>
    <td class="tg-xswp"></td>
  </tr>
  <tr style="visibility:hidden">
    <td class="tg-5sg4"></td>
    <td class="tg-9t4f"></td>
    <td class="tg-9t4f"></td>
    <td class="tg-9t4f"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-h8aj"></td>
    <td class="tg-xswp"></td>
    <td class="tg-xswp"></td>
    <td class="tg-xswp"></td>
    <td class="tg-xswp"></td>
  </tr>
  <tr>
    <td class="tg-5sg4"></td>
    <td class="tg-9t4f">diff'</td>
    <td class="tg-9t4f">diff2'</td>
    <td class="tg-9t4f">diffn'</td>
    <td class="tg-h8aj">grad'</td>
    <td class="tg-h8aj">gradv'</td>
    <td class="tg-h8aj">hessian'</td>
    <td class="tg-h8aj">hessianv'</td>
    <td class="tg-h8aj">gradhessian'</td>
    <td class="tg-h8aj">gradhessianv'</td>
    <td class="tg-h8aj">laplacian'</td>
    <td class="tg-xswp">jacobian'</td>
    <td class="tg-xswp">jacobianv'</td>
    <td class="tg-xswp">jacobianT'</td>
    <td class="tg-xswp">jacobianTv'</td>
  </tr>
  <tr>
    <td class="tg-klyj">AD</td>
    <td class="tg-9t4f">3.35</td>
    <td class="tg-9t4f">12.30</td>
    <td class="tg-9t4f">10.12</td>
    <td class="tg-h8aj">4.70</td>
    <td class="tg-h8aj">2.05</td>
    <td class="tg-h8aj">821.31</td>
    <td class="tg-h8aj">9.56</td>
    <td class="tg-h8aj">808.22</td>
    <td class="tg-h8aj">9.66</td>
    <td class="tg-h8aj">810.23</td>
    <td class="tg-xswp">15.05</td>
    <td class="tg-xswp">2.51</td>
    <td class="tg-xswp">15.26</td>
    <td class="tg-xswp">4.50</td>
  </tr>
  <tr>
    <td class="tg-klyj">Numerical</td>
    <td class="tg-9t4f">2.82</td>
    <td class="tg-9t4f">3.71</td>
    <td class="tg-9t4f"></td>
    <td class="tg-h8aj">160.56</td>
    <td class="tg-h8aj">4.66</td>
    <td class="tg-h8aj">16.78k</td>
    <td class="tg-h8aj">323.03</td>
    <td class="tg-h8aj">16.80k</td>
    <td class="tg-h8aj">327.68</td>
    <td class="tg-h8aj">16.78k</td>
    <td class="tg-xswp">111.62</td>
    <td class="tg-xswp">3.20</td>
    <td class="tg-xswp">112.43</td>
    <td class="tg-xswp"></td>
  </tr>
</table>

<br>

The benchmarks given in the above table can be replicated using the benchmarking tool: <pre>dsbench -vsize 100 -r 10000</pre>

Running Benchmarks on Your Machine
----------------------------------

If you would like to run the benchmarks on your own machine, you can use the **dsbench** command line tool distributed together with the latest release <a href="https://github.com/DiffSharp/DiffSharp/releases">on GitHub</a>.

<div class="row">
    <div class="span6">
        <img src="img/benchmarks.png" alt="Chart" style="width:569px"/>
    </div>
</div>

<br>

*)

