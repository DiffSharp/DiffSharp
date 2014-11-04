
(**
Benchmarks
==========

The following tables present the result of an experiment measuring the running times of the operations in the library. For descriptions of these operations, please refer to [API Overview](api-overview.html).

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-eugx{background-color:#e4ffb3;text-align:center}
.tg .tg-5y5n{background-color:#ecf4ff}
.tg .tg-a1rn{background-color:#ffffc7}
.tg .tg-e3zv{font-weight:bold}
.tg .tg-tfw5{background-color:#ffffc7;text-align:center}
.tg .tg-7nfe{background-color:#ecf4ff;text-align:right}
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
    <th class="tg-lgsi">diff </th>
    <th class="tg-lgsi">diff2 </th>
    <th class="tg-lgsi">diffn </th>
    <th class="tg-u986">grad </th>
    <th class="tg-2sn5">gradv</th>
    <th class="tg-u986">hessian </th>
    <th class="tg-u986">gradhessian </th>
    <th class="tg-u986">laplacian </th>
    <th class="tg-40di">jacobian </th>
    <th class="tg-5y5n">jacobianv</th>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Forward</td>
    <td class="tg-gkzk">3.83</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">20.50</td>
    <td class="tg-eugx">9.00</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-uy90">33.88</td>
    <td class="tg-ci37">5.10</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Forward2</td>
    <td class="tg-gkzk">5.00</td>
    <td class="tg-gkzk">5.00</td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">27.00</td>
    <td class="tg-eugx">11.00</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es">27.50</td>
    <td class="tg-uy90">36.25</td>
    <td class="tg-ci37">6.10</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.ForwardG</td>
    <td class="tg-gkzk">5.50</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">27.00</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-uy90">23.25</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.ForwardGH</td>
    <td class="tg-gkzk">21.67</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">130.00</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es">130.00</td>
    <td class="tg-v6es">135.50</td>
    <td class="tg-v6es">138.50</td>
    <td class="tg-uy90">70.00</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardN</td>
    <td class="tg-tfw5">19.67</td>
    <td class="tg-tfw5">52.33</td>
    <td class="tg-tfw5">55.50</td>
    <td class="tg-eugx">191.50</td>
    <td class="tg-eugx">66.00</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">490.00</td>
    <td class="tg-ci37">114.38</td>
    <td class="tg-ci37">32.80</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Reverse</td>
    <td class="tg-gkzk">9.50</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">18.00</td>
    <td class="tg-eugx"></td>
    <td class="tg-v6es">215.00</td>
    <td class="tg-v6es">232.00</td>
    <td class="tg-v6es">235.00</td>
    <td class="tg-uy90">40.25</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.Numerical</td>
    <td class="tg-gkzk">2.17</td>
    <td class="tg-gkzk">3.00</td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">27.00</td>
    <td class="tg-eugx">10.50</td>
    <td class="tg-v6es">310.00</td>
    <td class="tg-v6es">265.00</td>
    <td class="tg-v6es">266.5</td>
    <td class="tg-uy90">46.13</td>
    <td class="tg-ci37">6.30</td>
  </tr>
  <tr style="visibility:hidden">
    <td class="tg-031e"></td>
    <td class="tg-a1rn"></td>
    <td class="tg-a1rn"></td>
    <td class="tg-a1rn"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-eugx"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-5y5n"></td>
    <td class="tg-7nfe"></td>
  </tr>
  <tr>
    <td class="tg-031e"></td>
    <td class="tg-a1rn">diff'</td>
    <td class="tg-a1rn">diff2'</td>
    <td class="tg-a1rn">diffn'</td>
    <td class="tg-2sn5">grad'</td>
    <td class="tg-eugx">gradv'</td>
    <td class="tg-2sn5">hessian'</td>
    <td class="tg-2sn5">gradhessian'</td>
    <td class="tg-2sn5">laplacian'</td>
    <td class="tg-5y5n">jacobian'</td>
    <td class="tg-7nfe">jacobianv'</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.Forward</td>
    <td class="tg-tfw5">3.83</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">18.00</td>
    <td class="tg-eugx">5.50</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-ci37">33.38</td>
    <td class="tg-ci37">5.00</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.Forward2</td>
    <td class="tg-tfw5">5.67</td>
    <td class="tg-tfw5">4.83</td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">24.00</td>
    <td class="tg-eugx">8.50</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">24.50</td>
    <td class="tg-ci37">37.13</td>
    <td class="tg-ci37">6.10</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardG</td>
    <td class="tg-tfw5">6.00</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">26.50</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-ci37">22.75</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardGH</td>
    <td class="tg-tfw5">20.50</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">130.50</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">129.50</td>
    <td class="tg-eugx">128.50</td>
    <td class="tg-eugx">136.00</td>
    <td class="tg-ci37">69.38</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardN</td>
    <td class="tg-tfw5">20.33</td>
    <td class="tg-tfw5">53.17</td>
    <td class="tg-tfw5">54.00</td>
    <td class="tg-eugx">188.00</td>
    <td class="tg-eugx">60.00</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">478.00</td>
    <td class="tg-ci37">113.88</td>
    <td class="tg-ci37">33.80</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.Reverse</td>
    <td class="tg-tfw5">8.67</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">15.00</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">215.00</td>
    <td class="tg-eugx">221.00</td>
    <td class="tg-eugx">229.50</td>
    <td class="tg-ci37">39.75</td>
    <td class="tg-ci37"></td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.Numerical</td>
    <td class="tg-tfw5">3.17</td>
    <td class="tg-tfw5">4.67</td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">22.50</td>
    <td class="tg-eugx">10.50</td>
    <td class="tg-eugx">256.00</td>
    <td class="tg-eugx">259.50</td>
    <td class="tg-eugx">262.50</td>
    <td class="tg-ci37">39.50</td>
    <td class="tg-ci37">7.50</td>
  </tr>
</table>

<br>

The values are normalized with respect to the running time of the original function corresponding to each column. Operations **diffn** and **diffn'** are used with $n=2$.

The used functions were $ f(x) = (\sin \sqrt{x + 2}) ^ 3$ for the scalar-to-scalar case, $ f(x,y,z) = (x\;\sqrt{y + z}\;\log z) ^ y $ for the vector-to-scalar case, and $f(x,y,z) = (\sin{x ^ y}, \sqrt{y + 2}, \log{xz}) $ for the vector-to-vector case.

The running times were measured using [**Process.TotalProcessorTime**](http://msdn.microsoft.com/en-us/library/system.diagnostics.process.totalprocessortime(v=vs.110).aspx), averaged over a million calls to each operation, on a PC with an Intel Core i7-4510U 2.0 GHz CPU and 16 GB RAM, running Windows 8.1 and .NET Framework 4.5.1.
*)
