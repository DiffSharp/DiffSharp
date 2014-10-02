
(**
Benchmarks
==========

The following tables present the result of an experiment measuring the running times of the operations in the library. For descriptions of these operations, please refer to [API Overview](api-overview.html).

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-eugx{background-color:#e4ffb3;text-align:center}
.tg .tg-5y5n{background-color:#ecf4ff}
.tg .tg-a1rn{background-color:#ffffc7}
.tg .tg-e3zv{font-weight:bold}
.tg .tg-tfw5{background-color:#ffffc7;text-align:center}
.tg .tg-a0td{font-size:100%}
.tg .tg-lgsi{font-size:100%;background-color:#ffffc7}
.tg .tg-u986{font-size:100%;background-color:#e4ffb3}
.tg .tg-40di{font-size:100%;background-color:#ecf4ff;color:#000000}
.tg .tg-dyge{font-weight:bold;font-size:100%}
.tg .tg-gkzk{font-size:100%;background-color:#ffffc7;text-align:center}
.tg .tg-v6es{font-size:100%;background-color:#e4ffb3;text-align:center}
.tg .tg-uy90{font-size:100%;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-ci37{background-color:#ecf4ff;text-align:center}
.tg .tg-2sn5{background-color:#e4ffb3}
</style>
<table class="tg">
  <tr>
    <th class="tg-a0td"></th>
    <th class="tg-lgsi">diff </th>
    <th class="tg-lgsi">diff2 </th>
    <th class="tg-lgsi">diffn </th>
    <th class="tg-u986">diffdir </th>
    <th class="tg-u986">grad </th>
    <th class="tg-u986">hessian </th>
    <th class="tg-u986">gradhessian </th>
    <th class="tg-u986">laplacian </th>
    <th class="tg-40di">jacobian </th>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Forward</td>
    <td class="tg-gkzk">1.71</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">1.06</td>
    <td class="tg-v6es">2.38</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-uy90">13.93</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Forward2</td>
    <td class="tg-gkzk">2.71</td>
    <td class="tg-gkzk">2.86</td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">1.31</td>
    <td class="tg-v6es">3.38</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es">3.42</td>
    <td class="tg-uy90">16.00</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.ForwardG</td>
    <td class="tg-gkzk">2.71</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es">2.27</td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es"></td>
    <td class="tg-uy90">6.33</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.ForwardGH</td>
    <td class="tg-gkzk">16.29</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es">12.62</td>
    <td class="tg-v6es">14.96</td>
    <td class="tg-v6es">13.50</td>
    <td class="tg-v6es">14.65</td>
    <td class="tg-uy90">20.53</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardN</td>
    <td class="tg-tfw5">9.86</td>
    <td class="tg-tfw5">27.71</td>
    <td class="tg-tfw5">28.57</td>
    <td class="tg-eugx">4.85</td>
    <td class="tg-eugx">14.85</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">38.42</td>
    <td class="tg-ci37">30.13</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.AD.Reverse</td>
    <td class="tg-gkzk">4.29</td>
    <td class="tg-gkzk"></td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es"></td>
    <td class="tg-v6es">1.73</td>
    <td class="tg-v6es">21.58</td>
    <td class="tg-v6es">17.58</td>
    <td class="tg-v6es">20.19</td>
    <td class="tg-uy90">8.80</td>
  </tr>
  <tr>
    <td class="tg-dyge">DiffSharp.Numerical</td>
    <td class="tg-gkzk">1.06</td>
    <td class="tg-gkzk">1.14</td>
    <td class="tg-gkzk"></td>
    <td class="tg-v6es">1.04</td>
    <td class="tg-v6es">2.65</td>
    <td class="tg-v6es">25.92</td>
    <td class="tg-v6es">21.27</td>
    <td class="tg-v6es">23.42</td>
    <td class="tg-uy90">21.33</td>
  </tr>
  <tr style="visibility:hidden">
    <td class="tg-031e"></td>
    <td class="tg-a1rn"></td>
    <td class="tg-a1rn"></td>
    <td class="tg-a1rn"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-2sn5"></td>
    <td class="tg-5y5n"></td>
  </tr>
  <tr>
    <td class="tg-031e"></td>
    <td class="tg-a1rn">diff'</td>
    <td class="tg-a1rn">diff2'</td>
    <td class="tg-a1rn">diffn'</td>
    <td class="tg-2sn5">diffdir'</td>
    <td class="tg-2sn5">grad'</td>
    <td class="tg-2sn5">hessian'</td>
    <td class="tg-2sn5">gradhessian'</td>
    <td class="tg-2sn5">laplacian'</td>
    <td class="tg-5y5n">jacobian'</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.Forward</td>
    <td class="tg-tfw5">1.43</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">1.03</td>
    <td class="tg-eugx">2.12</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-ci37">13.8</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.Forward2</td>
    <td class="tg-tfw5">3.29</td>
    <td class="tg-tfw5">2.71</td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">1.08</td>
    <td class="tg-eugx">3.19</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">3.19</td>
    <td class="tg-ci37">15.6</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardG</td>
    <td class="tg-tfw5">2.86</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">2.19</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-ci37">5.93</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardGH</td>
    <td class="tg-tfw5">15.14</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">12.85</td>
    <td class="tg-eugx">12.96</td>
    <td class="tg-eugx">13.12</td>
    <td class="tg-eugx">14.42</td>
    <td class="tg-ci37">19.93</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.ForwardN</td>
    <td class="tg-tfw5">10.29</td>
    <td class="tg-tfw5">28.00</td>
    <td class="tg-tfw5">29.00</td>
    <td class="tg-eugx">4.54</td>
    <td class="tg-eugx">14.19</td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">37.46</td>
    <td class="tg-ci37">30.47</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.AD.Reverse</td>
    <td class="tg-tfw5">3.43</td>
    <td class="tg-tfw5"></td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx"></td>
    <td class="tg-eugx">1.35</td>
    <td class="tg-eugx">17.62</td>
    <td class="tg-eugx">17.08</td>
    <td class="tg-eugx">19.31</td>
    <td class="tg-ci37">8.4</td>
  </tr>
  <tr>
    <td class="tg-e3zv">DiffSharp.Numerical</td>
    <td class="tg-tfw5">1.29</td>
    <td class="tg-tfw5">1.57</td>
    <td class="tg-tfw5"></td>
    <td class="tg-eugx">1.15</td>
    <td class="tg-eugx">2.19</td>
    <td class="tg-eugx">22.19</td>
    <td class="tg-eugx">20.92</td>
    <td class="tg-eugx">23.38</td>
    <td class="tg-ci37">39.50</td>
  </tr>
</table>

<br>

The values are normalized with respect to the running time of the original function corresponding to each column.

The used functions were $ f(x) = (\sin \sqrt{x - 2}) ^ 3$ as the scalar-to-scalar, $ f(x,y,z) = (x\;\sqrt{y - z}\;\log z) ^ y $ as the vector-to-scalar, and $f(x,y,z) = (\sin{x ^ y}, \sqrt{y - 2}, \log{xz}) $ as the vector-to-vector.

The running times were measured using [**Process.TotalProcessorTime**](http://msdn.microsoft.com/en-us/library/system.diagnostics.process.totalprocessortime(v=vs.110).aspx), averaged over a million calls to each operation, on a PC with an Intel Core i7-4510U 2.0 GHz CPU and 16 GB RAM, running Windows 8.1 and .NET Framework 4.5.1.
*)
