
(**
Benchmarks
==========

The following tables present the result of an experiment measuring the running times of the operations in the library. For descriptions of these operations, please refer to [API Overview](api-overview.html).

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-5nhg{font-size:11px;background-color:#ecf4ff;text-align:center}
.tg .tg-0e45{font-size:11px}
.tg .tg-sgic{font-size:11px;background-color:#ffffc7;text-align:center}
.tg .tg-71xk{font-size:11px;background-color:#e4ffb3;text-align:center}
.tg .tg-sfug{font-size:11px;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-nl5m{font-weight:bold;font-size:11px}
</style>
<table class="tg">
  <tr>
    <th class="tg-0e45"></th>
    <th class="tg-sgic">diff </th>
    <th class="tg-sgic">diff2 </th>
    <th class="tg-sgic">diffn </th>
    <th class="tg-71xk">grad </th>
    <th class="tg-71xk">gradv</th>
    <th class="tg-71xk">hessian </th>
    <th class="tg-71xk">gradhessian </th>
    <th class="tg-71xk">laplacian </th>
    <th class="tg-sfug">jacobian </th>
    <th class="tg-5nhg">jacobianv</th>
    <th class="tg-5nhg">jacobianT</th>
    <th class="tg-5nhg">jacobianTv</th>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward</td>
    <td class="tg-sgic">2.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">8.00</td>
    <td class="tg-71xk">3.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-sfug">27.00</td>
    <td class="tg-5nhg">4.00</td>
    <td class="tg-5nhg">23.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward2</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">10.00</td>
    <td class="tg-71xk">4.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">10.00</td>
    <td class="tg-sfug">29.00</td>
    <td class="tg-5nhg">6.00</td>
    <td class="tg-5nhg">24.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardG</td>
    <td class="tg-sgic">4.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">11.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-sfug">18.00</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">23.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardGH</td>
    <td class="tg-sgic">13.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">47.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">47.00</td>
    <td class="tg-71xk">48.00</td>
    <td class="tg-71xk">48.00</td>
    <td class="tg-sfug">55.00</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">59.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardN</td>
    <td class="tg-sgic">11.00</td>
    <td class="tg-sgic">30.00</td>
    <td class="tg-sgic">30.00</td>
    <td class="tg-71xk">67.00</td>
    <td class="tg-71xk">22.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">203.00</td>
    <td class="tg-5nhg">85.00</td>
    <td class="tg-5nhg">23.00</td>
    <td class="tg-5nhg">81.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Reverse</td>
    <td class="tg-sgic">6.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">7.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">55.00</td>
    <td class="tg-71xk">55.00</td>
    <td class="tg-71xk">56.00</td>
    <td class="tg-sfug">31.00</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">37.00</td>
    <td class="tg-5nhg">8.00</td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Numerical</td>
    <td class="tg-sgic">1.00</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">7.00</td>
    <td class="tg-71xk">3.00</td>
    <td class="tg-71xk">57.00</td>
    <td class="tg-71xk">57.00</td>
    <td class="tg-71xk">57.00</td>
    <td class="tg-sfug">35.00</td>
    <td class="tg-5nhg">5.00</td>
    <td class="tg-5nhg">30.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr style="visibility:hidden">
    <td class="tg-0e45"></td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-0e45"></td>
    <td class="tg-sgic">diff'</td>
    <td class="tg-sgic">diff2'</td>
    <td class="tg-sgic">diffn'</td>
    <td class="tg-71xk">grad'</td>
    <td class="tg-71xk">gradv'</td>
    <td class="tg-71xk">hessian'</td>
    <td class="tg-71xk">gradhessian'</td>
    <td class="tg-71xk">laplacian'</td>
    <td class="tg-5nhg">jacobian'</td>
    <td class="tg-5nhg">jacobianv'</td>
    <td class="tg-5nhg">jacobianT'</td>
    <td class="tg-5nhg">jacobianTv'</td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward</td>
    <td class="tg-sgic">2.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">8.00</td>
    <td class="tg-71xk">3.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-5nhg">26.00</td>
    <td class="tg-5nhg">5.00</td>
    <td class="tg-5nhg">22.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward2</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">11.00</td>
    <td class="tg-71xk">4.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">10.00</td>
    <td class="tg-5nhg">30.00</td>
    <td class="tg-5nhg">6.00</td>
    <td class="tg-5nhg">26.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardG</td>
    <td class="tg-sgic">4.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">11.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-5nhg">18.00</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">22.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardGH</td>
    <td class="tg-sgic">12.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">47.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">48.00</td>
    <td class="tg-71xk">49.00</td>
    <td class="tg-71xk">47.00</td>
    <td class="tg-5nhg">54.00</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">59.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardN</td>
    <td class="tg-sgic">11.00</td>
    <td class="tg-sgic">30.00</td>
    <td class="tg-sgic">32.00</td>
    <td class="tg-71xk">68.00</td>
    <td class="tg-71xk">22.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">211.00</td>
    <td class="tg-5nhg">85.00</td>
    <td class="tg-5nhg">24.00</td>
    <td class="tg-5nhg">81.00</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Reverse</td>
    <td class="tg-sgic">6.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">7.00</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">55.00</td>
    <td class="tg-71xk">54.00</td>
    <td class="tg-71xk">57.00</td>
    <td class="tg-5nhg">30.00</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">36.00</td>
    <td class="tg-5nhg">7.00</td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Numerical</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic">3.00</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">8.00</td>
    <td class="tg-71xk">4.00</td>
    <td class="tg-71xk">66.00</td>
    <td class="tg-71xk">57.00</td>
    <td class="tg-71xk">57.00</td>
    <td class="tg-5nhg">35.00</td>
    <td class="tg-5nhg">6.00</td>
    <td class="tg-5nhg">31.00</td>
    <td class="tg-5nhg"></td>
  </tr>
</table>

<br>

The values are normalized with respect to the running time of the original function corresponding to each column. Operations **diffn** and **diffn'** are used with $n=2$.

The used functions were $ f(x) = (\sin \sqrt{x + 2}) ^ 3$ for the scalar-to-scalar case, $ f(x,y,z) = (x\;\sqrt{y + z}\;\log z) ^ y $ for the vector-to-scalar case, and $f(x,y,z) = (\sin{x ^ y}, \sqrt{y + 2}, \log{xz}) $ for the vector-to-vector case.

The running times were measured using [**Process.TotalProcessorTime**](http://msdn.microsoft.com/en-us/library/system.diagnostics.process.totalprocessortime(v=vs.110).aspx), averaged over a million calls to each operation, on a PC with an Intel Core i7-4510U 2.0 GHz CPU and 16 GB RAM, running Windows 8.1 and .NET Framework 4.5.1.
*)
