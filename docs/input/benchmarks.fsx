
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
    <td class="tg-sgic">3.15</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">13.45</td>
    <td class="tg-71xk">4.80</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-sfug">31.71</td>
    <td class="tg-5nhg">5.34</td>
    <td class="tg-5nhg">26.56</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward2</td>
    <td class="tg-sgic">3.94</td>
    <td class="tg-sgic">3.94</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">15.86</td>
    <td class="tg-71xk">5.70</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">16.43</td>
    <td class="tg-sfug">35.12</td>
    <td class="tg-5nhg">5.90</td>
    <td class="tg-5nhg">29.42</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardG</td>
    <td class="tg-sgic">5.35</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">22.67</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-sfug">24.89</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">30.91</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardGH</td>
    <td class="tg-sgic">18.09</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">97.05</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">97.56</td>
    <td class="tg-71xk">97.84</td>
    <td class="tg-71xk">103.75</td>
    <td class="tg-sfug">81.72</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">86.54</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardN</td>
    <td class="tg-sgic">15.17</td>
    <td class="tg-sgic">39.24</td>
    <td class="tg-sgic">39.85</td>
    <td class="tg-71xk">104.61</td>
    <td class="tg-71xk">34.44</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">326.68</td>
    <td class="tg-5nhg">101.36</td>
    <td class="tg-5nhg">26.89</td>
    <td class="tg-5nhg">95.12</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Reverse</td>
    <td class="tg-sgic">7.16</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">10.86</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">92.27</td>
    <td class="tg-71xk">92.68</td>
    <td class="tg-71xk">97.69</td>
    <td class="tg-sfug">39.02</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">44.55</td>
    <td class="tg-5nhg">9.19</td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Numerical</td>
    <td class="tg-sgic">2.37</td>
    <td class="tg-sgic">3.32</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">15.73</td>
    <td class="tg-71xk">6.93</td>
    <td class="tg-71xk">107.27</td>
    <td class="tg-71xk">107.94</td>
    <td class="tg-71xk">112.32</td>
    <td class="tg-sfug">45.76</td>
    <td class="tg-5nhg">7.84</td>
    <td class="tg-5nhg">40.63</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Symbolic</td>
    <td class="tg-sgic">40k</td>
    <td class="tg-sgic">191k</td>
    <td class="tg-sgic">191k</td>
    <td class="tg-71xk">360k</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">3.09M</td>
    <td class="tg-71xk">3.51M</td>
    <td class="tg-71xk">1.77M</td>
    <td class="tg-5nhg">333k</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">332k</td>
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
    <td class="tg-sgic">3.05</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">13.42</td>
    <td class="tg-71xk">4.85</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-5nhg">32.24</td>
    <td class="tg-5nhg">5.22</td>
    <td class="tg-5nhg">26.63</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward2</td>
    <td class="tg-sgic">3.95</td>
    <td class="tg-sgic">4.10</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">16.03</td>
    <td class="tg-71xk">5.77</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">16.15</td>
    <td class="tg-5nhg">34.70</td>
    <td class="tg-5nhg">5.97</td>
    <td class="tg-5nhg">29.23</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardG</td>
    <td class="tg-sgic">5.16</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">23.22</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-5nhg">24.98</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">30.67</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardGH</td>
    <td class="tg-sgic">18.46</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">102.72</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">101.97</td>
    <td class="tg-71xk">102.02</td>
    <td class="tg-71xk">103.66</td>
    <td class="tg-5nhg">81.35</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">87.43</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardN</td>
    <td class="tg-sgic">15.66</td>
    <td class="tg-sgic">40.49</td>
    <td class="tg-sgic">41.43</td>
    <td class="tg-71xk">106.37</td>
    <td class="tg-71xk">35.50</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">324.50</td>
    <td class="tg-5nhg">102.40</td>
    <td class="tg-5nhg">27.28</td>
    <td class="tg-5nhg">96.11</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Reverse</td>
    <td class="tg-sgic">6.86</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">10.97</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">97.05</td>
    <td class="tg-71xk">96.99</td>
    <td class="tg-71xk">97.72</td>
    <td class="tg-5nhg">37.86</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">44.58</td>
    <td class="tg-5nhg">9.06</td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Numerical</td>
    <td class="tg-sgic">3.24</td>
    <td class="tg-sgic">4.18</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">15.13</td>
    <td class="tg-71xk">7.84</td>
    <td class="tg-71xk">112.66</td>
    <td class="tg-71xk">111.65</td>
    <td class="tg-71xk">112.19</td>
    <td class="tg-5nhg">46.37</td>
    <td class="tg-5nhg">9.00</td>
    <td class="tg-5nhg">40.55</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Symbolic</td>
    <td class="tg-sgic">40k</td>
    <td class="tg-sgic">192k</td>
    <td class="tg-sgic">195k</td>
    <td class="tg-71xk">370k</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">3.17M</td>
    <td class="tg-71xk">3.55M</td>
    <td class="tg-71xk">1.77M</td>
    <td class="tg-5nhg">333k</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">333k</td>
    <td class="tg-5nhg"></td>
  </tr>
</table>

<br>

The values are normalized with respect to the running time of the original function corresponding to each column. Operations **diffn** and **diffn'** are used with $n=2$.

The used functions were $ f(x) = (\sin \sqrt{x + 2}) ^ 3$ for the scalar-to-scalar case, $ f(x,y,z) = (x\;\sqrt{y + z}\;\log z) ^ y $ for the vector-to-scalar case, and $f(x,y,z) = (\sin{x ^ y}, \sqrt{y + 2}, \log{xz}) $ for the vector-to-vector case.

The running times were measured using [**Process.TotalProcessorTime**](http://msdn.microsoft.com/en-us/library/system.diagnostics.process.totalprocessortime(v=vs.110).aspx), averaged over a million calls to each operation, on a PC with an Intel Core i7-4510U 2.0 GHz CPU and 16 GB RAM, running Windows 8.1 and .NET Framework 4.5.1.
*)
