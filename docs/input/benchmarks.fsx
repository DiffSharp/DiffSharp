
(**
Benchmarks
==========

The following tables present the running times of the operations in the library for a set of simple test cases. For descriptions of the differentiation operations, please refer to [API Overview](api-overview.html).

The values are normalized with respect to the running time of the original function corresponding to each column. Operations **diffn** and **diffn'** are used with $n=2$, for enabling comparisons with **diff2** and **diff2'**. The used functions are $ f(x) = (\sin \sqrt{x + 2}) ^ 3$ for the scalar-to-scalar case, $ f(x,y,z) = (x\;\sqrt{y + z}\;\log z) ^ y $ for the vector-to-scalar case, and $f(x,y,z) = (\sin{x ^ y}, \sqrt{y + 2}, \log{xz}) $ for the vector-to-vector case.

The running times are averaged over half a million calls to each operation, on a PC with an Intel Core i7-4510U 2.0 GHz CPU and 16 GB RAM, running Windows 8.1 and .NET Framework 4.5.1.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-1r2d{font-size:9px;background-color:#ecf4ff;text-align:center}
.tg .tg-glis{font-size:9px}
.tg .tg-wcxf{font-size:9px;background-color:#ffffc7;text-align:center}
.tg .tg-aycn{font-size:9px;background-color:#e4ffb3;text-align:center}
.tg .tg-wklz{font-size:9px;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-7dqz{font-weight:bold;font-size:9px}
</style>
<table class="tg">
  <tr>
    <th class="tg-glis"></th>
    <th class="tg-wcxf">diff </th>
    <th class="tg-wcxf">diff2 </th>
    <th class="tg-wcxf">diffn </th>
    <th class="tg-aycn">grad </th>
    <th class="tg-aycn">gradv</th>
    <th class="tg-aycn">hessian </th>
    <th class="tg-aycn">hessianv</th>
    <th class="tg-aycn">gradhessian </th>
    <th class="tg-aycn">gradhessianv</th>
    <th class="tg-aycn">laplacian </th>
    <th class="tg-wklz">jacobian </th>
    <th class="tg-1r2d">jacobianv</th>
    <th class="tg-1r2d">jacobianT</th>
    <th class="tg-1r2d">jacobianTv</th>
    <th class="tg-1r2d">jacobianvTv</th>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Forward</td>
    <td class="tg-wcxf">2.48</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">9.76</td>
    <td class="tg-aycn">3.03</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">31.54</td>
    <td class="tg-1r2d">3.18</td>
    <td class="tg-1r2d">24.90</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Forward2</td>
    <td class="tg-wcxf">3.65</td>
    <td class="tg-wcxf">3.59</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">12.30</td>
    <td class="tg-aycn">3.88</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">12.24</td>
    <td class="tg-wklz">32.75</td>
    <td class="tg-1r2d">3.68</td>
    <td class="tg-1r2d">24.39</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardG</td>
    <td class="tg-wcxf">3.27</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">7.08</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">16.50</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">20.41</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardGH</td>
    <td class="tg-wcxf">13.69</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">60.28</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">66.42</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">77.21</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">70.89</td>
    <td class="tg-wklz">53.47</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">61.13</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardN</td>
    <td class="tg-wcxf">16.33</td>
    <td class="tg-wcxf">42.50</td>
    <td class="tg-wcxf">43.73</td>
    <td class="tg-aycn">109.98</td>
    <td class="tg-aycn">35.91</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">363.65</td>
    <td class="tg-1r2d">94.72</td>
    <td class="tg-1r2d">25.84</td>
    <td class="tg-1r2d">93.36</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardReverse</td>
    <td class="tg-wcxf">68.16</td>
    <td class="tg-wcxf">11.94</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">21.16</td>
    <td class="tg-aycn">177.10</td>
    <td class="tg-aycn">81.02</td>
    <td class="tg-aycn">25.22</td>
    <td class="tg-aycn">86.78</td>
    <td class="tg-aycn">22.82</td>
    <td class="tg-aycn">91.83</td>
    <td class="tg-1r2d">165.12</td>
    <td class="tg-1r2d">130.76</td>
    <td class="tg-1r2d">172.05</td>
    <td class="tg-1r2d">44.24</td>
    <td class="tg-1r2d">43.77</td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Reverse</td>
    <td class="tg-wcxf">6.11</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">8.02</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">85.02</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">94.55</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">100.36</td>
    <td class="tg-wklz">39.40</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">46.65</td>
    <td class="tg-1r2d">7.19</td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Numerical</td>
    <td class="tg-wcxf">2.12</td>
    <td class="tg-wcxf">2.99</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">10.14</td>
    <td class="tg-aycn">12.86</td>
    <td class="tg-aycn">90.59</td>
    <td class="tg-aycn">36.94</td>
    <td class="tg-aycn">101.15</td>
    <td class="tg-aycn">42.46</td>
    <td class="tg-aycn">104.35</td>
    <td class="tg-wklz">41.39</td>
    <td class="tg-1r2d">20.54</td>
    <td class="tg-1r2d">41.31</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Compile)</td>
    <td class="tg-wcxf">865.51k</td>
    <td class="tg-wcxf">8.90M</td>
    <td class="tg-wcxf">9.10M</td>
    <td class="tg-aycn">5.80M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">113.89M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">130.26M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">63.15M</td>
    <td class="tg-1r2d">2.79M</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">2.91M</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Use)</td>
    <td class="tg-wcxf">143.53</td>
    <td class="tg-wcxf">53.44</td>
    <td class="tg-wcxf">27.42</td>
    <td class="tg-aycn">35.69k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">216.26k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">288.25k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">102.89k</td>
    <td class="tg-1r2d">18.81k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">20.99k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr style="visibility:hidden">
    <td class="tg-glis"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-glis"></td>
    <td class="tg-wcxf">diff'</td>
    <td class="tg-wcxf">diff2'</td>
    <td class="tg-wcxf">diffn'</td>
    <td class="tg-aycn">grad'</td>
    <td class="tg-aycn">gradv'</td>
    <td class="tg-aycn">hessian'</td>
    <td class="tg-aycn">hessianv'</td>
    <td class="tg-aycn">gradhessian'</td>
    <td class="tg-aycn">gradhessianv'</td>
    <td class="tg-aycn">laplacian'</td>
    <td class="tg-1r2d">jacobian'</td>
    <td class="tg-1r2d">jacobianv'</td>
    <td class="tg-1r2d">jacobianT'</td>
    <td class="tg-1r2d">jacobianTv'</td>
    <td class="tg-1r2d">jacobianvTv'</td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Forward</td>
    <td class="tg-wcxf">2.76</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">11.10</td>
    <td class="tg-aycn">3.26</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-1r2d">26.50</td>
    <td class="tg-1r2d">3.45</td>
    <td class="tg-1r2d">20.59</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Forward2</td>
    <td class="tg-wcxf">3.66</td>
    <td class="tg-wcxf">3.78</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">14.56</td>
    <td class="tg-aycn">4.78</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">13.81</td>
    <td class="tg-1r2d">28.06</td>
    <td class="tg-1r2d">3.99</td>
    <td class="tg-1r2d">22.02</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardG</td>
    <td class="tg-wcxf">3.16</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">8.00</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-1r2d">15.18</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">20.68</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardGH</td>
    <td class="tg-wcxf">13.00</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">66.50</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">68.30</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">67.26</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">78.84</td>
    <td class="tg-1r2d">50.01</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">58.48</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardN</td>
    <td class="tg-wcxf">16.44</td>
    <td class="tg-wcxf">43.45</td>
    <td class="tg-wcxf">44.35</td>
    <td class="tg-aycn">126.22</td>
    <td class="tg-aycn">41.42</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">359.91</td>
    <td class="tg-1r2d">105.48</td>
    <td class="tg-1r2d">26.58</td>
    <td class="tg-1r2d">88.59</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardReverse</td>
    <td class="tg-wcxf">53.38</td>
    <td class="tg-wcxf">11.19</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">23.87</td>
    <td class="tg-aycn">182.47</td>
    <td class="tg-aycn">90.55</td>
    <td class="tg-aycn">24.67</td>
    <td class="tg-aycn">89.23</td>
    <td class="tg-aycn">25.22</td>
    <td class="tg-aycn">97.72</td>
    <td class="tg-1r2d">166.80</td>
    <td class="tg-1r2d">129.13</td>
    <td class="tg-1r2d">158.23</td>
    <td class="tg-1r2d">44.69</td>
    <td class="tg-1r2d">45.58</td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Reverse</td>
    <td class="tg-wcxf">10.30</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">9.22</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">98.95</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">102.29</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">94.74</td>
    <td class="tg-1r2d">40.25</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">49.16</td>
    <td class="tg-1r2d">7.46</td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Numerical</td>
    <td class="tg-wcxf">3.05</td>
    <td class="tg-wcxf">4.04</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">10.57</td>
    <td class="tg-aycn">30.43</td>
    <td class="tg-aycn">102.17</td>
    <td class="tg-aycn">37.70</td>
    <td class="tg-aycn">100.20</td>
    <td class="tg-aycn">44.19</td>
    <td class="tg-aycn">102.59</td>
    <td class="tg-1r2d">46.53</td>
    <td class="tg-1r2d">12.80</td>
    <td class="tg-1r2d">36.12</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Compile)</td>
    <td class="tg-wcxf">904.78k</td>
    <td class="tg-wcxf">10.10M</td>
    <td class="tg-wcxf">10.07M</td>
    <td class="tg-aycn">6.49M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">123.75M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">130.39M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">62.68M</td>
    <td class="tg-1r2d">2.91M</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">2.95M</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Use)</td>
    <td class="tg-wcxf">26.47</td>
    <td class="tg-wcxf">67.12</td>
    <td class="tg-wcxf">45.10</td>
    <td class="tg-aycn">45.55k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">249.54k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">267.03k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">122.90k</td>
    <td class="tg-1r2d">24.27k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">24.35k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
</table>

<br>

Running Benchmarks on Your Machine
----------------------------------

If you would like to run the benchmarks on your own machine, you can use the **_DiffSharp Benchmarks_** command line tool distributed together with the latest release <a href="https://github.com/gbaydin/DiffSharp/releases">on GitHub</a>.

<div class="row">
    <div class="span6">
        <img src="img/benchmarks.png" alt="Chart" style="width:569px"/>
    </div>
</div>
*)

