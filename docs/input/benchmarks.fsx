
(**
Benchmarks
==========

The following tables present benchmark results measuring the running times of the operations in the library. For descriptions of these operations, please refer to [API Overview](api-overview.html).

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
    <td class="tg-wcxf">3.36</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">17.7</td>
    <td class="tg-aycn">5.7</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">30.61</td>
    <td class="tg-1r2d">5.69</td>
    <td class="tg-1r2d">25.61</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Forward2</td>
    <td class="tg-wcxf">4.11</td>
    <td class="tg-wcxf">4.07</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">20.70</td>
    <td class="tg-aycn">6.14</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">21.12</td>
    <td class="tg-wklz">34.38</td>
    <td class="tg-1r2d">6.23</td>
    <td class="tg-1r2d">27.73</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardG</td>
    <td class="tg-wcxf">6.78</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">23.93</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">25.82</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">32.39</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardGH</td>
    <td class="tg-wcxf">19.62</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">130.27</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">131.10</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">131.22</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">135.41</td>
    <td class="tg-wklz">95.60</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">101.82</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardN</td>
    <td class="tg-wcxf">16.63</td>
    <td class="tg-wcxf">42.52</td>
    <td class="tg-wcxf">43.50</td>
    <td class="tg-aycn">122.04</td>
    <td class="tg-aycn">38.99</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">377.53</td>
    <td class="tg-1r2d">106.35</td>
    <td class="tg-1r2d">28.81</td>
    <td class="tg-1r2d">97.88</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardReverse</td>
    <td class="tg-wcxf">64.71</td>
    <td class="tg-wcxf">17.23</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">31.62</td>
    <td class="tg-aycn">156.99</td>
    <td class="tg-aycn">115.23</td>
    <td class="tg-aycn">33.04</td>
    <td class="tg-aycn">118.86</td>
    <td class="tg-aycn">34.63</td>
    <td class="tg-aycn">118.90</td>
    <td class="tg-1r2d">183.83</td>
    <td class="tg-1r2d">111.50</td>
    <td class="tg-1r2d">177.79</td>
    <td class="tg-1r2d">47.52</td>
    <td class="tg-1r2d">50.10</td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Reverse</td>
    <td class="tg-wcxf">10.03</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">11.26</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">100.11</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">118.50</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">107.07</td>
    <td class="tg-wklz">37.50</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">44.37</td>
    <td class="tg-1r2d">9.23</td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Numerical</td>
    <td class="tg-wcxf">2.30</td>
    <td class="tg-wcxf">3.30</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">17.96</td>
    <td class="tg-aycn">13.37</td>
    <td class="tg-aycn">130.94</td>
    <td class="tg-aycn">52.95</td>
    <td class="tg-aycn">134.93</td>
    <td class="tg-aycn">70.67</td>
    <td class="tg-aycn">138.20</td>
    <td class="tg-wklz">46.88</td>
    <td class="tg-1r2d">13.49</td>
    <td class="tg-1r2d">41.47</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Compile)</td>
    <td class="tg-wcxf">64.66k</td>
    <td class="tg-wcxf">215.76k</td>
    <td class="tg-wcxf">216.79k</td>
    <td class="tg-aycn">408.76k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">3.32M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">3.90M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">1.90M</td>
    <td class="tg-1r2d">335.59k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">336.76k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Use)</td>
    <td class="tg-wcxf">126.65</td>
    <td class="tg-wcxf">82.63</td>
    <td class="tg-wcxf">47.53</td>
    <td class="tg-aycn">42.42k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">326.58k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">386.61k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">166.51k</td>
    <td class="tg-1r2d">26.41k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">22.75k</td>
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
    <td class="tg-wcxf">3.13</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">17.75</td>
    <td class="tg-aycn">5.31</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-1r2d">32.03</td>
    <td class="tg-1r2d">5.36</td>
    <td class="tg-1r2d">25.85</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Forward2</td>
    <td class="tg-wcxf">4.12</td>
    <td class="tg-wcxf">4.06</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">21.19</td>
    <td class="tg-aycn">6.28</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">21.59</td>
    <td class="tg-1r2d">34.77</td>
    <td class="tg-1r2d">6.39</td>
    <td class="tg-1r2d">28.2</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardG</td>
    <td class="tg-wcxf">6.44</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">24.59</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-1r2d">26.08</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">32.26</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardGH</td>
    <td class="tg-wcxf">19.64</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">127.79</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">134.40</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">136.75</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">138.41</td>
    <td class="tg-1r2d">98.14</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">103.49</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardN</td>
    <td class="tg-wcxf">16.98</td>
    <td class="tg-wcxf">45.23</td>
    <td class="tg-wcxf">45.44</td>
    <td class="tg-aycn">124.99</td>
    <td class="tg-aycn">41.62</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">382.00</td>
    <td class="tg-1r2d">106.83</td>
    <td class="tg-1r2d">29.69</td>
    <td class="tg-1r2d">99.43</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.ForwardReverse</td>
    <td class="tg-wcxf">67.57</td>
    <td class="tg-wcxf">18.4</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">32.79</td>
    <td class="tg-aycn">153.39</td>
    <td class="tg-aycn">118.3</td>
    <td class="tg-aycn">35.69</td>
    <td class="tg-aycn">122.17</td>
    <td class="tg-aycn">34.71</td>
    <td class="tg-aycn">120.11</td>
    <td class="tg-1r2d">183.25</td>
    <td class="tg-1r2d">114.86</td>
    <td class="tg-1r2d">177.69</td>
    <td class="tg-1r2d">46.96</td>
    <td class="tg-1r2d">46.86</td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.AD.Reverse</td>
    <td class="tg-wcxf">7.98</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">11.83</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">104.19</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">107.68</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">108.34</td>
    <td class="tg-1r2d">38.12</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">45.15</td>
    <td class="tg-1r2d">9.05</td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Numerical</td>
    <td class="tg-wcxf">3.21</td>
    <td class="tg-wcxf">4.40</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">18.62</td>
    <td class="tg-aycn">18.25</td>
    <td class="tg-aycn">135.82</td>
    <td class="tg-aycn">57.21</td>
    <td class="tg-aycn">141.66</td>
    <td class="tg-aycn">70.45</td>
    <td class="tg-aycn">139.86</td>
    <td class="tg-1r2d">47.53</td>
    <td class="tg-1r2d">14.56</td>
    <td class="tg-1r2d">41.13</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Compile)</td>
    <td class="tg-wcxf">50.81k</td>
    <td class="tg-wcxf">226.79k</td>
    <td class="tg-wcxf">227.25k</td>
    <td class="tg-aycn">388.75k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">3.51M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">3.81M</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">1.91M</td>
    <td class="tg-1r2d">337.51k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">341.25k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
  <tr>
    <td class="tg-7dqz">DiffSharp.Symbolic (Use)</td>
    <td class="tg-wcxf">33.16</td>
    <td class="tg-wcxf">59.39</td>
    <td class="tg-wcxf">58.70</td>
    <td class="tg-aycn">46.60k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">343.58k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">389.50k</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">173.78k</td>
    <td class="tg-1r2d">27.20k</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d"></td>
  </tr>
</table>

<br>

Running Benchmarks on Your Machine
----------------------------------

If you would like to run the benchmarks on your own machine, you can use the command line **_DiffSharp Benchmarks_** tool distributed together with the latest release <a href="https://github.com/gbaydin/DiffSharp/releases">on GitHub</a>.

<div class="row">
    <div class="span6">
        <img src="img/benchmarks.png" alt="Chart" style="width:569px"/>
    </div>
</div>
*)

