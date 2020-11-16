``` ini

BenchmarkDotNet=v0.12.1, OS=Windows 10.0.17134.1792 (1803/April2018Update/Redstone4)
Intel Xeon CPU E5-1620 0 3.60GHz, 1 CPU, 8 logical and 4 physical cores
.NET Core SDK=5.0.100
  [Host]   : .NET Core 3.1.9 (CoreCLR 4.700.20.47201, CoreFX 4.700.20.47203), X64 RyuJIT DEBUG
  ShortRun : .NET Core 3.1.9 (CoreCLR 4.700.20.47201, CoreFX 4.700.20.47203), X64 RyuJIT

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
|                           Method |   Categories | tensorSize | dtypeName | deviceName |             Mean |             Error |         StdDev |           Median | Ratio | RatioSD | Baseline |
|--------------------------------- |------------- |----------- |---------- |----------- |-----------------:|------------------:|---------------:|-----------------:|------:|--------:|--------- |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float32** |        **cpu** |    **571,773.83 μs** |      **4,782.343 μs** |     **262.136 μs** |    **571,868.10 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float32 |        cpu |    162,539.50 μs |     36,193.288 μs |   1,983.876 μs |    161,747.70 μs | 0.284 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float32 |        cpu |    276,007.23 μs |    333,680.151 μs |  18,290.135 μs |    281,495.60 μs | 0.483 |    0.03 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float32 |        cpu |    290,327.73 μs |    611,376.935 μs |  33,511.633 μs |    272,758.50 μs | 0.508 |    0.06 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float32 |        cpu |      5,764.83 μs |     31,314.151 μs |   1,716.434 μs |      4,861.30 μs | 0.010 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float32 |        cpu |      3,842.88 μs |        990.896 μs |      54.314 μs |      3,865.90 μs | 0.007 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float32 |        cpu |  1,625,463.60 μs |      3,266.973 μs |     179.074 μs |  1,625,559.50 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float32 |        cpu |    714,176.87 μs |  1,792,205.176 μs |  98,236.813 μs |    664,456.40 μs | 0.439 |    0.06 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float32 |        cpu |    704,073.97 μs |  1,247,680.936 μs |  68,389.602 μs |    702,224.80 μs | 0.433 |    0.04 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float32 |        cpu |    743,258.87 μs |  1,459,908.087 μs |  80,022.488 μs |    763,641.90 μs | 0.457 |    0.05 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float32 |        cpu |     13,320.99 μs |      8,886.348 μs |     487.091 μs |     13,418.91 μs | 0.008 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float32 |        cpu |     13,355.39 μs |      5,425.545 μs |     297.392 μs |     13,200.88 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float32 |        cpu |  1,759,653.80 μs |      7,103.177 μs |     389.349 μs |  1,759,608.30 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float32 |        cpu |    560,769.10 μs |    516,627.806 μs |  28,318.113 μs |    547,716.20 μs | 0.319 |    0.02 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float32 |        cpu |    648,424.07 μs |    533,075.640 μs |  29,219.675 μs |    647,648.30 μs | 0.368 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float32 |        cpu |    687,194.70 μs |    287,103.140 μs |  15,737.092 μs |    683,842.80 μs | 0.391 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float32 |        cpu |     13,138.17 μs |      1,938.985 μs |     106.282 μs |     13,108.61 μs | 0.007 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float32 |        cpu |     14,209.21 μs |      2,992.398 μs |     164.023 μs |     14,137.08 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float32 |        cpu |  1,863,751.03 μs |      6,493.466 μs |     355.929 μs |  1,863,817.10 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float32 |        cpu |    704,519.03 μs |  1,263,793.800 μs |  69,272.803 μs |    671,948.40 μs |  0.38 |    0.04 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float32 |        cpu |    735,690.87 μs |    299,468.131 μs |  16,414.859 μs |    729,644.60 μs |  0.39 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float32 |        cpu |    735,604.97 μs |     82,129.102 μs |   4,501.773 μs |    734,726.00 μs |  0.39 |    0.00 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float32 |        cpu |     47,166.00 μs |     18,576.657 μs |   1,018.249 μs |     46,670.99 μs |  0.03 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float32 |        cpu |     45,719.40 μs |     11,084.489 μs |     607.578 μs |     45,784.43 μs |  0.02 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float32 |        cpu |    752,989.57 μs |      4,024.139 μs |     220.577 μs |    753,005.40 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float32 |        cpu |    496,342.00 μs |     23,934.546 μs |   1,311.933 μs |    495,887.70 μs |  0.66 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float32 |        cpu |    574,945.27 μs |    485,676.702 μs |  26,621.579 μs |    565,512.90 μs |  0.76 |    0.04 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float32 |        cpu |  1,046,887.93 μs |    805,451.032 μs |  44,149.489 μs |  1,023,570.90 μs |  1.39 |    0.06 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float32 |        cpu |     14,113.94 μs |      3,259.109 μs |     178.643 μs |     14,055.51 μs |  0.02 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float32 |        cpu |    136,052.77 μs |     25,886.489 μs |   1,418.926 μs |    135,365.10 μs |  0.18 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float32 |        cpu |  1,854,320.23 μs |      2,931.832 μs |     160.704 μs |  1,854,361.90 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float32 |        cpu |  1,526,913.90 μs |     81,016.514 μs |   4,440.788 μs |  1,527,631.00 μs | 0.823 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float32 |        cpu |  1,748,382.43 μs |  1,572,500.150 μs |  86,194.039 μs |  1,724,854.70 μs | 0.943 |    0.05 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float32 |        cpu |  2,405,812.80 μs |    661,675.236 μs |  36,268.652 μs |  2,402,414.50 μs | 1.297 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float32 |        cpu |      7,779.07 μs |      5,695.675 μs |     312.199 μs |      7,678.39 μs | 0.004 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float32 |        cpu |    545,743.03 μs |    594,534.246 μs |  32,588.428 μs |    561,563.50 μs | 0.294 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float32 |        cpu |    894,420.23 μs |      3,068.801 μs |     168.211 μs |    894,493.60 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float32 |        cpu |    452,986.37 μs |     73,117.157 μs |   4,007.798 μs |    452,057.00 μs |  0.51 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float32 |        cpu |    628,937.33 μs |    287,049.844 μs |  15,734.171 μs |    627,724.10 μs |  0.70 |    0.02 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float32 |        cpu |  3,327,512.43 μs |  1,604,245.906 μs |  87,934.131 μs |  3,314,122.20 μs |  3.72 |    0.10 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float32 |        cpu |     16,946.77 μs |      8,126.496 μs |     445.441 μs |     17,185.59 μs |  0.02 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float32 |        cpu |    665,491.23 μs |    261,632.852 μs |  14,340.979 μs |    663,031.20 μs |  0.74 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float32 |        cpu |    400,792.23 μs |      3,914.882 μs |     214.588 μs |    400,705.00 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float32 |        cpu |    353,001.37 μs |    799,822.770 μs |  43,840.985 μs |    350,344.20 μs |  0.88 |    0.11 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float32 |        cpu |  1,206,929.67 μs |    740,054.163 μs |  40,564.866 μs |  1,184,568.90 μs |  3.01 |    0.10 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float32 |        cpu |    141,316.87 μs |     35,759.157 μs |   1,960.080 μs |    141,727.50 μs |  0.35 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float32 |        cpu |     67,468.73 μs |    139,204.900 μs |   7,630.290 μs |     64,586.60 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float32 |        cpu |     74,134.50 μs |    156,483.889 μs |   8,577.410 μs |     76,886.70 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float32 |        cpu |     99,415.80 μs |     32,830.183 μs |   1,799.533 μs |     99,661.10 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float32 |        cpu |     76,253.30 μs |    168,568.015 μs |   9,239.782 μs |     75,141.20 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float32 |        cpu |     69,003.70 μs |     27,684.468 μs |   1,517.479 μs |     68,633.90 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float32** |       **cuda** |  **3,543,609.70 μs** |      **5,902.630 μs** |     **323.543 μs** |  **3,543,711.50 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float32 |       cuda |    183,275.20 μs |    121,483.898 μs |   6,658.942 μs |    186,975.40 μs | 0.052 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float32 |       cuda |  3,473,800.27 μs |    883,503.193 μs |  48,427.791 μs |  3,478,576.10 μs | 0.980 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float32 |       cuda |  3,309,210.03 μs |  1,260,657.955 μs |  69,100.916 μs |  3,270,620.60 μs | 0.934 |    0.02 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float32 |       cuda |      2,348.93 μs |      2,096.691 μs |     114.927 μs |      2,296.77 μs | 0.001 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float32 |       cuda |      4,228.94 μs |      4,314.776 μs |     236.507 μs |      4,295.12 μs | 0.001 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float32 |       cuda |  5,444,447.07 μs |      4,638.467 μs |     254.250 μs |  5,444,591.20 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float32 |       cuda |  3,104,838.03 μs |    959,463.268 μs |  52,591.419 μs |  3,112,890.20 μs | 0.570 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float32 |       cuda |  3,222,267.47 μs |    771,478.458 μs |  42,287.338 μs |  3,199,339.30 μs | 0.592 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float32 |       cuda |  3,712,546.63 μs |  1,975,359.502 μs | 108,276.120 μs |  3,657,861.00 μs | 0.682 |    0.02 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float32 |       cuda |     12,879.62 μs |      8,586.801 μs |     470.672 μs |     13,098.40 μs | 0.002 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float32 |       cuda |     14,719.53 μs |      5,728.343 μs |     313.990 μs |     14,771.63 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float32 |       cuda |  5,160,675.57 μs |      4,467.120 μs |     244.858 μs |  5,160,571.30 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float32 |       cuda |  2,798,986.13 μs |  1,074,544.246 μs |  58,899.396 μs |  2,766,382.10 μs | 0.542 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float32 |       cuda |  3,101,031.77 μs |  2,255,741.685 μs | 123,644.813 μs |  3,132,236.00 μs | 0.601 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float32 |       cuda |  3,096,312.07 μs |  2,195,820.328 μs | 120,360.322 μs |  3,123,891.40 μs | 0.600 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float32 |       cuda |     14,924.02 μs |     18,536.980 μs |   1,016.074 μs |     14,361.92 μs | 0.003 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float32 |       cuda |     14,920.57 μs |      4,315.875 μs |     236.568 μs |     14,931.11 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float32 |       cuda |  5,611,491.53 μs |      5,014.691 μs |     274.872 μs |  5,611,569.50 μs | 1.000 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float32 |       cuda |  3,312,298.27 μs |  4,168,986.297 μs | 228,516.207 μs |  3,361,688.90 μs | 0.590 |    0.04 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float32 |       cuda |  3,538,167.57 μs |  5,492,930.583 μs | 301,086.060 μs |  3,493,204.40 μs | 0.631 |    0.05 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float32 |       cuda |  3,449,894.60 μs |  3,146,242.041 μs | 172,456.143 μs |  3,537,407.80 μs | 0.615 |    0.03 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float32 |       cuda |     48,165.27 μs |     26,018.548 μs |   1,426.164 μs |     48,586.26 μs | 0.009 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float32 |       cuda |     45,872.52 μs |      3,573.380 μs |     195.869 μs |     45,875.27 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float32 |       cuda |  3,475,345.93 μs |      3,332.743 μs |     182.679 μs |  3,475,424.60 μs | 1.000 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float32 |       cuda |  2,391,030.87 μs |    485,432.272 μs |  26,608.181 μs |  2,397,253.80 μs | 0.688 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float32 |       cuda |  2,548,750.50 μs |  3,025,976.042 μs | 165,863.958 μs |  2,534,166.00 μs | 0.733 |    0.05 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float32 |       cuda |  3,709,615.87 μs |  2,301,358.749 μs | 126,145.239 μs |  3,688,431.90 μs | 1.067 |    0.04 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float32 |       cuda |     17,276.89 μs |     15,227.238 μs |     834.656 μs |     17,117.03 μs | 0.005 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float32 |       cuda |    165,752.00 μs |     34,996.113 μs |   1,918.255 μs |    165,275.80 μs | 0.048 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float32 |       cuda |  4,497,551.97 μs |      4,857.317 μs |     266.246 μs |  4,497,664.70 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float32 |       cuda |  3,307,617.43 μs |  5,395,700.658 μs | 295,756.560 μs |  3,173,632.20 μs | 0.735 |    0.07 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float32 |       cuda |  3,562,162.23 μs |  4,582,947.844 μs | 251,206.836 μs |  3,560,858.90 μs | 0.792 |    0.06 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float32 |       cuda |  6,635,310.77 μs |  3,231,209.766 μs | 177,113.510 μs |  6,584,165.20 μs | 1.475 |    0.04 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float32 |       cuda |      8,816.47 μs |     13,032.404 μs |     714.350 μs |      8,870.11 μs | 0.002 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float32 |       cuda |    510,140.17 μs |     94,983.337 μs |   5,206.357 μs |    511,374.70 μs | 0.113 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float32 |       cuda |  3,633,621.00 μs |      5,534.139 μs |     303.345 μs |  3,633,713.40 μs | 1.000 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float32 |       cuda |  2,322,710.20 μs |  1,187,635.840 μs |  65,098.328 μs |  2,341,255.60 μs | 0.639 |    0.02 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float32 |       cuda |  2,762,481.17 μs |  1,786,784.332 μs |  97,939.678 μs |  2,709,967.80 μs | 0.760 |    0.03 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float32 |       cuda |  8,833,434.30 μs | 11,265,585.842 μs | 617,504.774 μs |  8,621,001.00 μs | 2.431 |    0.17 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float32 |       cuda |     25,960.54 μs |     63,712.161 μs |   3,492.279 μs |     25,162.18 μs | 0.007 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float32 |       cuda |    627,656.97 μs |    793,619.341 μs |  43,500.954 μs |    607,869.00 μs | 0.173 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float32 |       cuda |  1,769,606.00 μs |        777.442 μs |      42.614 μs |  1,769,625.70 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float32 |       cuda |  1,706,803.13 μs |  1,049,106.744 μs |  57,505.081 μs |  1,712,842.30 μs |  0.96 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float32 |       cuda |  3,582,854.23 μs |  2,413,028.504 μs | 132,266.235 μs |  3,586,316.20 μs |  2.02 |    0.07 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float32 |       cuda |    150,490.30 μs |    192,091.672 μs |  10,529.193 μs |    147,680.00 μs |  0.09 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float32 |       cuda |    345,498.73 μs |    283,759.762 μs |  15,553.830 μs |    339,660.30 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float32 |       cuda |    385,656.33 μs |    604,783.811 μs |  33,150.241 μs |    398,888.20 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float32 |       cuda |    486,768.03 μs |    406,272.091 μs |  22,269.144 μs |    496,291.10 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float32 |       cuda |     59,053.70 μs |     46,360.295 μs |   2,541.164 μs |     58,619.20 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float32 |       cuda |     66,537.80 μs |     10,383.429 μs |     569.151 μs |     66,536.70 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float64** |        **cpu** |    **539,343.23 μs** |      **2,939.666 μs** |     **161.133 μs** |    **539,314.50 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float64 |        cpu |    185,960.93 μs |     73,827.758 μs |   4,046.749 μs |    184,727.40 μs | 0.345 |    0.01 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float64 |        cpu |    305,840.23 μs |    293,516.576 μs |  16,088.634 μs |    306,855.30 μs | 0.567 |    0.03 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float64 |        cpu |    321,910.47 μs |    554,199.247 μs |  30,377.531 μs |    331,076.90 μs | 0.597 |    0.06 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float64 |        cpu |      2,633.58 μs |      2,557.856 μs |     140.205 μs |      2,570.27 μs | 0.005 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float64 |        cpu |      4,807.19 μs |      8,831.703 μs |     484.095 μs |      4,609.36 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float64 |        cpu |  1,673,712.37 μs |      4,570.076 μs |     250.501 μs |  1,673,672.00 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float64 |        cpu |    627,988.63 μs |    560,596.793 μs |  30,728.202 μs |    627,732.30 μs | 0.375 |    0.02 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float64 |        cpu |    688,515.40 μs |    291,638.523 μs |  15,985.692 μs |    695,324.10 μs | 0.411 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float64 |        cpu |    718,030.33 μs |    798,397.141 μs |  43,762.841 μs |    737,136.70 μs | 0.429 |    0.03 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float64 |        cpu |     12,716.36 μs |      4,112.463 μs |     225.418 μs |     12,713.00 μs | 0.008 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float64 |        cpu |     15,439.37 μs |     23,069.031 μs |   1,264.491 μs |     15,210.27 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float64 |        cpu |  1,667,530.30 μs |      3,399.332 μs |     186.329 μs |  1,667,538.30 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float64 |        cpu |    596,330.40 μs |    472,206.862 μs |  25,883.251 μs |    609,836.60 μs | 0.358 |    0.02 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float64 |        cpu |    774,463.40 μs |    987,628.784 μs |  54,135.266 μs |    804,594.20 μs | 0.464 |    0.03 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float64 |        cpu |    699,282.23 μs |    731,880.938 μs |  40,116.864 μs |    698,068.90 μs | 0.419 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float64 |        cpu |     15,058.00 μs |     14,783.342 μs |     810.325 μs |     15,520.32 μs | 0.009 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float64 |        cpu |     17,500.63 μs |     37,011.683 μs |   2,028.735 μs |     16,969.95 μs | 0.010 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float64 |        cpu |  1,932,853.97 μs |      5,765.693 μs |     316.037 μs |  1,932,995.50 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float64 |        cpu |    778,084.33 μs |    597,878.572 μs |  32,771.742 μs |    780,701.00 μs |  0.40 |    0.02 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float64 |        cpu |    865,149.03 μs |  1,188,818.338 μs |  65,163.145 μs |    878,589.10 μs |  0.45 |    0.03 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float64 |        cpu |    805,548.90 μs |    435,230.334 μs |  23,856.443 μs |    815,716.90 μs |  0.42 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float64 |        cpu |     50,750.44 μs |     13,608.306 μs |     745.917 μs |     50,705.67 μs |  0.03 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float64 |        cpu |     48,884.67 μs |     21,706.829 μs |   1,189.825 μs |     48,568.44 μs |  0.03 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float64 |        cpu |    756,757.67 μs |      8,448.590 μs |     463.096 μs |    756,978.40 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float64 |        cpu |    552,972.63 μs |    704,141.395 μs |  38,596.366 μs |    542,653.90 μs |  0.73 |    0.05 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float64 |        cpu |    596,541.40 μs |    208,012.304 μs |  11,401.856 μs |    599,094.90 μs |  0.79 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float64 |        cpu |  1,061,219.37 μs |    625,421.587 μs |  34,281.468 μs |  1,076,139.30 μs |  1.40 |    0.05 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float64 |        cpu |     15,095.31 μs |      4,129.927 μs |     226.375 μs |     15,005.33 μs |  0.02 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float64 |        cpu |    143,489.97 μs |     72,020.824 μs |   3,947.704 μs |    145,029.80 μs |  0.19 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float64 |        cpu |  1,919,610.83 μs |      9,803.258 μs |     537.350 μs |  1,919,340.10 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float64 |        cpu |  1,556,071.67 μs |     27,099.503 μs |   1,485.415 μs |  1,556,109.70 μs | 0.811 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float64 |        cpu |  1,704,173.13 μs |  1,021,394.184 μs |  55,986.062 μs |  1,676,445.20 μs | 0.888 |    0.03 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float64 |        cpu |  2,452,131.97 μs |    478,650.077 μs |  26,236.426 μs |  2,461,470.30 μs | 1.277 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float64 |        cpu |      7,610.25 μs |        954.365 μs |      52.312 μs |      7,589.39 μs | 0.004 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float64 |        cpu |    526,548.50 μs |    524,117.672 μs |  28,728.658 μs |    512,516.10 μs | 0.274 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float64 |        cpu |    898,354.47 μs |      3,045.573 μs |     166.938 μs |    898,338.30 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float64 |        cpu |    463,925.37 μs |    199,342.632 μs |  10,926.642 μs |    466,399.50 μs |  0.52 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float64 |        cpu |    630,082.77 μs |    664,090.431 μs |  36,401.037 μs |    647,825.80 μs |  0.70 |    0.04 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float64 |        cpu |  3,465,085.03 μs |  2,986,632.498 μs | 163,707.405 μs |  3,524,964.90 μs |  3.86 |    0.18 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float64 |        cpu |     17,108.44 μs |     13,481.655 μs |     738.975 μs |     17,063.89 μs |  0.02 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float64 |        cpu |    600,662.80 μs |     89,215.879 μs |   4,890.223 μs |    603,182.20 μs |  0.67 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float64 |        cpu |    391,934.77 μs |      3,093.428 μs |     169.561 μs |    392,029.90 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float64 |        cpu |    329,745.10 μs |    314,355.351 μs |  17,230.878 μs |    332,993.00 μs |  0.84 |    0.04 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float64 |        cpu |  1,066,692.13 μs |    991,839.254 μs |  54,366.056 μs |  1,037,676.50 μs |  2.72 |    0.14 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float64 |        cpu |    139,813.33 μs |      9,443.249 μs |     517.616 μs |    139,847.90 μs |  0.36 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float64 |        cpu |     54,688.73 μs |     39,720.522 μs |   2,177.216 μs |     53,584.60 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float64 |        cpu |     62,258.10 μs |     20,937.628 μs |   1,147.662 μs |     61,809.00 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float64 |        cpu |    102,533.80 μs |    106,319.151 μs |   5,827.711 μs |    100,933.40 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float64 |        cpu |     53,684.47 μs |     18,143.011 μs |     994.480 μs |     53,594.90 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float64 |        cpu |     78,811.67 μs |    355,367.025 μs |  19,478.866 μs |     68,360.80 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float64** |       **cuda** |  **3,565,691.23 μs** |      **4,916.974 μs** |     **269.516 μs** |  **3,565,617.30 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float64 |       cuda |    176,596.90 μs |    183,295.449 μs |  10,047.042 μs |    176,455.90 μs | 0.050 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float64 |       cuda |  3,475,689.43 μs |  3,379,988.458 μs | 185,268.573 μs |  3,496,933.70 μs | 0.975 |    0.05 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float64 |       cuda |  3,323,739.27 μs |  4,140,432.878 μs | 226,951.097 μs |  3,389,207.90 μs | 0.932 |    0.06 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float64 |       cuda |      2,496.68 μs |      1,329.766 μs |      72.889 μs |      2,514.61 μs | 0.001 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float64 |       cuda |      5,044.61 μs |      7,149.469 μs |     391.887 μs |      5,010.60 μs | 0.001 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float64 |       cuda |  5,246,865.83 μs |      4,015.810 μs |     220.120 μs |  5,246,805.60 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float64 |       cuda |  2,745,979.33 μs |    785,868.710 μs |  43,076.116 μs |  2,757,979.70 μs | 0.523 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float64 |       cuda |  2,883,494.33 μs |  2,748,787.251 μs | 150,670.305 μs |  2,918,139.60 μs | 0.550 |    0.03 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float64 |       cuda |  3,121,654.73 μs |  3,176,174.601 μs | 174,096.847 μs |  3,181,496.30 μs | 0.595 |    0.03 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float64 |       cuda |     12,770.18 μs |      5,016.576 μs |     274.975 μs |     12,742.63 μs | 0.002 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float64 |       cuda |     14,488.23 μs |      6,660.726 μs |     365.097 μs |     14,485.77 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float64 |       cuda |  5,088,685.67 μs |      4,918.693 μs |     269.610 μs |  5,088,706.60 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float64 |       cuda |  2,721,197.50 μs |    850,835.636 μs |  46,637.172 μs |  2,717,898.90 μs | 0.535 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float64 |       cuda |  3,212,773.60 μs |  2,693,589.056 μs | 147,644.705 μs |  3,154,253.80 μs | 0.631 |    0.03 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float64 |       cuda |  2,873,786.33 μs |  1,895,205.595 μs | 103,882.614 μs |  2,917,369.50 μs | 0.565 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float64 |       cuda |     14,788.27 μs |     10,482.101 μs |     574.559 μs |     14,644.22 μs | 0.003 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float64 |       cuda |     15,440.04 μs |     17,010.166 μs |     932.385 μs |     14,905.03 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float64 |       cuda |  5,458,535.83 μs |      6,338.357 μs |     347.427 μs |  5,458,488.50 μs | 1.000 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float64 |       cuda |  3,383,937.43 μs |  1,465,260.767 μs |  80,315.887 μs |  3,387,292.90 μs | 0.620 |    0.01 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float64 |       cuda |  3,272,269.00 μs |  2,111,372.631 μs | 115,731.458 μs |  3,329,941.50 μs | 0.599 |    0.02 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float64 |       cuda |  3,294,657.73 μs |  2,503,637.858 μs | 137,232.839 μs |  3,224,316.90 μs | 0.604 |    0.03 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float64 |       cuda |     45,500.20 μs |      4,754.433 μs |     260.607 μs |     45,361.36 μs | 0.008 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float64 |       cuda |     54,333.21 μs |     11,401.465 μs |     624.953 μs |     53,998.09 μs | 0.010 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float64 |       cuda |  3,187,773.30 μs |      8,079.747 μs |     442.878 μs |  3,187,713.60 μs | 1.000 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float64 |       cuda |  2,464,519.27 μs |    362,825.620 μs |  19,887.697 μs |  2,465,731.90 μs | 0.773 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float64 |       cuda |  2,479,123.17 μs |    501,454.165 μs |  27,486.395 μs |  2,475,524.10 μs | 0.778 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float64 |       cuda |  3,472,405.90 μs |  1,296,733.040 μs |  71,078.314 μs |  3,504,399.60 μs | 1.089 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float64 |       cuda |     14,592.05 μs |      4,787.427 μs |     262.415 μs |     14,465.60 μs | 0.005 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float64 |       cuda |    146,216.67 μs |    112,184.763 μs |   6,149.225 μs |    147,942.60 μs | 0.046 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float64 |       cuda |  4,454,804.87 μs |      5,326.828 μs |     291.981 μs |  4,454,783.60 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float64 |       cuda |  3,336,106.63 μs |  1,069,597.535 μs |  58,628.250 μs |  3,330,963.30 μs | 0.749 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float64 |       cuda |  3,536,805.00 μs |  2,685,530.136 μs | 147,202.969 μs |  3,532,457.00 μs | 0.794 |    0.03 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float64 |       cuda |  6,304,190.47 μs |  2,348,617.677 μs | 128,735.660 μs |  6,344,560.70 μs | 1.415 |    0.03 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float64 |       cuda |      7,773.90 μs |        353.356 μs |      19.369 μs |      7,764.10 μs | 0.002 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float64 |       cuda |    513,562.23 μs |    430,975.433 μs |  23,623.218 μs |    524,819.40 μs | 0.115 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float64 |       cuda |  3,708,664.47 μs |      6,121.809 μs |     335.557 μs |  3,708,703.50 μs | 1.000 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float64 |       cuda |  2,838,533.47 μs |  3,916,313.123 μs | 214,666.337 μs |  2,908,540.20 μs | 0.765 |    0.06 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float64 |       cuda |  2,680,567.47 μs |  1,203,486.461 μs |  65,967.154 μs |  2,642,797.90 μs | 0.723 |    0.02 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float64 |       cuda |  7,973,932.30 μs |  1,783,037.825 μs |  97,734.320 μs |  7,993,674.10 μs | 2.150 |    0.03 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float64 |       cuda |     16,143.54 μs |        749.943 μs |      41.107 μs |     16,166.00 μs | 0.004 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float64 |       cuda |    610,369.90 μs |    164,684.780 μs |   9,026.929 μs |    607,477.60 μs | 0.165 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float64 |       cuda |  1,797,398.77 μs |        618.075 μs |      33.879 μs |  1,797,396.10 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float64 |       cuda |  1,307,488.47 μs |    431,983.775 μs |  23,678.488 μs |  1,300,074.50 μs |  0.73 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float64 |       cuda |  3,539,655.87 μs |  1,151,806.315 μs |  63,134.391 μs |  3,521,084.50 μs |  1.97 |    0.04 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float64 |       cuda |    139,982.00 μs |     56,384.718 μs |   3,090.637 μs |    139,227.90 μs |  0.08 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float64 |       cuda |    288,404.90 μs |    283,498.512 μs |  15,539.510 μs |    280,764.40 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float64 |       cuda |    297,707.13 μs |    445,215.423 μs |  24,403.760 μs |    284,319.50 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float64 |       cuda |    420,900.67 μs |    726,459.605 μs |  39,819.702 μs |    416,972.90 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float64 |       cuda |     55,530.03 μs |     10,096.780 μs |     553.439 μs |     55,646.00 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float64 |       cuda |     72,989.53 μs |     46,612.555 μs |   2,554.991 μs |     72,785.10 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |     **int32** |        **cpu** |    **515,502.30 μs** |      **7,220.392 μs** |     **395.774 μs** |    **515,301.80 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |     int32 |        cpu |    162,043.80 μs |    154,935.727 μs |   8,492.550 μs |    158,330.50 μs | 0.314 |    0.02 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |     int32 |        cpu |    265,841.47 μs |    122,135.758 μs |   6,694.673 μs |    267,566.80 μs | 0.516 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |     int32 |        cpu |    279,077.27 μs |    197,834.492 μs |  10,843.976 μs |    277,495.90 μs | 0.541 |    0.02 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |     int32 |        cpu |      2,372.15 μs |      4,220.523 μs |     231.341 μs |      2,396.20 μs | 0.005 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |     int32 |        cpu |      2,715.41 μs |      2,359.746 μs |     129.346 μs |      2,647.65 μs | 0.005 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |     int32 |        cpu |  1,656,776.33 μs |      4,356.283 μs |     238.783 μs |  1,656,788.50 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |     int32 |        cpu |    597,347.37 μs |    256,705.899 μs |  14,070.917 μs |    590,896.10 μs | 0.361 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |     int32 |        cpu |    643,989.30 μs |    771,375.371 μs |  42,281.687 μs |    621,488.10 μs | 0.389 |    0.03 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |     int32 |        cpu |    637,751.10 μs |    376,179.065 μs |  20,619.644 μs |    648,523.80 μs | 0.385 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |     int32 |        cpu |     13,361.13 μs |     27,326.895 μs |   1,497.879 μs |     13,066.73 μs | 0.008 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |     int32 |        cpu |     13,321.43 μs |      3,772.882 μs |     206.804 μs |     13,220.88 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |         16 |     int32 |        cpu |  1,637,807.17 μs |      4,569.253 μs |     250.456 μs |  1,637,769.30 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |     int32 |        cpu |    551,542.73 μs |    455,725.618 μs |  24,979.859 μs |    538,348.10 μs | 0.337 |    0.02 |       No |
|             ones_RawTensor_Torch |         ones |         16 |     int32 |        cpu |    597,492.23 μs |    911,992.042 μs |  49,989.361 μs |    569,506.90 μs | 0.365 |    0.03 |       No |
|                ones_Tensor_Torch |         ones |         16 |     int32 |        cpu |    585,962.97 μs |    202,518.906 μs |  11,100.745 μs |    590,444.70 μs | 0.358 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |         16 |     int32 |        cpu |     14,350.66 μs |     10,742.014 μs |     588.806 μs |     14,449.20 μs | 0.009 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |     int32 |        cpu |     14,950.66 μs |     10,606.330 μs |     581.369 μs |     14,655.55 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |         16 |     int32 |        cpu |    673,441.70 μs |    347,741.131 μs |  19,060.865 μs |    671,378.10 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |         16 |     int32 |        cpu |    725,562.87 μs |        548.855 μs |      30.085 μs |    725,557.70 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |     int32 |        cpu |    515,751.67 μs |     57,284.975 μs |   3,139.983 μs |    517,474.30 μs |  0.71 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |         16 |     int32 |        cpu |    622,066.37 μs |    853,477.743 μs |  46,781.995 μs |    605,409.20 μs |  0.86 |    0.06 |       No |
|            addition_Tensor_Torch |     addition |         16 |     int32 |        cpu |  1,076,557.03 μs |    456,453.775 μs |  25,019.772 μs |  1,063,776.70 μs |  1.48 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |         16 |     int32 |        cpu |     22,535.87 μs |        564.360 μs |      30.935 μs |     22,525.30 μs |  0.03 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |     int32 |        cpu |    138,891.53 μs |    142,209.545 μs |   7,794.985 μs |    134,455.60 μs |  0.19 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |     int32 |        cpu |  1,924,438.93 μs |      3,827.798 μs |     209.815 μs |  1,924,528.50 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |     int32 |        cpu |  1,503,591.00 μs |    956,246.391 μs |  52,415.091 μs |  1,479,391.10 μs | 0.781 |    0.03 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |     int32 |        cpu |  1,588,433.90 μs |    239,604.937 μs |  13,133.555 μs |  1,586,129.90 μs | 0.825 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |     int32 |        cpu |  3,970,387.23 μs |  1,573,309.898 μs |  86,238.424 μs |  3,979,099.90 μs | 2.063 |    0.04 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |     int32 |        cpu |      7,266.10 μs |      2,820.928 μs |     154.625 μs |      7,221.84 μs | 0.004 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |     int32 |        cpu |  4,279,303.60 μs |  2,272,930.513 μs | 124,586.991 μs |  4,223,027.30 μs | 2.224 |    0.06 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |     int32 |        cpu |    882,757.37 μs |      4,984.290 μs |     273.206 μs |    882,870.90 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |     int32 |        cpu |    441,226.23 μs |    117,804.535 μs |   6,457.264 μs |    438,224.30 μs |  0.50 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |     int32 |        cpu |    623,340.30 μs |    613,141.301 μs |  33,608.344 μs |    634,911.90 μs |  0.71 |    0.04 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |     int32 |        cpu |  3,262,734.47 μs |    972,527.360 μs |  53,307.506 μs |  3,279,888.50 μs |  3.70 |    0.06 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |     int32 |        cpu |     15,925.41 μs |      6,928.293 μs |     379.763 μs |     15,945.19 μs |  0.02 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |     int32 |        cpu |  7,830,174.30 μs |  1,507,261.486 μs |  82,618.088 μs |  7,790,726.60 μs |  8.87 |    0.09 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |     int32 |        cpu |    392,505.20 μs |      7,576.534 μs |     415.295 μs |    392,275.60 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |     int32 |        cpu |    266,911.40 μs |    133,063.261 μs |   7,293.646 μs |    263,086.80 μs |  0.68 |    0.02 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |     int32 |        cpu |  1,016,264.40 μs |    445,758.455 μs |  24,433.525 μs |  1,017,920.40 μs |  2.59 |    0.06 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |     int32 |        cpu |    138,032.90 μs |     17,721.727 μs |     971.388 μs |    137,643.90 μs |  0.35 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |     int32 |        cpu |     48,205.73 μs |     26,937.746 μs |   1,476.549 μs |     47,743.50 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |     int32 |        cpu |     54,713.22 μs |     14,823.356 μs |     812.518 μs |     54,546.44 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |     int32 |        cpu |     96,285.70 μs |     19,124.901 μs |   1,048.300 μs |     95,835.50 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |     int32 |        cpu |     52,185.77 μs |     10,794.839 μs |     591.702 μs |     51,998.43 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |     int32 |        cpu |     66,206.45 μs |      9,156.998 μs |     501.926 μs |     66,421.82 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |     **int32** |       **cuda** |  **3,374,810.37 μs** |      **4,580.497 μs** |     **251.073 μs** |  **3,374,874.00 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |     int32 |       cuda |    166,520.73 μs |     58,125.162 μs |   3,186.036 μs |    167,920.10 μs | 0.049 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |     int32 |       cuda |  2,959,817.63 μs |    296,340.390 μs |  16,243.417 μs |  2,967,480.90 μs | 0.877 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |     int32 |       cuda |  3,381,523.23 μs | 12,508,581.879 μs | 685,637.581 μs |  3,058,205.60 μs | 1.002 |    0.20 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |     int32 |       cuda |      2,349.12 μs |      4,851.144 μs |     265.908 μs |      2,219.71 μs | 0.001 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |     int32 |       cuda |      2,723.29 μs |      1,020.347 μs |      55.929 μs |      2,711.70 μs | 0.001 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |     int32 |       cuda |  5,128,744.53 μs |      8,844.064 μs |     484.773 μs |  5,128,836.40 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |     int32 |       cuda |  2,753,272.50 μs |    458,139.378 μs |  25,112.165 μs |  2,755,192.60 μs | 0.537 |    0.00 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |     int32 |       cuda |  3,171,957.73 μs |  1,455,220.460 μs |  79,765.544 μs |  3,183,149.90 μs | 0.618 |    0.02 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |     int32 |       cuda |  2,935,559.77 μs |    303,233.856 μs |  16,621.271 μs |  2,934,589.10 μs | 0.572 |    0.00 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |     int32 |       cuda |     11,564.33 μs |      2,052.525 μs |     112.506 μs |     11,585.62 μs | 0.002 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |     int32 |       cuda |     12,603.52 μs |        376.986 μs |      20.664 μs |     12,614.85 μs | 0.002 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |         16 |     int32 |       cuda |  4,960,847.03 μs |      6,785.759 μs |     371.950 μs |  4,961,028.30 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |     int32 |       cuda |  2,687,036.87 μs |    547,984.728 μs |  30,036.892 μs |  2,681,952.10 μs | 0.542 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |         16 |     int32 |       cuda |  2,885,577.77 μs |  1,360,649.724 μs |  74,581.803 μs |  2,852,448.20 μs | 0.582 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |         16 |     int32 |       cuda |  3,129,511.47 μs |  2,728,792.179 μs | 149,574.307 μs |  3,112,110.70 μs | 0.631 |    0.03 |       No |
|         ones_RawTensor_Reference |         ones |         16 |     int32 |       cuda |     12,831.01 μs |      4,024.045 μs |     220.571 μs |     12,901.43 μs | 0.003 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |     int32 |       cuda |     14,675.76 μs |      2,353.543 μs |     129.006 μs |     14,618.21 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |         16 |     int32 |       cuda |  2,816,767.30 μs |    462,621.957 μs |  25,357.871 μs |  2,825,926.80 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |         16 |     int32 |       cuda |  3,065,860.60 μs |      4,198.379 μs |     230.127 μs |  3,065,731.10 μs | 1.000 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |     int32 |       cuda |  2,386,992.33 μs |    312,326.544 μs |  17,119.672 μs |  2,380,418.40 μs | 0.779 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |         16 |     int32 |       cuda |  2,929,099.80 μs |    391,823.301 μs |  21,477.157 μs |  2,919,994.80 μs | 0.955 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |         16 |     int32 |       cuda |  3,727,245.80 μs |  3,689,285.125 μs | 202,222.167 μs |  3,728,363.50 μs | 1.216 |    0.07 |       No |
|     addition_RawTensor_Reference |     addition |         16 |     int32 |       cuda |     13,187.26 μs |      3,788.919 μs |     207.683 μs |     13,128.90 μs | 0.004 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |     int32 |       cuda |    150,713.13 μs |    106,939.251 μs |   5,861.701 μs |    150,708.80 μs | 0.049 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |     int32 |       cuda |  4,234,340.30 μs |      1,106.393 μs |      60.645 μs |  4,234,362.90 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |     int32 |       cuda |  3,689,908.50 μs |    439,004.305 μs |  24,063.307 μs |  3,689,680.90 μs | 0.871 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |     int32 |       cuda |  3,735,911.87 μs |  3,178,770.703 μs | 174,239.149 μs |  3,819,495.20 μs | 0.882 |    0.04 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |     int32 |       cuda | 11,714,638.57 μs |  8,388,607.303 μs | 459,807.872 μs | 11,617,029.00 μs | 2.767 |    0.11 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |     int32 |       cuda |      7,485.35 μs |      2,452.864 μs |     134.450 μs |      7,411.63 μs | 0.002 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |     int32 |       cuda |  4,332,552.07 μs |  2,705,362.820 μs | 148,290.065 μs |  4,407,022.90 μs | 1.023 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |     int32 |       cuda |  3,662,860.77 μs |      4,891.362 μs |     268.112 μs |  3,662,760.80 μs | 1.000 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |     int32 |       cuda |  2,384,346.53 μs |  3,235,521.225 μs | 177,349.836 μs |  2,291,190.60 μs | 0.651 |    0.05 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |     int32 |       cuda |  2,849,813.33 μs |    521,435.828 μs |  28,581.657 μs |  2,862,065.10 μs | 0.778 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |     int32 |       cuda |  8,159,552.83 μs |  1,097,429.355 μs |  60,153.806 μs |  8,154,637.50 μs | 2.228 |    0.02 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |     int32 |       cuda |     15,135.33 μs |      2,392.938 μs |     131.165 μs |     15,087.73 μs | 0.004 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |     int32 |       cuda |  8,320,894.13 μs |  2,976,632.233 μs | 163,159.257 μs |  8,336,560.20 μs | 2.272 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |     int32 |       cuda |  1,699,812.87 μs |      7,553.271 μs |     414.020 μs |  1,699,985.90 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |     int32 |       cuda |  1,512,431.23 μs |  3,358,508.056 μs | 184,091.159 μs |  1,419,534.60 μs |  0.89 |    0.11 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |     int32 |       cuda |  3,707,131.93 μs |  3,166,979.498 μs | 173,592.833 μs |  3,665,685.80 μs |  2.18 |    0.10 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |     int32 |       cuda |    159,598.80 μs |    411,254.620 μs |  22,542.253 μs |    149,052.90 μs |  0.09 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |     int32 |       cuda |  1,801,991.97 μs |  1,311,135.252 μs |  71,867.747 μs |  1,769,248.50 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |     int32 |       cuda |  1,942,664.80 μs |    540,617.761 μs |  29,633.084 μs |  1,938,059.80 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |     int32 |       cuda |     62,501.50 μs |    125,612.275 μs |   6,885.233 μs |     64,325.90 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |     int32 |       cuda |     80,394.37 μs |    213,180.860 μs |  11,685.162 μs |     79,318.40 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float32** |        **cpu** |     **28,749.73 μs** |      **1,280.154 μs** |      **70.170 μs** |     **28,773.53 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float32 |        cpu |      1,718.22 μs |        421.986 μs |      23.130 μs |      1,708.94 μs | 0.060 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float32 |        cpu |      3,935.50 μs |      1,423.073 μs |      78.003 μs |      3,924.10 μs | 0.137 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float32 |        cpu |      4,006.10 μs |      2,346.831 μs |     128.638 μs |      4,055.58 μs | 0.139 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float32 |        cpu |        159.48 μs |         45.405 μs |       2.489 μs |        159.10 μs | 0.006 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float32 |        cpu |        172.47 μs |          8.768 μs |       0.481 μs |        172.39 μs | 0.006 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float32 |        cpu |     15,788.87 μs |        248.087 μs |      13.598 μs |     15,793.62 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float32 |        cpu |     19,532.48 μs |     21,488.455 μs |   1,177.855 μs |     19,005.53 μs |  1.24 |    0.08 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float32 |        cpu |     10,436.90 μs |      2,474.136 μs |     135.616 μs |     10,479.94 μs |  0.66 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float32 |        cpu |     10,179.25 μs |      2,862.805 μs |     156.920 μs |     10,119.80 μs |  0.64 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float32 |        cpu |      1,192.22 μs |        145.073 μs |       7.952 μs |      1,191.55 μs |  0.08 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float32 |        cpu |      1,286.89 μs |        893.851 μs |      48.995 μs |      1,278.66 μs |  0.08 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float32 |        cpu |     14,770.95 μs |        734.395 μs |      40.255 μs |     14,783.70 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float32 |        cpu |     19,711.03 μs |      8,076.951 μs |     442.725 μs |     19,866.70 μs |  1.33 |    0.03 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float32 |        cpu |     10,589.06 μs |     10,865.778 μs |     595.590 μs |     10,689.62 μs |  0.72 |    0.04 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float32 |        cpu |     10,885.19 μs |     11,071.065 μs |     606.842 μs |     10,624.89 μs |  0.74 |    0.04 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float32 |        cpu |      2,811.73 μs |      1,193.470 μs |      65.418 μs |      2,831.67 μs |  0.19 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float32 |        cpu |      2,780.74 μs |      1,862.812 μs |     102.107 μs |      2,737.19 μs |  0.19 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float32 |        cpu |     32,596.26 μs |      1,198.927 μs |      65.717 μs |     32,585.03 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float32 |        cpu |     54,354.12 μs |     15,203.920 μs |     833.378 μs |     54,055.53 μs |  1.67 |    0.02 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float32 |        cpu |     29,951.23 μs |      5,067.654 μs |     277.775 μs |     30,088.64 μs |  0.92 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float32 |        cpu |     28,355.80 μs |     28,337.643 μs |   1,553.282 μs |     27,487.29 μs |  0.87 |    0.05 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float32 |        cpu |     34,338.67 μs |     18,871.301 μs |   1,034.400 μs |     34,600.77 μs |  1.05 |    0.03 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float32 |        cpu |     33,255.58 μs |      2,906.649 μs |     159.323 μs |     33,321.43 μs |  1.02 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float32 |        cpu |      9,876.53 μs |      1,718.150 μs |      94.178 μs |      9,832.86 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float32 |        cpu |     12,017.32 μs |      2,130.588 μs |     116.785 μs |     12,078.66 μs |  1.22 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float32 |        cpu |     11,759.84 μs |      5,793.195 μs |     317.545 μs |     11,787.49 μs |  1.19 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float32 |        cpu |     15,865.02 μs |     15,222.238 μs |     834.382 μs |     15,433.57 μs |  1.61 |    0.09 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float32 |        cpu |      7,582.25 μs |      4,803.228 μs |     263.281 μs |      7,617.59 μs |  0.77 |    0.03 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float32 |        cpu |      7,870.96 μs |      2,443.251 μs |     133.923 μs |      7,828.01 μs |  0.80 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float32 |        cpu |     18,705.03 μs |        914.322 μs |      50.117 μs |     18,732.41 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float32 |        cpu |     21,038.88 μs |     12,178.302 μs |     667.534 μs |     20,976.90 μs |  1.12 |    0.03 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float32 |        cpu |     23,229.08 μs |     11,686.618 μs |     640.583 μs |     23,107.21 μs |  1.24 |    0.03 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float32 |        cpu |     27,548.12 μs |      6,695.224 μs |     366.988 μs |     27,356.13 μs |  1.47 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float32 |        cpu |      2,768.66 μs |      1,916.645 μs |     105.058 μs |      2,759.33 μs |  0.15 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float32 |        cpu |     15,730.49 μs |      3,753.177 μs |     205.724 μs |     15,828.23 μs |  0.84 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float32 |        cpu |     10,838.72 μs |      2,766.772 μs |     151.656 μs |     10,905.55 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float32 |        cpu |     11,517.04 μs |      1,857.477 μs |     101.815 μs |     11,489.43 μs |  1.06 |    0.02 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float32 |        cpu |     11,998.03 μs |      9,790.443 μs |     536.647 μs |     12,091.89 μs |  1.11 |    0.06 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float32 |        cpu |     42,467.48 μs |      4,129.670 μs |     226.361 μs |     42,445.15 μs |  3.92 |    0.04 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float32 |        cpu |      7,549.27 μs |      6,757.194 μs |     370.385 μs |      7,338.50 μs |  0.70 |    0.04 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float32 |        cpu |     23,914.95 μs |     16,915.158 μs |     927.177 μs |     24,229.83 μs |  2.21 |    0.11 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float32 |        cpu |      5,839.84 μs |        335.243 μs |      18.376 μs |      5,837.35 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float32 |        cpu |      4,560.19 μs |      1,422.883 μs |      77.993 μs |      4,536.47 μs |  0.78 |    0.02 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float32 |        cpu |     15,187.27 μs |      7,863.824 μs |     431.043 μs |     14,982.93 μs |  2.60 |    0.08 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float32 |        cpu |      8,291.52 μs |      4,770.729 μs |     261.500 μs |      8,191.38 μs |  1.42 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float32 |        cpu |      2,134.93 μs |      8,520.958 μs |     467.062 μs |      2,025.80 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float32 |        cpu |      3,197.40 μs |      1,957.907 μs |     107.319 μs |      3,224.34 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float32 |        cpu |      3,768.22 μs |      2,338.891 μs |     128.202 μs |      3,756.52 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float32 |        cpu |    224,012.27 μs |    225,062.322 μs |  12,336.425 μs |    230,523.50 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float32 |        cpu |    211,984.83 μs |     72,001.036 μs |   3,946.620 μs |    210,011.80 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float32** |       **cuda** |     **53,747.47 μs** |      **4,071.099 μs** |     **223.151 μs** |     **53,826.97 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float32 |       cuda |      1,161.87 μs |        589.993 μs |      32.340 μs |      1,152.70 μs | 0.022 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float32 |       cuda |     28,248.80 μs |     55,943.120 μs |   3,066.431 μs |     26,485.60 μs | 0.525 |    0.06 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float32 |       cuda |     34,517.47 μs |     15,865.265 μs |     869.629 μs |     34,353.80 μs | 0.642 |    0.02 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float32 |       cuda |        168.06 μs |         47.762 μs |       2.618 μs |        168.86 μs | 0.003 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float32 |       cuda |        174.50 μs |        118.310 μs |       6.485 μs |        173.24 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float32 |       cuda |     40,960.74 μs |        744.474 μs |      40.807 μs |     40,968.18 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float32 |       cuda |     20,899.30 μs |      4,423.462 μs |     242.465 μs |     20,896.20 μs |  0.51 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float32 |       cuda |     22,855.80 μs |      6,000.382 μs |     328.901 μs |     22,849.10 μs |  0.56 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float32 |       cuda |     23,297.87 μs |     13,045.653 μs |     715.076 μs |     23,132.00 μs |  0.57 |    0.02 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float32 |       cuda |      1,335.50 μs |      2,119.112 μs |     116.156 μs |      1,389.68 μs |  0.03 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float32 |       cuda |      1,211.02 μs |        192.698 μs |      10.562 μs |      1,206.61 μs |  0.03 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float32 |       cuda |     41,631.18 μs |      1,736.414 μs |      95.179 μs |     41,651.70 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float32 |       cuda |     20,478.30 μs |      2,272.718 μs |     124.575 μs |     20,529.40 μs |  0.49 |    0.00 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float32 |       cuda |     23,654.90 μs |     13,686.149 μs |     750.184 μs |     23,295.20 μs |  0.57 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float32 |       cuda |     30,894.07 μs |     58,035.570 μs |   3,181.125 μs |     29,965.70 μs |  0.74 |    0.08 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float32 |       cuda |      2,778.94 μs |      2,613.812 μs |     143.272 μs |      2,700.89 μs |  0.07 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float32 |       cuda |      2,766.33 μs |      1,599.778 μs |      87.689 μs |      2,727.29 μs |  0.07 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float32 |       cuda |     45,680.17 μs |      4,674.643 μs |     256.233 μs |     45,804.61 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float32 |       cuda |     22,546.80 μs |      4,410.721 μs |     241.766 μs |     22,452.00 μs |  0.49 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float32 |       cuda |     25,393.03 μs |     18,606.067 μs |   1,019.861 μs |     25,136.40 μs |  0.56 |    0.02 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float32 |       cuda |     24,140.37 μs |      8,371.596 μs |     458.875 μs |     24,371.30 μs |  0.53 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float32 |       cuda |     32,589.90 μs |      1,147.672 μs |      62.908 μs |     32,577.25 μs |  0.71 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float32 |       cuda |     38,075.80 μs |      8,597.150 μs |     471.239 μs |     38,342.84 μs |  0.83 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float32 |       cuda |     27,778.90 μs |        728.736 μs |      39.944 μs |     27,763.09 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float32 |       cuda |     17,372.47 μs |      1,635.355 μs |      89.639 μs |     17,418.00 μs |  0.63 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float32 |       cuda |     20,661.83 μs |     42,465.420 μs |   2,327.673 μs |     19,578.70 μs |  0.74 |    0.08 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float32 |       cuda |     27,389.77 μs |      2,960.234 μs |     162.260 μs |     27,458.20 μs |  0.99 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float32 |       cuda |      7,614.46 μs |      3,603.945 μs |     197.544 μs |      7,716.47 μs |  0.27 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float32 |       cuda |      8,845.77 μs |      5,288.569 μs |     289.884 μs |      8,691.02 μs |  0.32 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float32 |       cuda |     34,601.02 μs |        955.139 μs |      52.354 μs |     34,616.02 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float32 |       cuda |     23,412.90 μs |      7,166.639 μs |     392.828 μs |     23,261.50 μs |  0.68 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float32 |       cuda |     28,366.27 μs |     54,830.961 μs |   3,005.470 μs |     26,957.10 μs |  0.82 |    0.09 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float32 |       cuda |     60,442.07 μs |    179,577.309 μs |   9,843.238 μs |     65,815.40 μs |  1.75 |    0.29 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float32 |       cuda |      2,696.22 μs |      1,117.873 μs |      61.274 μs |      2,665.68 μs |  0.08 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float32 |       cuda |     14,071.69 μs |      1,601.689 μs |      87.794 μs |     14,107.73 μs |  0.41 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float32 |       cuda |     28,667.97 μs |      1,528.006 μs |      83.755 μs |     28,713.27 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float32 |       cuda |     18,505.27 μs |      9,602.018 μs |     526.319 μs |     18,632.80 μs |  0.65 |    0.02 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float32 |       cuda |     21,567.07 μs |     17,741.258 μs |     972.458 μs |     21,611.40 μs |  0.75 |    0.03 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float32 |       cuda |     59,958.13 μs |        871.520 μs |      47.771 μs |     59,982.40 μs |  2.09 |    0.01 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float32 |       cuda |      7,705.08 μs |      3,912.753 μs |     214.471 μs |      7,601.36 μs |  0.27 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float32 |       cuda |     24,427.78 μs |      4,921.195 μs |     269.747 μs |     24,533.67 μs |  0.85 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float32 |       cuda |     14,700.73 μs |        609.284 μs |      33.397 μs |     14,698.67 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float32 |       cuda |     10,573.93 μs |      7,772.261 μs |     426.024 μs |     10,488.50 μs |  0.72 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float32 |       cuda |     29,343.33 μs |     58,561.263 μs |   3,209.940 μs |     27,540.50 μs |  2.00 |    0.22 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float32 |       cuda |      7,865.77 μs |      4,216.660 μs |     231.129 μs |      7,919.26 μs |  0.54 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float32 |       cuda |      4,154.13 μs |      7,056.182 μs |     386.773 μs |      3,946.10 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float32 |       cuda |      2,984.47 μs |      1,696.554 μs |      92.994 μs |      2,937.20 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float32 |       cuda |      3,719.50 μs |      1,805.679 μs |      98.975 μs |      3,726.30 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float32 |       cuda |    212,570.27 μs |     71,465.709 μs |   3,917.277 μs |    214,476.70 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float32 |       cuda |    230,206.83 μs |    144,195.735 μs |   7,903.855 μs |    229,678.50 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float64** |        **cpu** |     **29,764.01 μs** |      **2,681.413 μs** |     **146.977 μs** |     **29,838.35 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float64 |        cpu |      1,638.19 μs |        568.467 μs |      31.160 μs |      1,637.58 μs | 0.055 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float64 |        cpu |      4,215.24 μs |      1,341.665 μs |      73.541 μs |      4,173.31 μs | 0.142 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float64 |        cpu |      4,466.49 μs |        908.832 μs |      49.816 μs |      4,442.10 μs | 0.150 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float64 |        cpu |        172.89 μs |        101.806 μs |       5.580 μs |        174.65 μs | 0.006 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float64 |        cpu |        181.78 μs |         77.040 μs |       4.223 μs |        181.07 μs | 0.006 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float64 |        cpu |     17,719.80 μs |      1,988.886 μs |     109.018 μs |     17,722.91 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float64 |        cpu |     12,026.32 μs |      6,286.219 μs |     344.569 μs |     12,146.70 μs |  0.68 |    0.02 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float64 |        cpu |     20,187.86 μs |      7,793.077 μs |     427.165 μs |     20,417.44 μs |  1.14 |    0.02 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float64 |        cpu |     18,904.35 μs |     18,753.177 μs |   1,027.925 μs |     19,201.26 μs |  1.07 |    0.05 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float64 |        cpu |      2,075.19 μs |      1,575.994 μs |      86.386 μs |      2,039.02 μs |  0.12 |    0.01 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float64 |        cpu |      2,170.28 μs |      1,477.460 μs |      80.985 μs |      2,212.43 μs |  0.12 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float64 |        cpu |     18,621.40 μs |      1,493.119 μs |      81.843 μs |     18,601.33 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float64 |        cpu |     11,849.47 μs |     11,473.635 μs |     628.909 μs |     12,164.05 μs |  0.64 |    0.04 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float64 |        cpu |     19,161.31 μs |      8,782.024 μs |     481.372 μs |     19,358.90 μs |  1.03 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float64 |        cpu |     18,774.24 μs |      3,759.519 μs |     206.072 μs |     18,824.86 μs |  1.01 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float64 |        cpu |      3,569.87 μs |        933.088 μs |      51.146 μs |      3,559.08 μs |  0.19 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float64 |        cpu |      3,653.54 μs |      3,150.834 μs |     172.708 μs |      3,601.33 μs |  0.20 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float64 |        cpu |     54,766.78 μs |      1,954.844 μs |     107.152 μs |     54,724.54 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float64 |        cpu |     28,384.20 μs |     17,058.296 μs |     935.023 μs |     28,752.99 μs |  0.52 |    0.02 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float64 |        cpu |     54,669.03 μs |     10,737.297 μs |     588.548 μs |     54,800.53 μs |  1.00 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float64 |        cpu |     58,061.46 μs |      4,601.552 μs |     252.227 μs |     58,030.71 μs |  1.06 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float64 |        cpu |     33,700.80 μs |      2,527.051 μs |     138.516 μs |     33,733.78 μs |  0.62 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float64 |        cpu |     33,984.72 μs |     17,681.822 μs |     969.200 μs |     33,696.77 μs |  0.62 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float64 |        cpu |     13,672.78 μs |        906.539 μs |      49.690 μs |     13,652.33 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float64 |        cpu |     12,483.62 μs |     12,983.742 μs |     711.683 μs |     12,657.58 μs |  0.91 |    0.05 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float64 |        cpu |     11,570.32 μs |      9,351.750 μs |     512.601 μs |     11,859.51 μs |  0.85 |    0.04 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float64 |        cpu |     15,788.91 μs |      4,001.077 μs |     219.313 μs |     15,788.79 μs |  1.15 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float64 |        cpu |      7,088.09 μs |        486.280 μs |      26.655 μs |      7,073.60 μs |  0.52 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float64 |        cpu |      9,165.32 μs |      2,018.605 μs |     110.647 μs |      9,119.37 μs |  0.67 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float64 |        cpu |     24,655.89 μs |      3,708.216 μs |     203.260 μs |     24,572.30 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float64 |        cpu |     21,718.14 μs |      3,101.938 μs |     170.028 μs |     21,684.26 μs |  0.88 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float64 |        cpu |     22,174.74 μs |      2,772.719 μs |     151.982 μs |     22,098.84 μs |  0.90 |    0.00 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float64 |        cpu |     26,748.23 μs |      8,537.227 μs |     467.954 μs |     26,953.98 μs |  1.08 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float64 |        cpu |      2,667.94 μs |         27.247 μs |       1.493 μs |      2,668.16 μs |  0.11 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float64 |        cpu |     14,077.37 μs |        857.158 μs |      46.984 μs |     14,060.45 μs |  0.57 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float64 |        cpu |     14,725.57 μs |      1,172.262 μs |      64.256 μs |     14,704.66 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float64 |        cpu |     11,546.63 μs |      1,168.326 μs |      64.040 μs |     11,527.74 μs |  0.78 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float64 |        cpu |     11,987.71 μs |     17,414.878 μs |     954.568 μs |     12,431.90 μs |  0.81 |    0.06 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float64 |        cpu |     47,888.23 μs |     42,811.108 μs |   2,346.621 μs |     46,619.30 μs |  3.25 |    0.17 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float64 |        cpu |      7,970.10 μs |        271.391 μs |      14.876 μs |      7,969.10 μs |  0.54 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float64 |        cpu |     21,775.68 μs |     20,315.971 μs |   1,113.587 μs |     21,342.97 μs |  1.48 |    0.07 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float64 |        cpu |      7,815.86 μs |      1,656.133 μs |      90.778 μs |      7,801.54 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float64 |        cpu |      5,289.89 μs |      2,719.167 μs |     149.047 μs |      5,335.48 μs |  0.68 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float64 |        cpu |     15,110.24 μs |      4,373.532 μs |     239.728 μs |     15,053.96 μs |  1.93 |    0.05 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float64 |        cpu |      8,562.51 μs |      2,743.676 μs |     150.390 μs |      8,550.42 μs |  1.10 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float64 |        cpu |      3,817.30 μs |      2,579.721 μs |     141.403 μs |      3,831.33 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float64 |        cpu |      3,656.05 μs |      4,230.901 μs |     231.910 μs |      3,714.14 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float64 |        cpu |      4,453.94 μs |      3,716.396 μs |     203.708 μs |      4,488.45 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float64 |        cpu |    207,806.07 μs |     32,368.065 μs |   1,774.203 μs |    207,334.20 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float64 |        cpu |    208,096.30 μs |     71,941.191 μs |   3,943.339 μs |    206,041.30 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float64** |       **cuda** |     **52,728.88 μs** |      **1,266.705 μs** |      **69.432 μs** |     **52,762.61 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float64 |       cuda |      1,183.23 μs |      1,419.453 μs |      77.805 μs |      1,145.60 μs | 0.022 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float64 |       cuda |     33,293.63 μs |    106,140.697 μs |   5,817.930 μs |     36,594.50 μs | 0.631 |    0.11 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float64 |       cuda |     33,657.00 μs |     97,076.733 μs |   5,321.103 μs |     36,235.30 μs | 0.638 |    0.10 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float64 |       cuda |        170.39 μs |         29.743 μs |       1.630 μs |        169.65 μs | 0.003 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float64 |       cuda |        184.04 μs |         90.670 μs |       4.970 μs |        181.66 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float64 |       cuda |     40,754.82 μs |      3,605.814 μs |     197.647 μs |     40,696.18 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float64 |       cuda |     20,293.93 μs |      1,785.512 μs |      97.870 μs |     20,282.30 μs |  0.50 |    0.00 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float64 |       cuda |     26,553.03 μs |     62,402.179 μs |   3,420.474 μs |     27,276.20 μs |  0.65 |    0.08 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float64 |       cuda |     22,208.70 μs |      1,668.027 μs |      91.430 μs |     22,242.90 μs |  0.54 |    0.00 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float64 |       cuda |      2,023.40 μs |        794.977 μs |      43.575 μs |      1,999.78 μs |  0.05 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float64 |       cuda |      2,096.38 μs |        256.789 μs |      14.075 μs |      2,096.60 μs |  0.05 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float64 |       cuda |     40,510.88 μs |      1,102.800 μs |      60.448 μs |     40,486.92 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float64 |       cuda |     23,086.80 μs |     25,641.493 μs |   1,405.497 μs |     22,643.10 μs |  0.57 |    0.03 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float64 |       cuda |     22,372.20 μs |      3,973.770 μs |     217.816 μs |     22,300.20 μs |  0.55 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float64 |       cuda |     21,689.77 μs |      5,298.793 μs |     290.445 μs |     21,832.20 μs |  0.54 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float64 |       cuda |      3,622.96 μs |        706.600 μs |      38.731 μs |      3,621.54 μs |  0.09 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float64 |       cuda |      3,554.60 μs |      1,148.622 μs |      62.960 μs |      3,561.56 μs |  0.09 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float64 |       cuda |     42,743.74 μs |      2,339.710 μs |     128.247 μs |     42,683.40 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float64 |       cuda |     21,670.03 μs |      7,339.277 μs |     402.291 μs |     21,881.70 μs |  0.51 |    0.01 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float64 |       cuda |     23,903.70 μs |      2,305.166 μs |     126.354 μs |     23,951.10 μs |  0.56 |    0.00 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float64 |       cuda |     30,937.90 μs |    118,801.546 μs |   6,511.914 μs |     29,351.30 μs |  0.72 |    0.15 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float64 |       cuda |     38,586.30 μs |      6,686.355 μs |     366.502 μs |     38,581.54 μs |  0.90 |    0.01 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float64 |       cuda |     33,397.05 μs |      2,592.607 μs |     142.110 μs |     33,412.40 μs |  0.78 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float64 |       cuda |     26,676.37 μs |      3,150.344 μs |     172.681 μs |     26,692.09 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float64 |       cuda |     20,685.60 μs |     47,921.719 μs |   2,626.751 μs |     22,195.10 μs |  0.78 |    0.09 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float64 |       cuda |     19,851.70 μs |      2,403.736 μs |     131.757 μs |     19,784.60 μs |  0.74 |    0.00 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float64 |       cuda |     29,081.20 μs |      3,338.605 μs |     183.000 μs |     29,130.50 μs |  1.09 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float64 |       cuda |      7,227.12 μs |      3,077.957 μs |     168.713 μs |      7,164.37 μs |  0.27 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float64 |       cuda |      9,247.62 μs |      3,830.156 μs |     209.944 μs |      9,147.54 μs |  0.35 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float64 |       cuda |     34,619.48 μs |        912.255 μs |      50.004 μs |     34,632.95 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float64 |       cuda |     29,552.37 μs |    121,739.520 μs |   6,672.954 μs |     25,718.40 μs |  0.85 |    0.19 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float64 |       cuda |     27,363.13 μs |      5,955.145 μs |     326.422 μs |     27,280.60 μs |  0.79 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float64 |       cuda |     62,912.50 μs |    268,401.247 μs |  14,711.978 μs |     62,515.00 μs |  1.82 |    0.42 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float64 |       cuda |      2,747.89 μs |        828.723 μs |      45.425 μs |      2,725.87 μs |  0.08 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float64 |       cuda |     14,684.15 μs |      5,821.332 μs |     319.087 μs |     14,685.60 μs |  0.42 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float64 |       cuda |     27,689.05 μs |        722.083 μs |      39.580 μs |     27,698.04 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float64 |       cuda |     21,760.57 μs |     47,521.388 μs |   2,604.808 μs |     22,374.60 μs |  0.79 |    0.09 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float64 |       cuda |     21,044.43 μs |      6,952.892 μs |     381.111 μs |     20,825.80 μs |  0.76 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float64 |       cuda |     61,311.10 μs |     25,558.139 μs |   1,400.928 μs |     61,485.10 μs |  2.21 |    0.05 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float64 |       cuda |      7,514.39 μs |      4,004.532 μs |     219.502 μs |      7,550.24 μs |  0.27 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float64 |       cuda |     23,102.70 μs |     19,466.763 μs |   1,067.039 μs |     22,664.49 μs |  0.83 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float64 |       cuda |     14,854.97 μs |      2,169.453 μs |     118.915 μs |     14,872.00 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float64 |       cuda |     12,739.20 μs |     70,125.543 μs |   3,843.818 μs |     10,737.00 μs |  0.86 |    0.26 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float64 |       cuda |     28,576.50 μs |      4,161.836 μs |     228.124 μs |     28,546.10 μs |  1.92 |    0.02 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float64 |       cuda |      8,708.88 μs |     11,114.121 μs |     609.202 μs |      8,415.22 μs |  0.59 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float64 |       cuda |      3,110.63 μs |      4,906.337 μs |     268.933 μs |      3,173.30 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float64 |       cuda |      2,314.37 μs |        941.923 μs |      51.630 μs |      2,294.90 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float64 |       cuda |      3,266.10 μs |      2,946.961 μs |     161.533 μs |      3,187.40 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float64 |       cuda |    207,305.27 μs |      1,794.057 μs |      98.338 μs |    207,272.10 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float64 |       cuda |    226,949.20 μs |    193,999.864 μs |  10,633.787 μs |    226,389.90 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |     **int32** |        **cpu** |     **22,752.66 μs** |      **2,652.716 μs** |     **145.404 μs** |     **22,749.19 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |     int32 |        cpu |      1,606.63 μs |      1,024.351 μs |      56.148 μs |      1,618.73 μs | 0.071 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |     int32 |        cpu |      3,818.53 μs |      2,498.531 μs |     136.953 μs |      3,848.13 μs | 0.168 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |     int32 |        cpu |      4,017.43 μs |      5,268.845 μs |     288.803 μs |      3,878.63 μs | 0.177 |    0.01 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |     int32 |        cpu |        115.38 μs |          8.677 μs |       0.476 μs |        115.31 μs | 0.005 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |     int32 |        cpu |        118.61 μs |         42.882 μs |       2.350 μs |        118.61 μs | 0.005 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |     int32 |        cpu |     15,816.29 μs |      1,958.180 μs |     107.334 μs |     15,764.18 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |     int32 |        cpu |     11,318.04 μs |      2,288.815 μs |     125.458 μs |     11,272.83 μs |  0.72 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |     int32 |        cpu |      9,756.91 μs |      2,418.347 μs |     132.558 μs |      9,812.92 μs |  0.62 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |     int32 |        cpu |     10,374.15 μs |      5,060.798 μs |     277.399 μs |     10,389.30 μs |  0.66 |    0.02 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |     int32 |        cpu |      1,145.95 μs |        450.258 μs |      24.680 μs |      1,135.27 μs |  0.07 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |     int32 |        cpu |      1,227.16 μs |      2,135.389 μs |     117.048 μs |      1,162.09 μs |  0.08 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |       2048 |     int32 |        cpu |     13,758.93 μs |      1,751.455 μs |      96.003 μs |     13,749.37 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |     int32 |        cpu |     11,533.07 μs |      9,252.392 μs |     507.155 μs |     11,311.00 μs |  0.84 |    0.04 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |     int32 |        cpu |      9,822.98 μs |      5,529.880 μs |     303.111 μs |      9,693.62 μs |  0.71 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |       2048 |     int32 |        cpu |     10,545.33 μs |      6,225.110 μs |     341.219 μs |     10,383.06 μs |  0.77 |    0.03 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |     int32 |        cpu |      2,656.32 μs |        478.175 μs |      26.210 μs |      2,666.52 μs |  0.19 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |     int32 |        cpu |      2,698.70 μs |        729.211 μs |      39.971 μs |      2,692.00 μs |  0.20 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |       2048 |     int32 |        cpu |     41,998.92 μs |     31,302.180 μs |   1,715.778 μs |     41,239.09 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |       2048 |     int32 |        cpu |      8,809.91 μs |        714.506 μs |      39.164 μs |      8,821.52 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |     int32 |        cpu |     12,216.55 μs |      3,727.926 μs |     204.340 μs |     12,197.36 μs |  1.39 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |     int32 |        cpu |     11,440.68 μs |      1,272.365 μs |      69.743 μs |     11,471.61 μs |  1.30 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |       2048 |     int32 |        cpu |     14,506.82 μs |      5,947.732 μs |     326.015 μs |     14,470.29 μs |  1.65 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |     int32 |        cpu |      6,466.43 μs |      2,921.984 μs |     160.164 μs |      6,412.81 μs |  0.73 |    0.02 |       No |
|        addition_Tensor_Reference |     addition |       2048 |     int32 |        cpu |      7,601.38 μs |        907.337 μs |      49.734 μs |      7,576.28 μs |  0.86 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |     int32 |        cpu |     18,619.54 μs |        302.630 μs |      16.588 μs |     18,617.89 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |     int32 |        cpu |     20,575.35 μs |      6,664.730 μs |     365.316 μs |     20,560.42 μs |  1.11 |    0.02 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |     int32 |        cpu |     22,048.36 μs |     14,983.942 μs |     821.320 μs |     21,606.04 μs |  1.18 |    0.04 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |     int32 |        cpu |     53,742.53 μs |     24,196.328 μs |   1,326.282 μs |     53,117.90 μs |  2.89 |    0.07 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |     int32 |        cpu |      3,017.90 μs |        758.008 μs |      41.549 μs |      3,008.10 μs |  0.16 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |     int32 |        cpu |    169,857.83 μs |    392,539.947 μs |  21,516.439 μs |    160,907.90 μs |  9.12 |    1.16 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |     int32 |        cpu |      9,823.81 μs |        276.627 μs |      15.163 μs |      9,825.30 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |     int32 |        cpu |     11,545.38 μs |        754.831 μs |      41.375 μs |     11,535.80 μs |  1.18 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |     int32 |        cpu |     11,361.45 μs |      4,859.567 μs |     266.369 μs |     11,462.12 μs |  1.16 |    0.03 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |     int32 |        cpu |     44,850.53 μs |     36,370.250 μs |   1,993.576 μs |     44,334.71 μs |  4.57 |    0.21 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |     int32 |        cpu |      6,552.18 μs |      1,022.571 μs |      56.051 μs |      6,533.71 μs |  0.67 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |     int32 |        cpu |    280,820.00 μs |     62,420.686 μs |   3,421.488 μs |    281,585.20 μs | 28.59 |    0.37 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |     int32 |        cpu |      5,834.87 μs |        368.266 μs |      20.186 μs |      5,833.20 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |     int32 |        cpu |      4,730.71 μs |      3,732.511 μs |     204.592 μs |      4,661.10 μs |  0.81 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |     int32 |        cpu |     14,621.00 μs |     10,199.889 μs |     559.090 μs |     14,542.47 μs |  2.51 |    0.09 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |     int32 |        cpu |      7,584.93 μs |      2,432.063 μs |     133.310 μs |      7,520.08 μs |  1.30 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |     int32 |        cpu |      6,899.88 μs |        766.468 μs |      42.013 μs |      6,885.00 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |     int32 |        cpu |      6,898.77 μs |      1,959.271 μs |     107.394 μs |      6,926.47 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |     int32 |        cpu |      7,109.71 μs |      1,740.508 μs |      95.403 μs |      7,143.79 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |     int32 |        cpu |    218,660.07 μs |     86,496.971 μs |   4,741.191 μs |    219,911.80 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |     int32 |        cpu |    214,073.20 μs |     21,202.509 μs |   1,162.181 μs |    213,439.30 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |     **int32** |       **cuda** |     **48,987.93 μs** |         **52.011 μs** |       **2.851 μs** |     **48,989.07 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |     int32 |       cuda |      1,133.13 μs |        193.576 μs |      10.611 μs |      1,136.30 μs | 0.023 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |     int32 |       cuda |     30,391.70 μs |     96,603.217 μs |   5,295.148 μs |     31,561.40 μs | 0.620 |    0.11 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |     int32 |       cuda |     25,680.50 μs |      6,853.480 μs |     375.662 μs |     25,578.60 μs | 0.524 |    0.01 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |     int32 |       cuda |        118.09 μs |         66.203 μs |       3.629 μs |        119.83 μs | 0.002 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |     int32 |       cuda |        118.58 μs |         52.773 μs |       2.893 μs |        117.28 μs | 0.002 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |     int32 |       cuda |     42,981.71 μs |        365.621 μs |      20.041 μs |     42,989.87 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |     int32 |       cuda |     20,943.43 μs |      6,405.306 μs |     351.096 μs |     20,853.00 μs |  0.49 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |     int32 |       cuda |     23,445.80 μs |     39,612.147 μs |   2,171.275 μs |     22,217.80 μs |  0.55 |    0.05 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |     int32 |       cuda |     23,127.77 μs |     13,918.531 μs |     762.922 μs |     22,735.70 μs |  0.54 |    0.02 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |     int32 |       cuda |      1,182.80 μs |        470.019 μs |      25.763 μs |      1,180.66 μs |  0.03 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |     int32 |       cuda |      1,165.19 μs |        526.365 μs |      28.852 μs |      1,161.36 μs |  0.03 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |       2048 |     int32 |       cuda |     39,974.16 μs |        657.318 μs |      36.030 μs |     39,991.72 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |     int32 |       cuda |     21,299.33 μs |      7,249.936 μs |     397.393 μs |     21,078.40 μs |  0.53 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |     int32 |       cuda |     26,865.80 μs |      4,108.517 μs |     225.202 μs |     26,864.80 μs |  0.67 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |     int32 |       cuda |     23,044.73 μs |      6,696.143 μs |     367.038 μs |     23,071.60 μs |  0.58 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |     int32 |       cuda |      2,940.18 μs |      2,275.696 μs |     124.739 μs |      3,004.20 μs |  0.07 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |     int32 |       cuda |      3,026.56 μs |        555.579 μs |      30.453 μs |      3,042.42 μs |  0.08 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |       2048 |     int32 |       cuda |     31,742.87 μs |     21,862.641 μs |   1,198.365 μs |     32,160.80 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |       2048 |     int32 |       cuda |     26,844.33 μs |        535.962 μs |      29.378 μs |     26,833.22 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |     int32 |       cuda |     21,281.23 μs |     43,299.652 μs |   2,373.400 μs |     22,443.60 μs |  0.79 |    0.09 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |     int32 |       cuda |     20,071.63 μs |     12,360.890 μs |     677.542 μs |     20,374.40 μs |  0.75 |    0.03 |       No |
|            addition_Tensor_Torch |     addition |       2048 |     int32 |       cuda |     29,302.63 μs |     15,165.103 μs |     831.250 μs |     29,354.20 μs |  1.09 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |     int32 |       cuda |      6,401.00 μs |      1,095.248 μs |      60.034 μs |      6,419.06 μs |  0.24 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |       2048 |     int32 |       cuda |      7,626.22 μs |      1,567.315 μs |      85.910 μs |      7,614.89 μs |  0.28 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |     int32 |       cuda |     36,991.75 μs |         79.430 μs |       4.354 μs |     36,994.01 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |     int32 |       cuda |     28,619.03 μs |     43,626.036 μs |   2,391.290 μs |     27,719.70 μs |  0.77 |    0.06 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |     int32 |       cuda |     29,361.80 μs |     18,280.342 μs |   1,002.007 μs |     29,716.00 μs |  0.79 |    0.03 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |     int32 |       cuda |    106,777.43 μs |     81,636.893 μs |   4,474.794 μs |    108,940.30 μs |  2.89 |    0.12 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |     int32 |       cuda |      3,188.23 μs |      1,756.078 μs |      96.257 μs |      3,148.29 μs |  0.09 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |     int32 |       cuda |    154,438.07 μs |     25,446.473 μs |   1,394.807 μs |    154,541.10 μs |  4.17 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |     int32 |       cuda |     28,601.03 μs |        297.682 μs |      16.317 μs |     28,608.34 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |     int32 |       cuda |     17,376.27 μs |      5,552.960 μs |     304.376 μs |     17,317.90 μs |  0.61 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |     int32 |       cuda |     24,356.90 μs |      3,885.891 μs |     212.999 μs |     24,442.60 μs |  0.85 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |     int32 |       cuda |     58,062.60 μs |      9,363.904 μs |     513.267 μs |     57,824.30 μs |  2.03 |    0.02 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |     int32 |       cuda |      6,585.36 μs |      4,039.376 μs |     221.412 μs |      6,462.12 μs |  0.23 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |     int32 |       cuda |    285,740.33 μs |     77,052.935 μs |   4,223.531 μs |    285,319.80 μs |  9.99 |    0.14 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |     int32 |       cuda |     14,987.66 μs |         59.345 μs |       3.253 μs |     14,986.34 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |     int32 |       cuda |     11,255.77 μs |      2,621.214 μs |     143.678 μs |     11,214.00 μs |  0.75 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |     int32 |       cuda |     30,769.23 μs |     41,369.956 μs |   2,267.627 μs |     29,882.70 μs |  2.05 |    0.15 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |     int32 |       cuda |      7,772.94 μs |      2,225.753 μs |     122.001 μs |      7,818.22 μs |  0.52 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |     int32 |       cuda |     17,638.47 μs |     37,762.728 μs |   2,069.903 μs |     17,655.50 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |     int32 |       cuda |     20,673.37 μs |     48,212.843 μs |   2,642.709 μs |     20,706.80 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |     int32 |       cuda |    305,013.27 μs |  1,373,155.871 μs |  75,267.307 μs |    289,966.70 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |     int32 |       cuda |    364,420.50 μs |    768,749.725 μs |  42,137.766 μs |    388,737.80 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float32** |        **cpu** |     **30,778.81 μs** |        **684.207 μs** |      **37.504 μs** |     **30,789.03 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float32 |        cpu |         80.07 μs |         94.045 μs |       5.155 μs |         79.20 μs | 0.003 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float32 |        cpu |      2,314.73 μs |      1,137.806 μs |      62.367 μs |      2,285.90 μs | 0.075 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float32 |        cpu |      1,819.24 μs |        917.850 μs |      50.310 μs |      1,809.99 μs | 0.059 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float32 |        cpu |      2,900.80 μs |     10,357.036 μs |     567.704 μs |      2,601.69 μs | 0.094 |    0.02 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float32 |        cpu |      3,041.41 μs |      7,198.424 μs |     394.570 μs |      2,892.54 μs | 0.099 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float32 |        cpu |      1,975.94 μs |        371.522 μs |      20.364 μs |      1,964.61 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float32 |        cpu |      9,584.74 μs |     11,206.167 μs |     614.248 μs |      9,817.08 μs |  4.85 |    0.30 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float32 |        cpu |      4,874.97 μs |      1,601.397 μs |      87.778 μs |      4,838.50 μs |  2.47 |    0.06 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float32 |        cpu |      4,345.97 μs |     11,288.253 μs |     618.747 μs |      4,060.60 μs |  2.20 |    0.29 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float32 |        cpu |      7,849.97 μs |      5,962.065 μs |     326.801 μs |      7,824.40 μs |  3.97 |    0.20 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float32 |        cpu |      4,770.64 μs |      5,419.123 μs |     297.040 μs |      4,871.73 μs |  2.41 |    0.13 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float32 |        cpu |      1,993.79 μs |         60.620 μs |       3.323 μs |      1,994.02 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float32 |        cpu |      8,836.83 μs |      4,855.400 μs |     266.141 μs |      8,971.82 μs |  4.43 |    0.13 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float32 |        cpu |      5,007.45 μs |        906.602 μs |      49.694 μs |      5,031.97 μs |  2.51 |    0.03 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float32 |        cpu |      5,029.19 μs |        930.218 μs |      50.988 μs |      5,042.92 μs |  2.52 |    0.03 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float32 |        cpu |      5,861.94 μs |        500.091 μs |      27.412 μs |      5,854.94 μs |  2.94 |    0.01 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float32 |        cpu |      5,987.51 μs |      1,890.393 μs |     103.619 μs |      5,984.17 μs |  3.00 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float32 |        cpu |     18,987.45 μs |        278.971 μs |      15.291 μs |     18,995.71 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float32 |        cpu |     44,709.06 μs |      3,557.003 μs |     194.971 μs |     44,771.39 μs |  2.35 |    0.01 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float32 |        cpu |     21,501.32 μs |     10,035.148 μs |     550.060 μs |     21,774.98 μs |  1.13 |    0.03 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float32 |        cpu |     21,068.77 μs |      4,451.034 μs |     243.976 μs |     20,985.70 μs |  1.11 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float32 |        cpu |     35,857.46 μs |      3,289.561 μs |     180.312 μs |     35,776.35 μs |  1.89 |    0.01 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float32 |        cpu |     37,064.62 μs |     11,404.804 μs |     625.136 μs |     36,825.15 μs |  1.95 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float32 |        cpu |      7,792.03 μs |        724.859 μs |      39.732 μs |      7,799.49 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float32 |        cpu |      6,167.36 μs |      1,284.692 μs |      70.418 μs |      6,171.06 μs |  0.79 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float32 |        cpu |      6,341.41 μs |      1,643.783 μs |      90.101 μs |      6,354.98 μs |  0.81 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float32 |        cpu |      6,567.52 μs |      3,957.991 μs |     216.951 μs |      6,654.16 μs |  0.84 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float32 |        cpu |     10,116.10 μs |        494.010 μs |      27.078 μs |     10,109.79 μs |  1.30 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float32 |        cpu |     10,625.71 μs |      7,778.306 μs |     426.355 μs |     10,422.18 μs |  1.36 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float32 |        cpu |      7,981.71 μs |        162.497 μs |       8.907 μs |      7,977.82 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float32 |        cpu |      6,952.09 μs |      6,653.382 μs |     364.694 μs |      6,905.03 μs |  0.87 |    0.04 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float32 |        cpu |      6,947.15 μs |      1,797.648 μs |      98.535 μs |      6,897.19 μs |  0.87 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float32 |        cpu |      7,478.77 μs |      1,604.285 μs |      87.936 μs |      7,452.48 μs |  0.94 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float32 |        cpu |      5,809.86 μs |      1,387.154 μs |      76.035 μs |      5,785.02 μs |  0.73 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float32 |        cpu |     15,271.46 μs |      3,537.637 μs |     193.910 μs |     15,237.13 μs |  1.91 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float32 |        cpu |      7,985.27 μs |        148.570 μs |       8.144 μs |      7,985.52 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float32 |        cpu |      6,464.39 μs |      2,105.370 μs |     115.402 μs |      6,427.25 μs |  0.81 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float32 |        cpu |      6,427.45 μs |      1,685.687 μs |      92.398 μs |      6,457.87 μs |  0.80 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float32 |        cpu |     13,388.01 μs |     10,187.236 μs |     558.397 μs |     13,089.07 μs |  1.68 |    0.07 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float32 |        cpu |     10,173.46 μs |      3,180.373 μs |     174.327 μs |     10,234.82 μs |  1.27 |    0.02 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float32 |        cpu |     22,613.81 μs |      2,639.575 μs |     144.684 μs |     22,610.36 μs |  2.83 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float32 |        cpu |      6,793.33 μs |        421.462 μs |      23.102 μs |      6,787.91 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float32 |        cpu |      2,359.69 μs |         36.440 μs |       1.997 μs |      2,359.45 μs |  0.35 |    0.00 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float32 |        cpu |      6,645.04 μs |      2,310.426 μs |     126.642 μs |      6,600.42 μs |  0.98 |    0.02 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float32 |        cpu |     10,304.24 μs |      1,243.325 μs |      68.151 μs |     10,270.29 μs |  1.52 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float32 |        cpu |      1,742.47 μs |        197.761 μs |      10.840 μs |      1,747.50 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float32 |        cpu |      1,727.09 μs |        313.236 μs |      17.170 μs |      1,735.99 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float32 |        cpu |      1,652.15 μs |        256.823 μs |      14.077 μs |      1,655.42 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float32 |        cpu |    996,014.17 μs |    167,520.570 μs |   9,182.368 μs |    995,448.70 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float32 |        cpu |    979,389.27 μs |    169,645.540 μs |   9,298.844 μs |    976,923.10 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float32** |       **cuda** |     **31,858.79 μs** |      **1,462.281 μs** |      **80.153 μs** |     **31,846.06 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float32 |       cuda |         79.90 μs |        420.410 μs |      23.044 μs |         84.80 μs | 0.003 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float32 |       cuda |      3,407.17 μs |      4,127.284 μs |     226.230 μs |      3,304.70 μs | 0.107 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float32 |       cuda |      3,340.97 μs |        913.485 μs |      50.071 μs |      3,328.80 μs | 0.105 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float32 |       cuda |      2,572.47 μs |        437.719 μs |      23.993 μs |      2,570.13 μs | 0.081 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float32 |       cuda |      2,630.02 μs |        831.192 μs |      45.560 μs |      2,603.98 μs | 0.083 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float32 |       cuda |      1,973.27 μs |        114.965 μs |       6.302 μs |      1,975.90 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float32 |       cuda |        940.67 μs |      5,182.702 μs |     284.081 μs |        825.20 μs |  0.48 |    0.14 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float32 |       cuda |        802.53 μs |      1,867.512 μs |     102.365 μs |        745.90 μs |  0.41 |    0.05 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float32 |       cuda |        742.33 μs |        131.004 μs |       7.181 μs |        745.60 μs |  0.38 |    0.00 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float32 |       cuda |      3,801.69 μs |        448.316 μs |      24.574 μs |      3,809.23 μs |  1.93 |    0.01 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float32 |       cuda |      3,825.05 μs |        266.041 μs |      14.583 μs |      3,831.66 μs |  1.94 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float32 |       cuda |      1,999.25 μs |         14.879 μs |       0.816 μs |      1,999.43 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float32 |       cuda |      1,037.47 μs |      9,465.626 μs |     518.843 μs |        815.30 μs |  0.52 |    0.26 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float32 |       cuda |        709.10 μs |        253.785 μs |      13.911 μs |        710.60 μs |  0.35 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float32 |       cuda |        825.27 μs |      1,158.289 μs |      63.490 μs |        812.70 μs |  0.41 |    0.03 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float32 |       cuda |      5,777.20 μs |        442.696 μs |      24.266 μs |      5,768.54 μs |  2.89 |    0.01 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float32 |       cuda |      5,748.75 μs |        565.150 μs |      30.978 μs |      5,745.10 μs |  2.88 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float32 |       cuda |      1,999.27 μs |          5.528 μs |       0.303 μs |      1,999.33 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float32 |       cuda |      1,005.77 μs |      7,544.318 μs |     413.530 μs |        801.30 μs |  0.50 |    0.21 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float32 |       cuda |        875.60 μs |        934.613 μs |      51.229 μs |        849.80 μs |  0.44 |    0.03 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float32 |       cuda |        746.07 μs |        406.984 μs |      22.308 μs |        734.20 μs |  0.37 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float32 |       cuda |     38,424.46 μs |      1,968.678 μs |     107.910 μs |     38,378.16 μs | 19.22 |    0.05 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float32 |       cuda |     38,393.26 μs |      8,534.110 μs |     467.783 μs |     38,290.34 μs | 19.20 |    0.23 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float32 |       cuda |      5,997.76 μs |         42.044 μs |       2.305 μs |      5,998.95 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float32 |       cuda |        574.87 μs |        355.235 μs |      19.472 μs |        583.30 μs |  0.10 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float32 |       cuda |        677.67 μs |        679.297 μs |      37.235 μs |        660.40 μs |  0.11 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float32 |       cuda |        898.33 μs |        277.547 μs |      15.213 μs |        889.60 μs |  0.15 |    0.00 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float32 |       cuda |     10,208.04 μs |        855.744 μs |      46.906 μs |     10,199.54 μs |  1.70 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float32 |       cuda |     11,109.03 μs |      5,796.880 μs |     317.747 μs |     10,944.58 μs |  1.85 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float32 |       cuda |      5,939.47 μs |      1,179.012 μs |      64.626 μs |      5,949.61 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float32 |       cuda |        793.57 μs |        435.797 μs |      23.888 μs |        789.30 μs |  0.13 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float32 |       cuda |        843.93 μs |        303.495 μs |      16.636 μs |        838.10 μs |  0.14 |    0.00 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float32 |       cuda |      1,563.47 μs |      1,556.518 μs |      85.318 μs |      1,529.70 μs |  0.26 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float32 |       cuda |      5,743.67 μs |        777.241 μs |      42.603 μs |      5,731.91 μs |  0.97 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float32 |       cuda |     13,461.77 μs |      2,097.576 μs |     114.975 μs |     13,448.20 μs |  2.27 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float32 |       cuda |      5,799.73 μs |        909.356 μs |      49.845 μs |      5,797.80 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float32 |       cuda |        562.33 μs |        294.512 μs |      16.143 μs |        571.20 μs |  0.10 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float32 |       cuda |        684.60 μs |        373.221 μs |      20.458 μs |        693.50 μs |  0.12 |    0.00 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float32 |       cuda |      2,054.97 μs |      1,489.915 μs |      81.667 μs |      2,063.90 μs |  0.35 |    0.02 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float32 |       cuda |     10,054.10 μs |        311.577 μs |      17.079 μs |     10,063.69 μs |  1.73 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float32 |       cuda |     23,531.98 μs |     26,475.298 μs |   1,451.200 μs |     22,763.93 μs |  4.06 |    0.28 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float32 |       cuda |      4,822.24 μs |        544.496 μs |      29.846 μs |      4,832.32 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float32 |       cuda |        408.63 μs |      1,766.473 μs |      96.826 μs |        355.30 μs |  0.08 |    0.02 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float32 |       cuda |      1,499.60 μs |      9,997.637 μs |     548.004 μs |      1,490.70 μs |  0.31 |    0.11 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float32 |       cuda |      9,534.38 μs |        817.113 μs |      44.789 μs |      9,510.67 μs |  1.98 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float32 |       cuda |        566.00 μs |      2,318.243 μs |     127.071 μs |        501.30 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float32 |       cuda |        318.50 μs |        522.570 μs |      28.644 μs |        311.80 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float32 |       cuda |        367.13 μs |      1,539.371 μs |      84.378 μs |        345.90 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float32 |       cuda |    971,655.17 μs |    142,623.112 μs |   7,817.654 μs |    974,991.40 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float32 |       cuda |    986,156.20 μs |    235,083.641 μs |  12,885.728 μs |    990,987.60 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float64** |        **cpu** |     **31,631.03 μs** |      **1,391.178 μs** |      **76.255 μs** |     **31,673.89 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float64 |        cpu |         44.42 μs |         24.142 μs |       1.323 μs |         44.62 μs | 0.001 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float64 |        cpu |      2,336.08 μs |        589.110 μs |      32.291 μs |      2,338.25 μs | 0.074 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float64 |        cpu |      2,453.04 μs |        500.767 μs |      27.449 μs |      2,441.24 μs | 0.078 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float64 |        cpu |      2,758.40 μs |        441.125 μs |      24.180 μs |      2,768.84 μs | 0.087 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float64 |        cpu |      2,625.84 μs |         90.368 μs |       4.953 μs |      2,626.75 μs | 0.083 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float64 |        cpu |      2,875.85 μs |        178.734 μs |       9.797 μs |      2,875.67 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float64 |        cpu |      5,028.11 μs |      1,222.153 μs |      66.990 μs |      5,058.02 μs |  1.75 |    0.03 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float64 |        cpu |      8,958.19 μs |      6,275.802 μs |     343.998 μs |      8,869.19 μs |  3.11 |    0.12 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float64 |        cpu |      9,378.07 μs |      4,080.912 μs |     223.689 μs |      9,393.13 μs |  3.26 |    0.07 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float64 |        cpu |      4,640.28 μs |      1,213.245 μs |      66.502 μs |      4,636.49 μs |  1.61 |    0.02 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float64 |        cpu |      4,606.48 μs |        863.927 μs |      47.355 μs |      4,610.46 μs |  1.60 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float64 |        cpu |      3,891.36 μs |        864.843 μs |      47.405 μs |      3,877.75 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float64 |        cpu |      4,914.69 μs |      2,253.191 μs |     123.505 μs |      4,909.28 μs |  1.26 |    0.04 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float64 |        cpu |      8,736.43 μs |      1,808.314 μs |      99.120 μs |      8,680.98 μs |  2.25 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float64 |        cpu |      9,015.22 μs |      6,767.578 μs |     370.954 μs |      8,898.84 μs |  2.32 |    0.10 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float64 |        cpu |      7,418.46 μs |        832.829 μs |      45.650 μs |      7,392.94 μs |  1.91 |    0.03 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float64 |        cpu |      7,589.80 μs |        948.739 μs |      52.004 μs |      7,614.80 μs |  1.95 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float64 |        cpu |     38,602.93 μs |      1,471.718 μs |      80.670 μs |     38,583.90 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float64 |        cpu |     21,213.49 μs |      4,946.823 μs |     271.152 μs |     21,233.08 μs |  0.55 |    0.01 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float64 |        cpu |     44,904.60 μs |     13,937.564 μs |     763.965 μs |     44,732.63 μs |  1.16 |    0.02 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float64 |        cpu |     44,959.14 μs |      7,957.479 μs |     436.176 μs |     44,813.77 μs |  1.16 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float64 |        cpu |     40,297.30 μs |      3,257.859 μs |     178.574 μs |     40,395.33 μs |  1.04 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float64 |        cpu |     38,228.11 μs |     22,891.828 μs |   1,254.778 μs |     37,614.82 μs |  0.99 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float64 |        cpu |     11,906.51 μs |      2,049.491 μs |     112.340 μs |     11,939.13 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float64 |        cpu |      6,527.57 μs |      4,108.706 μs |     225.212 μs |      6,504.46 μs |  0.55 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float64 |        cpu |      6,520.87 μs |      2,021.927 μs |     110.829 μs |      6,570.11 μs |  0.55 |    0.00 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float64 |        cpu |      6,594.14 μs |      2,111.668 μs |     115.748 μs |      6,627.04 μs |  0.55 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float64 |        cpu |     10,165.23 μs |      1,906.322 μs |     104.492 μs |     10,171.37 μs |  0.85 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float64 |        cpu |     10,271.14 μs |      1,842.767 μs |     101.008 μs |     10,227.05 μs |  0.86 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float64 |        cpu |     11,905.91 μs |        780.422 μs |      42.778 μs |     11,893.28 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float64 |        cpu |      7,114.09 μs |      9,419.975 μs |     516.341 μs |      7,340.72 μs |  0.60 |    0.04 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float64 |        cpu |      7,067.62 μs |      2,101.395 μs |     115.185 μs |      7,120.07 μs |  0.59 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float64 |        cpu |      7,301.91 μs |      1,897.127 μs |     103.988 μs |      7,360.99 μs |  0.61 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float64 |        cpu |      5,912.74 μs |        189.834 μs |      10.405 μs |      5,909.90 μs |  0.50 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float64 |        cpu |     14,514.25 μs |      6,323.886 μs |     346.634 μs |     14,376.49 μs |  1.22 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float64 |        cpu |      9,771.81 μs |        748.491 μs |      41.027 μs |      9,768.25 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float64 |        cpu |      6,248.12 μs |        796.139 μs |      43.639 μs |      6,250.60 μs |  0.64 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float64 |        cpu |      6,384.58 μs |      1,036.235 μs |      56.800 μs |      6,399.18 μs |  0.65 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float64 |        cpu |     12,739.18 μs |      5,415.249 μs |     296.828 μs |     12,586.55 μs |  1.30 |    0.03 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float64 |        cpu |     10,215.15 μs |      3,948.780 μs |     216.446 μs |     10,104.29 μs |  1.05 |    0.03 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float64 |        cpu |     21,608.24 μs |     12,689.216 μs |     695.539 μs |     21,245.13 μs |  2.21 |    0.08 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float64 |        cpu |     10,964.54 μs |        524.399 μs |      28.744 μs |     10,957.02 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float64 |        cpu |      2,556.15 μs |        858.311 μs |      47.047 μs |      2,571.54 μs |  0.23 |    0.00 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float64 |        cpu |      6,778.48 μs |      2,457.526 μs |     134.705 μs |      6,745.95 μs |  0.62 |    0.01 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float64 |        cpu |     11,268.90 μs |      1,097.462 μs |      60.156 μs |     11,266.78 μs |  1.03 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float64 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float64 |        cpu |      3,578.21 μs |        687.910 μs |      37.707 μs |      3,582.41 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float64 |        cpu |      3,597.32 μs |      1,721.452 μs |      94.359 μs |      3,604.71 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float64 |        cpu |      3,437.48 μs |      1,773.720 μs |      97.224 μs |      3,418.91 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float64 |        cpu |  1,051,225.10 μs |    226,530.464 μs |  12,416.899 μs |  1,051,615.30 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float64 |        cpu |  1,058,214.10 μs |    332,676.564 μs |  18,235.125 μs |  1,055,761.00 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float64** |       **cuda** |     **38,916.84 μs** |      **1,741.939 μs** |      **95.482 μs** |     **38,953.97 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float64 |       cuda |         54.10 μs |         53.904 μs |       2.955 μs |         53.20 μs | 0.001 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float64 |       cuda |      4,805.60 μs |      3,587.352 μs |     196.635 μs |      4,846.30 μs | 0.123 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float64 |       cuda |      4,772.57 μs |      2,506.462 μs |     137.388 μs |      4,703.30 μs | 0.123 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float64 |       cuda |      2,606.23 μs |        506.252 μs |      27.749 μs |      2,618.46 μs | 0.067 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float64 |       cuda |      2,672.64 μs |        246.571 μs |      13.515 μs |      2,680.11 μs | 0.069 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float64 |       cuda |      1,960.10 μs |        241.848 μs |      13.257 μs |      1,959.53 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float64 |       cuda |        679.43 μs |        256.286 μs |      14.048 μs |        672.50 μs |  0.35 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float64 |       cuda |        856.97 μs |      2,651.755 μs |     145.352 μs |        803.40 μs |  0.44 |    0.07 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float64 |       cuda |        827.47 μs |      1,466.113 μs |      80.363 μs |        804.90 μs |  0.42 |    0.04 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float64 |       cuda |      4,825.97 μs |      4,881.092 μs |     267.549 μs |      4,674.21 μs |  2.46 |    0.14 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float64 |       cuda |      5,517.73 μs |      3,550.030 μs |     194.589 μs |      5,520.66 μs |  2.82 |    0.12 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float64 |       cuda |      1,999.15 μs |          8.423 μs |       0.462 μs |      1,999.35 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float64 |       cuda |        759.73 μs |      1,359.727 μs |      74.531 μs |        774.00 μs |  0.38 |    0.04 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float64 |       cuda |      1,008.67 μs |      4,565.672 μs |     250.260 μs |      1,006.10 μs |  0.50 |    0.13 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float64 |       cuda |        847.73 μs |      1,855.883 μs |     101.727 μs |        793.20 μs |  0.42 |    0.05 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float64 |       cuda |      7,474.40 μs |      1,405.018 μs |      77.014 μs |      7,515.81 μs |  3.74 |    0.04 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float64 |       cuda |      7,865.98 μs |      1,597.632 μs |      87.572 μs |      7,858.90 μs |  3.93 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float64 |       cuda |      1,975.66 μs |        208.865 μs |      11.449 μs |      1,979.62 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float64 |       cuda |        691.43 μs |        391.047 μs |      21.435 μs |        703.00 μs |  0.35 |    0.01 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float64 |       cuda |        814.33 μs |      1,341.717 μs |      73.544 μs |        779.70 μs |  0.41 |    0.04 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float64 |       cuda |        792.97 μs |        785.914 μs |      43.079 μs |        774.50 μs |  0.40 |    0.02 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float64 |       cuda |     44,381.86 μs |     17,947.301 μs |     983.752 μs |     43,964.98 μs | 22.47 |    0.62 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float64 |       cuda |     39,057.66 μs |     11,048.441 μs |     605.602 μs |     39,218.08 μs | 19.77 |    0.19 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float64 |       cuda |      5,889.08 μs |        955.386 μs |      52.368 μs |      5,873.86 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float64 |       cuda |        595.10 μs |        153.768 μs |       8.429 μs |        594.30 μs |  0.10 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float64 |       cuda |        698.13 μs |      2,421.552 μs |     132.733 μs |        621.90 μs |  0.12 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float64 |       cuda |        941.90 μs |        133.765 μs |       7.332 μs |        943.50 μs |  0.16 |    0.00 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float64 |       cuda |     10,170.49 μs |        972.347 μs |      53.298 μs |     10,162.54 μs |  1.73 |    0.02 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float64 |       cuda |     11,825.21 μs |     10,217.848 μs |     560.075 μs |     11,940.08 μs |  2.01 |    0.11 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float64 |       cuda |      5,908.46 μs |        329.838 μs |      18.080 μs |      5,903.00 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float64 |       cuda |        779.87 μs |        379.944 μs |      20.826 μs |        783.40 μs |  0.13 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float64 |       cuda |        842.47 μs |        304.765 μs |      16.705 μs |        840.90 μs |  0.14 |    0.00 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float64 |       cuda |      1,640.07 μs |      3,985.571 μs |     218.463 μs |      1,543.30 μs |  0.28 |    0.04 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float64 |       cuda |      5,857.14 μs |      1,416.960 μs |      77.668 μs |      5,851.92 μs |  0.99 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float64 |       cuda |     13,808.92 μs |      4,543.510 μs |     249.045 μs |     13,847.93 μs |  2.34 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float64 |       cuda |      5,807.43 μs |        582.366 μs |      31.921 μs |      5,818.28 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float64 |       cuda |        553.83 μs |        120.220 μs |       6.590 μs |        551.80 μs |  0.10 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float64 |       cuda |        668.70 μs |        446.267 μs |      24.461 μs |        666.70 μs |  0.12 |    0.00 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float64 |       cuda |      2,058.77 μs |        987.915 μs |      54.151 μs |      2,087.60 μs |  0.35 |    0.01 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float64 |       cuda |     10,611.67 μs |     12,695.383 μs |     695.877 μs |     10,222.85 μs |  1.83 |    0.13 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float64 |       cuda |     20,483.41 μs |      1,041.242 μs |      57.074 μs |     20,516.00 μs |  3.53 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float64 |       cuda |      5,836.97 μs |        448.688 μs |      24.594 μs |      5,832.64 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float64 |       cuda |        427.47 μs |      1,406.798 μs |      77.111 μs |        415.40 μs |  0.07 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float64 |       cuda |      1,018.67 μs |        904.477 μs |      49.577 μs |        996.20 μs |  0.17 |    0.01 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float64 |       cuda |     11,216.97 μs |      1,642.124 μs |      90.010 μs |     11,182.54 μs |  1.92 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float64 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float64 |       cuda |        136.17 μs |      1,067.609 μs |      58.519 μs |        106.20 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float64 |       cuda |        103.70 μs |        516.826 μs |      28.329 μs |         88.10 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float64 |       cuda |        146.63 μs |        801.185 μs |      43.916 μs |        132.80 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float64 |       cuda |  1,057,721.50 μs |    802,713.478 μs |  43,999.434 μs |  1,033,360.90 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float64 |       cuda |  1,066,570.07 μs |    435,279.875 μs |  23,859.159 μs |  1,076,403.20 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |     **int32** |        **cpu** |     **24,979.99 μs** |        **557.636 μs** |      **30.566 μs** |     **24,993.72 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |     int32 |        cpu |         45.48 μs |         25.404 μs |       1.392 μs |         45.39 μs | 0.002 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |     int32 |        cpu |      1,705.93 μs |        325.877 μs |      17.862 μs |      1,714.25 μs | 0.068 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |     int32 |        cpu |      1,706.63 μs |        286.121 μs |      15.683 μs |      1,710.24 μs | 0.068 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |     int32 |        cpu |      2,388.67 μs |      1,295.357 μs |      71.003 μs |      2,395.07 μs | 0.096 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |     int32 |        cpu |      2,391.18 μs |        249.151 μs |      13.657 μs |      2,395.79 μs | 0.096 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |     int32 |        cpu |      1,973.78 μs |        472.991 μs |      25.926 μs |      1,973.24 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |     int32 |        cpu |      4,922.27 μs |      3,310.045 μs |     181.435 μs |      4,827.40 μs |  2.49 |    0.06 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |     int32 |        cpu |      5,001.36 μs |      3,349.244 μs |     183.583 μs |      5,047.93 μs |  2.53 |    0.12 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |     int32 |        cpu |      4,917.55 μs |        942.840 μs |      51.680 μs |      4,947.02 μs |  2.49 |    0.04 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |     int32 |        cpu |      4,195.72 μs |     14,453.888 μs |     792.266 μs |      3,762.09 μs |  2.12 |    0.37 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |     int32 |        cpu |      7,317.70 μs |      6,964.180 μs |     381.730 μs |      7,228.60 μs |  3.71 |    0.16 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |      65536 |     int32 |        cpu |      1,996.89 μs |         85.239 μs |       4.672 μs |      1,999.42 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |     int32 |        cpu |      5,122.99 μs |      1,784.109 μs |      97.793 μs |      5,072.46 μs |  2.57 |    0.06 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |     int32 |        cpu |      5,148.26 μs |      1,366.578 μs |      74.907 μs |      5,134.24 μs |  2.58 |    0.04 |       No |
|                ones_Tensor_Torch |         ones |      65536 |     int32 |        cpu |      5,045.81 μs |        627.471 μs |      34.394 μs |      5,027.58 μs |  2.53 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |     int32 |        cpu |      5,673.80 μs |      1,130.531 μs |      61.968 μs |      5,683.26 μs |  2.84 |    0.04 |       No |
|            ones_Tensor_Reference |         ones |      65536 |     int32 |        cpu |      5,722.58 μs |        557.591 μs |      30.563 μs |      5,730.87 μs |  2.87 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |      65536 |     int32 |        cpu |     34,344.88 μs |      3,639.044 μs |     199.468 μs |     34,343.26 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |      65536 |     int32 |        cpu |      6,996.94 μs |         44.684 μs |       2.449 μs |      6,995.59 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |     int32 |        cpu |      6,240.24 μs |      2,318.525 μs |     127.086 μs |      6,238.33 μs |  0.89 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |     int32 |        cpu |      6,308.55 μs |      2,002.396 μs |     109.758 μs |      6,298.59 μs |  0.90 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |      65536 |     int32 |        cpu |      6,680.59 μs |      1,288.462 μs |      70.625 μs |      6,659.26 μs |  0.95 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |     int32 |        cpu |      9,786.12 μs |     10,731.710 μs |     588.241 μs |      9,909.48 μs |  1.40 |    0.08 |       No |
|        addition_Tensor_Reference |     addition |      65536 |     int32 |        cpu |      9,621.78 μs |     12,309.907 μs |     674.748 μs |      9,255.23 μs |  1.38 |    0.10 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |     int32 |        cpu |      7,712.75 μs |        818.920 μs |      44.888 μs |      7,715.30 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |     int32 |        cpu |      6,258.03 μs |      2,218.552 μs |     121.606 μs |      6,299.43 μs |  0.81 |    0.02 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |     int32 |        cpu |      6,276.63 μs |      1,520.540 μs |      83.346 μs |      6,241.60 μs |  0.81 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |     int32 |        cpu |     11,806.26 μs |      5,761.514 μs |     315.808 μs |     11,853.85 μs |  1.53 |    0.04 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |     int32 |        cpu |      5,729.58 μs |      3,182.921 μs |     174.467 μs |      5,661.99 μs |  0.74 |    0.02 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |     int32 |        cpu |    127,195.77 μs |      7,461.806 μs |     409.007 μs |    127,000.60 μs | 16.49 |    0.10 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |     int32 |        cpu |      6,785.13 μs |        566.648 μs |      31.060 μs |      6,772.62 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |     int32 |        cpu |      5,873.45 μs |      5,363.704 μs |     294.003 μs |      5,887.29 μs |  0.87 |    0.04 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |     int32 |        cpu |      5,804.55 μs |        586.972 μs |      32.174 μs |      5,817.86 μs |  0.86 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |     int32 |        cpu |     12,405.80 μs |      2,636.940 μs |     144.540 μs |     12,463.48 μs |  1.83 |    0.02 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |     int32 |        cpu |      9,244.29 μs |      3,811.438 μs |     208.918 μs |      9,205.35 μs |  1.36 |    0.04 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |     int32 |        cpu |    264,550.50 μs |    316,494.581 μs |  17,348.136 μs |    264,794.40 μs | 38.99 |    2.54 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |     int32 |        cpu |      5,765.73 μs |        916.673 μs |      50.246 μs |      5,752.59 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |     int32 |        cpu |      2,034.92 μs |      1,315.806 μs |      72.124 μs |      2,000.73 μs |  0.35 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |     int32 |        cpu |      6,030.84 μs |      2,728.767 μs |     149.573 μs |      5,981.23 μs |  1.05 |    0.03 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |     int32 |        cpu |      9,548.48 μs |      6,125.084 μs |     335.737 μs |      9,710.79 μs |  1.66 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |     int32 |        cpu |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |     int32 |        cpu |     34,296.17 μs |     14,909.784 μs |     817.256 μs |     33,833.31 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |     int32 |        cpu |     34,220.09 μs |      9,933.434 μs |     544.485 μs |     34,047.03 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |     int32 |        cpu |     34,308.81 μs |      8,997.153 μs |     493.164 μs |     34,166.61 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |     int32 |        cpu |    979,293.33 μs |    239,825.316 μs |  13,145.635 μs |    979,423.70 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |     int32 |        cpu |    980,669.30 μs |    354,795.941 μs |  19,447.563 μs |    970,569.60 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |     **int32** |       **cuda** |     **25,646.32 μs** |      **1,643.638 μs** |      **90.093 μs** |     **25,619.65 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |     int32 |       cuda |         55.30 μs |         54.212 μs |       2.972 μs |         56.60 μs | 0.002 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |     int32 |       cuda |      3,029.80 μs |      4,612.169 μs |     252.809 μs |      2,895.80 μs | 0.118 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |     int32 |       cuda |      3,314.37 μs |      5,825.020 μs |     319.289 μs |      3,135.30 μs | 0.129 |    0.01 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |     int32 |       cuda |      2,378.64 μs |      1,354.741 μs |      74.258 μs |      2,349.28 μs | 0.093 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |     int32 |       cuda |      2,377.25 μs |      1,720.540 μs |      94.309 μs |      2,390.76 μs | 0.093 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |     int32 |       cuda |      1,938.62 μs |        192.718 μs |      10.564 μs |      1,941.95 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |     int32 |       cuda |        680.90 μs |        367.343 μs |      20.135 μs |        684.80 μs |  0.35 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |     int32 |       cuda |        711.60 μs |        102.750 μs |       5.632 μs |        710.20 μs |  0.37 |    0.00 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |     int32 |       cuda |        784.90 μs |        847.033 μs |      46.429 μs |        780.20 μs |  0.40 |    0.03 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |     int32 |       cuda |      3,576.74 μs |        770.653 μs |      42.242 μs |      3,562.47 μs |  1.84 |    0.01 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |     int32 |       cuda |      3,553.25 μs |        812.828 μs |      44.554 μs |      3,535.38 μs |  1.83 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     ones_PyTorch |         ones |      65536 |     int32 |       cuda |      1,944.43 μs |        238.682 μs |      13.083 μs |      1,949.73 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |     int32 |       cuda |        789.60 μs |        236.050 μs |      12.939 μs |        783.10 μs |  0.41 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |     int32 |       cuda |        747.17 μs |        984.668 μs |      53.973 μs |        723.00 μs |  0.38 |    0.03 |       No |
|                ones_Tensor_Torch |         ones |      65536 |     int32 |       cuda |        750.97 μs |        888.230 μs |      48.687 μs |        740.90 μs |  0.39 |    0.03 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |     int32 |       cuda |      5,626.52 μs |      2,653.959 μs |     145.472 μs |      5,631.71 μs |  2.89 |    0.07 |       No |
|            ones_Tensor_Reference |         ones |      65536 |     int32 |       cuda |      5,486.23 μs |        507.960 μs |      27.843 μs |      5,482.65 μs |  2.82 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                     rand_PyTorch |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |      65536 |     int32 |       cuda |        757.77 μs |      1,907.086 μs |     104.534 μs |        705.80 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                 addition_PyTorch |     addition |      65536 |     int32 |       cuda |      4,793.65 μs |        929.111 μs |      50.928 μs |      4,783.07 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |     int32 |       cuda |        919.97 μs |     11,159.915 μs |     611.713 μs |        570.30 μs |  0.19 |    0.12 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |     int32 |       cuda |        611.03 μs |        179.136 μs |       9.819 μs |        606.50 μs |  0.13 |    0.00 |       No |
|            addition_Tensor_Torch |     addition |      65536 |     int32 |       cuda |      1,093.07 μs |      3,512.766 μs |     192.547 μs |        982.20 μs |  0.23 |    0.04 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |     int32 |       cuda |      9,403.45 μs |      3,853.916 μs |     211.246 μs |      9,380.81 μs |  1.96 |    0.06 |       No |
|        addition_Tensor_Reference |     addition |      65536 |     int32 |       cuda |      9,208.62 μs |      1,290.449 μs |      70.734 μs |      9,217.51 μs |  1.92 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |     int32 |       cuda |      5,765.20 μs |        677.226 μs |      37.121 μs |      5,755.99 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |     int32 |       cuda |        807.40 μs |      1,463.075 μs |      80.196 μs |        766.50 μs |  0.14 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |     int32 |       cuda |        871.63 μs |         68.513 μs |       3.755 μs |        871.40 μs |  0.15 |    0.00 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |     int32 |       cuda |      3,296.37 μs |      5,548.324 μs |     304.122 μs |      3,219.00 μs |  0.57 |    0.05 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |     int32 |       cuda |      6,082.24 μs |      4,736.194 μs |     259.607 μs |      5,983.40 μs |  1.05 |    0.04 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |     int32 |       cuda |    129,068.47 μs |     40,083.915 μs |   2,197.135 μs |    130,036.80 μs | 22.39 |    0.51 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |     int32 |       cuda |      4,816.29 μs |        268.953 μs |      14.742 μs |      4,821.43 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |     int32 |       cuda |        703.97 μs |      1,741.065 μs |      95.434 μs |        656.80 μs |  0.15 |    0.02 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |     int32 |       cuda |        640.50 μs |        312.980 μs |      17.155 μs |        645.50 μs |  0.13 |    0.00 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |     int32 |       cuda |      2,077.67 μs |      1,962.262 μs |     107.558 μs |      2,084.40 μs |  0.43 |    0.02 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |     int32 |       cuda |     10,006.04 μs |      6,423.403 μs |     352.088 μs |     10,151.53 μs |  2.08 |    0.07 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |     int32 |       cuda |    260,110.13 μs |     12,767.012 μs |     699.803 μs |    259,739.50 μs | 54.01 |    0.19 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |     int32 |       cuda |      4,771.02 μs |        367.794 μs |      20.160 μs |      4,761.66 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |     int32 |       cuda |        347.07 μs |        197.482 μs |      10.825 μs |        343.60 μs |  0.07 |    0.00 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |     int32 |       cuda |        855.60 μs |        265.564 μs |      14.556 μs |        855.10 μs |  0.18 |    0.00 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |     int32 |       cuda |      9,182.15 μs |        528.004 μs |      28.942 μs |      9,196.62 μs |  1.92 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |                  |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |     int32 |       cuda |               NA |                NA |             NA |               NA |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |     int32 |       cuda |        462.03 μs |        179.701 μs |       9.850 μs |        462.00 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |     int32 |       cuda |        493.00 μs |        175.149 μs |       9.601 μs |        489.30 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |     int32 |       cuda |    977,640.00 μs |    119,208.768 μs |   6,534.235 μs |    978,174.80 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |     int32 |       cuda |    958,407.57 μs |     86,433.230 μs |   4,737.697 μs |    960,383.10 μs |     ? |       ? |       No |

Benchmarks with issues:
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_TorchSharp: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_TorchSharp: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addInPlace_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_TorchSharp: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
