let Sub_S_V_code = "coef_x - x;"
let Sub_V_S_code = "x - coef_x;"
let Sub_S_M_code = "coef_x - x;"
let Sub_M_S_code = "x - coef_x;"
//Mul_M_M_Add_V_MCols
let Mul_Had_M_M_code = "x * y;"

#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\ManagedCuda.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\NVRTC.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\CudaBlas.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\CudaSolve.dll"

#load "Backend.fs"
open DiffSharp.Backend

#load "Util.fs"
open DiffSharp.Util
