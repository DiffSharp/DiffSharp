// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open DiffSharp

module Dtypes =

    // We run most tests at all these tensor types
    let Bool = [ Dtype.Bool ]
    let SignedIntegral = [ Dtype.Int8; Dtype.Int16; Dtype.Int32; Dtype.Int64 ]
    let UnsignedIntegral = [ Dtype.Byte ]
    let Integral = SignedIntegral @ UnsignedIntegral
    let Float16s = [ Dtype.Float16; Dtype.BFloat16 ]
    let FloatingPointExcept16s = [ Dtype.Float32; Dtype.Float64 ]
    let FloatingPoint = Float16s @ FloatingPointExcept16s
    let Float32 = [ Dtype.Float32 ]

    // Some operations have quirky behaviour on bool types, we pin these down manually
    let SignedIntegralAndFloatingPointExcept16s = FloatingPointExcept16s @ SignedIntegral
    let SignedIntegralAndFloatingPoint = FloatingPoint @ SignedIntegral
    let IntegralAndFloatingPointExcept16s = FloatingPointExcept16s @ Integral
    let IntegralAndFloatingPoint = FloatingPoint @ Integral
    let IntegralAndBool = Integral @ Bool
    let AllExceptFloat16s = FloatingPointExcept16s @ Integral @ Bool
    let All = FloatingPoint @ Integral @ Bool

module Combos =

    // Use these to experiment in your local branch
    //let backends = [ Backend.Reference ]
    //let backends = [ Backend.Torch ]
    //let backends = [ Backend.Reference; Backend.Torch; Backend.Register("TestDuplicate") ]
    //let backends = [ Backend.Reference; Backend.Torch ]
    //let backends = [ Backend.Reference; Backend.Register("TestDuplicate") ]
    //let backends = [ Backend.Register("TestDuplicate") ]
    //let getDevices _ = [ Device.CPU ]
    //let getDevices _ = [ Device.GPU ]
    
    //Use this in committed code
    let backends = [ Backend.Reference; Backend.Torch ]
    let getDevices (deviceType: DeviceType option, backend: Backend option) =
        dsharp.devices(?deviceType=deviceType, ?backend=backend)

    let makeCombos dtypes =
        [ for backend in backends do
            let ds = getDevices (None, Some backend)
            for device in ds do
              for dtype in dtypes do
                yield ComboInfo(defaultBackend=backend, defaultDevice=device, defaultDtype=dtype, defaultFetchDevices=getDevices) ]

    /// These runs though all devices, backends and various Dtype
    let Float32 = makeCombos Dtypes.Float32
    let Integral = makeCombos Dtypes.Integral
    let FloatingPointExcept16s = makeCombos Dtypes.FloatingPointExcept16s
    let FloatingPoint = makeCombos Dtypes.FloatingPoint
    let UnsignedIntegral = makeCombos Dtypes.UnsignedIntegral
    let SignedIntegral = makeCombos Dtypes.SignedIntegral
    let SignedIntegralAndFloatingPointExcept16s = makeCombos Dtypes.SignedIntegralAndFloatingPointExcept16s
    let SignedIntegralAndFloatingPoint = makeCombos Dtypes.SignedIntegralAndFloatingPoint
    let IntegralAndFloatingPointExcept16s = makeCombos Dtypes.IntegralAndFloatingPointExcept16s
    let IntegralAndFloatingPoint = makeCombos Dtypes.IntegralAndFloatingPoint
    let Bool = makeCombos Dtypes.Bool
    let IntegralAndBool = makeCombos Dtypes.IntegralAndBool
    let All = makeCombos Dtypes.All
    let AllExcept16s = makeCombos Dtypes.All

    /// This runs though all devices and backends but leaves the default Dtype
    let AllDevicesAndBackendsFloat32 = 
        [ for backend in backends do
            let ds = getDevices (None, Some backend)
            for device in ds do
              yield ComboInfo(defaultDtype=Dtype.Float32, defaultBackend=backend, defaultDevice=device, defaultFetchDevices=getDevices) ]
