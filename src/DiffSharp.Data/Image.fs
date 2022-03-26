// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

open SixLabors.ImageSharp
open SixLabors.ImageSharp.Processing
open System
open System.IO


/// Contains auto-opened utilities related to the DiffSharp programming model.
[<AutoOpen>]
module ImageUtil =
    /// Saves the given pixel array to a file and optionally resizes it in the process. Resizing uses bicubic interpolation. Supports .png and .jpg formats.
    let saveImage (pixels:float32[,,]) (fileName:string) (resize:option<int*int>) =
        let c, h, w = pixels.GetLength(0), pixels.GetLength(1), pixels.GetLength(2)
        let image = new Image<PixelFormats.RgbaVector>(w, h)
        for y=0 to h-1 do
            for x=0 to w-1 do
                let r, g, b = 
                    if c = 1 then
                        let v = float32(pixels[0, y, x])
                        v, v, v
                    else
                        float32(pixels[0, y, x]), float32(pixels[1, y, x]), float32(pixels[2, y, x])
                image.Item(x, y) <- PixelFormats.RgbaVector(r, g, b)
        let fs = new FileStream(fileName, FileMode.Create)
        let encoder =
            if fileName.EndsWith(".jpg") then
                Formats.Jpeg.JpegEncoder() :> Formats.IImageEncoder
            elif fileName.EndsWith(".png") then
                Formats.Png.PngEncoder() :> Formats.IImageEncoder
            else
                failwithf "Expecting fileName (%A) to end with .png or .jpg" fileName
        match resize with
            | Some(width, height) ->
                if width < 0 || height < 0 then failwithf "Expecting width (%A) and height (%A) >= 0" width height
                image.Mutate(Action<IImageProcessingContext>(fun x -> x.Resize(width, height) |> ignore))
            | None -> ()
        image.Save(fs, encoder)
        fs.Close()

    /// Loads a pixel array from a file and optionally resizes it in the process. Resizing uses bicubic interpolation.
    let loadImage (fileName:string) (resize:option<int*int>) =
        let image:Image<PixelFormats.RgbaVector> = Image.Load(fileName)
        match resize with
            | Some(width, height) ->
                if width < 0 || height < 0 then failwithf "Expecting width (%A) and height (%A) >= 0" width height
                image.Mutate(Action<IImageProcessingContext>(fun x -> x.Resize(width, height) |> ignore))
            | None -> ()
        let pixels = Array3D.init 3 image.Height image.Width (fun c y x -> let p = image.Item(x, y)
                                                                           if c = 0 then p.R
                                                                           elif c = 1 then p.G
                                                                           else p.B)
        pixels


[<AutoOpen>]
module ImageExtensions =
    type Tensor with
        /// <summary>Save tensor to an image file using png or jpg format</summary>
        member t.saveImage(fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?resize:int*int, ?gridCols:int) =
            let pixels:Tensor = t.move(Device.CPU).toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)
            saveImage (pixels.float32().toArray() :?> float32[,,]) fileName resize

        /// <summary>Load an image file and return it as a tensor</summary>
        static member loadImage(fileName:string, ?normalize:bool, ?resize:int*int, ?device: Device, ?dtype: Dtype, ?backend: Backend) =
            let normalize = defaultArg normalize false
            let pixels = loadImage fileName resize
            let pixels:Tensor = Tensor.create(pixels, ?device=device, ?dtype=dtype, ?backend=backend)
            if normalize then pixels.normalize() else pixels


    type dsharp with
        /// <summary>Load an image file as a tensor.</summary>
        /// <param name="fileName">The file name of the image to load.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1).</param>
        /// <param name="resize">An optional new size for the image.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member loadImage(fileName:string, ?normalize:bool, ?resize:int*int, ?device: Device, ?dtype: Dtype, ?backend: Backend) =
            Tensor.loadImage(fileName=fileName, ?normalize=normalize, ?resize=resize, ?device=device, ?dtype=dtype, ?backend=backend)

        /// <summary>Save a given Tensor into an image file.</summary>
        /// <remarks>If the input tensor has 4 dimensions, then make a single image grid.</remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="fileName">The name of the file to save to.</param>
        /// <param name="pixelMin">The minimum pixel value.</param>
        /// <param name="pixelMax">The maximum pixel value.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
        /// <param name="resize">An optional new size for the image.</param>
        /// <param name="gridCols">Number of columns of images in the grid.</param>
        static member saveImage(input:Tensor, fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?resize:int*int, ?gridCols:int) =
            input.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?resize=resize, ?gridCols=gridCols)
