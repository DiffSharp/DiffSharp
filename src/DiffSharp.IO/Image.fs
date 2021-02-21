namespace DiffSharp.IO

open DiffSharp

[<AutoOpen>]
module ImageExtensions =
    type Tensor with
        /// <summary>Save tensor to an image file using png or jpg format</summary>
        member t.saveImage(fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) =
            let pixels:Tensor = t.toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)
            let c, h, w = pixels.shape.[0], pixels.shape.[1], pixels.shape.[2]        
            let image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.RgbaVector>(w, h)
            for y=0 to h-1 do
                for x=0 to w-1 do
                    let r, g, b = 
                        if c = 1 then
                            let v = float32(pixels.[0, y, x])
                            v, v, v
                        else
                            float32(pixels.[0, y, x]), float32(pixels.[1, y, x]), float32(pixels.[2, y, x])
                    image.Item(x, y) <- SixLabors.ImageSharp.PixelFormats.RgbaVector(r, g, b)
            let fs = new System.IO.FileStream(fileName, System.IO.FileMode.Create)
            let encoder =
                if fileName.EndsWith(".jpg") then
                    SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder() :> SixLabors.ImageSharp.Formats.IImageEncoder
                elif fileName.EndsWith(".png") then
                    SixLabors.ImageSharp.Formats.Png.PngEncoder() :> SixLabors.ImageSharp.Formats.IImageEncoder
                else
                    failwithf "Expecting fileName (%A) to end with .png or .jpg" fileName
            image.Save(fs, encoder)
            fs.Close()

        /// <summary>Load an image file and return it as a tensor</summary>
        static member loadImage(fileName:string, ?normalize:bool, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
            let normalize = defaultArg normalize false
            let image:SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.RgbaVector> = SixLabors.ImageSharp.Image.Load(fileName)
            let pixels = Array3D.init 3 image.Height image.Width (fun c y x -> let p = image.Item(x, y)
                                                                               if c = 0 then p.R
                                                                               elif c = 1 then p.G
                                                                               else p.B)
            let mutable pixels:Tensor = Tensor.create(pixels, ?dtype=dtype, ?device=device, ?backend=backend)
            if normalize then pixels <- pixels.normalize()
            pixels

    type dsharp with
        /// <summary>Load an image file as a tensor.</summary>
        /// <param name="fileName">The file name of the image to load.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1).</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member loadImage(fileName:string, ?normalize:bool, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
            Tensor.loadImage(fileName=fileName, ?normalize=normalize, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Save a given Tensor into an image file.</summary>
        /// <remarks>If the input tensor has 4 dimensions, then make a single image grid.</remarks>
        /// <param name="input">The input tensor.</param>
        /// <param name="fileName">The name of the file to save to.</param>
        /// <param name="pixelMin">The minimum pixel value.</param>
        /// <param name="pixelMax">The maximum pixel value.</param>
        /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
        /// <param name="gridCols">Number of columns of images in the grid.</param>
        static member saveImage(input:Tensor, fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) =
            input.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)