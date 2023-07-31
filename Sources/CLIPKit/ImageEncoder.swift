//
//  ImageEncoder.swift
//
//
//  Created by Runkai Zhang on 7/25/23.
//

import CoreML
import ImageIO

public struct ImageEncoder {
    var model: MLModel
    
    init(resourcesAt baseURL: URL,
         configuration config: MLModelConfiguration = .init()
    ) throws {
        let imgEncoderModel = try MLModel(contentsOf: baseURL, configuration: config)
        self.model = imgEncoderModel
    }
    
    /// Prediction queue
    let queue = DispatchQueue(label: "imgencoder.predict")
    
    public func encode(image: CGImage, desiredSize: CGSize) async throws -> MLShapedArray<Float32> {
        do {
            guard let resizedImage = resizeCGImage(image, to: desiredSize) else {
                throw ImageEncodingError.resizeError
            }
            
            guard let buffer = convertCGImageToBuffer(resizedImage) else {
                throw ImageEncodingError.bufferConversionError
            }
            
            guard let inputFeatures = try? MLDictionaryFeatureProvider(dictionary: ["colorImage": buffer]) else {
                throw ImageEncodingError.featureProviderError
            }
            
            let result = try queue.sync { try model.prediction(from: inputFeatures) }
            guard let embeddingFeature = result.featureValue(for: "embOutput"),
                  let multiArray = embeddingFeature.multiArrayValue else {
                throw ImageEncodingError.predictionError
            }
            
            return MLShapedArray<Float32>(converting: multiArray)
        } catch {
            print("Error in encoding: \(error)")
            throw error
        }
    }
    
    private func convertCGImageToBuffer(_ image: CGImage) -> CVPixelBuffer? {
        // Define the image width, height, and color space
        let width = image.width
        let height = image.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        // Create a dictionary with bitmap format options
        let bitmapInfo: UInt32 = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        // Create a data buffer to hold the image pixel data
        guard let data = malloc(width * height * bytesPerPixel) else {
            return nil
        }
        
        // Create a bitmap context to draw the CGImage
        guard let context = CGContext(data: data,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else {
            free(data)
            return nil
        }
        
        // Clear the context and draw the CGImage into it
        context.clear(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        context.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        
        // Create a CVPixelBuffer from the data buffer
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(nil,
                                                  width,
                                                  height,
                                                  kCVPixelFormatType_32BGRA,
                                                  data,
                                                  bytesPerRow,
                                                  nil,
                                                  nil,
                                                  nil,
                                                  &pixelBuffer)
        if status != kCVReturnSuccess {
            free(data)
            return nil
        }
        
        return pixelBuffer
    }
    
    private func resizeCGImage(_ image: CGImage, to newSize: CGSize) -> CGImage? {
        // Calculate the input image aspect ratio
        let inputAspectRatio = CGFloat(image.width) / CGFloat(image.height)
        
        // Calculate the target aspect ratio
        let targetAspectRatio = newSize.width / newSize.height
        
        var scaledWidth: CGFloat
        var scaledHeight: CGFloat
        
        if inputAspectRatio > targetAspectRatio {
            // The input image is wider than the target aspect ratio, so we fix the width and adjust the height
            scaledWidth = newSize.width
            scaledHeight = scaledWidth / inputAspectRatio
        } else {
            // The input image is taller or has the same aspect ratio as the target, so we fix the height and adjust the width
            scaledHeight = newSize.height
            scaledWidth = scaledHeight * inputAspectRatio
        }
        
        // Create a bitmap context with the new size
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(data: nil,
                                      width: Int(scaledWidth),
                                      height: Int(scaledHeight),
                                      bitsPerComponent: image.bitsPerComponent,
                                      bytesPerRow: 0,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else {
            return nil
        }
        
        // Set the interpolation quality to high for better resizing results
        context.interpolationQuality = .high
        
        // Calculate the origin position for drawing the image to the context to center it
        let originX = (scaledWidth - CGFloat(image.width)) * 0.5
        let originY = (scaledHeight - CGFloat(image.height)) * 0.5
        
        // Draw the original image into the context with the new size
        let drawRect = CGRect(x: originX, y: originY, width: CGFloat(image.width), height: CGFloat(image.height))
        context.draw(image, in: drawRect)
        
        // Retrieve the resized image from the context
        let resizedImage = context.makeImage()
        
        return resizedImage
    }

}

enum ImageEncodingError: Error {
    case resizeError
    case bufferConversionError
    case featureProviderError
    case predictionError
}
