//
//  TextEncoder.swift
//
//
//  Created by Runkai Zhang on 7/26/23.
//

import Foundation
import CoreML

///  A model for encoding text
public struct TextEncoder {

    /// Text tokenizer
    var tokenizer: BPETokenizer

    /// Embedding model
    var model: MLModel
    
    init(resourcesAt baseURL: URL,
         configuration config: MLModelConfiguration = .init()
    ) throws {
        let textEncoderURL = baseURL
        let vocabURL = Bundle.module.url(forResource: "vocab", withExtension: "json")
        let mergesURL = Bundle.module.url(forResource: "merges", withExtension: "txt")
        
        // Text tokenizer and encoder
        let tokenizer = try BPETokenizer(mergesAt: mergesURL!, vocabularyAt: vocabURL!)
        let textEncoderModel = try MLModel(contentsOf: textEncoderURL, configuration: config)
        
        self.tokenizer = tokenizer
        self.model = textEncoderModel
    }
    
    /**
    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - model: Model for encoding tokenized text
    public init(tokenizer: BPETokenizer, model: MLModel) {
        self.tokenizer = tokenizer
        self.model = model
    }
     */

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ text: String) throws -> MLShapedArray<Float32> {

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize, padding to the expected length
        var (tokens, ids) = tokenizer.tokenize(input: text, minCount: inputLength)

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        return try encode(ids: ids)
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "textencoder.predict")

    func encode(ids: [Int]) throws -> MLShapedArray<Float32> {
        let inputName = inputDescription.name
        let inputShape = inputShape

        let floatIds = ids.map { Float32($0) }
        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray)])

        let result = try queue.sync { try model.prediction(from: inputFeatures) }
        let embeddingFeature = result.featureValue(for: "embOutput")
        return MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)
    }

    var inputDescription: MLFeatureDescription {
        model.modelDescription.inputDescriptionsByName.first!.value
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
