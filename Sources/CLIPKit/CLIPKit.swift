import Foundation
import SwiftAnnoy

public class CLIPKit {
    public var imageEncoder: ImageEncoder?
    public var textEncoder: TextEncoder?

    public init() {}
    
    public func performNearestNeighbor(subject: [Float], targets: [[Float]]) {
        let index = AnnoyIndex<Float>(itemLength: 512, metric: .euclidean)

        var items: [[Float]] = [subject]
        items.append(contentsOf: targets)
        
        print(items.count)
        
        // add multple items
        try? index.addItems(items: &items)
        
        try? index.build(numTrees: 1)
        
        // by item
        let results = index.getNNsForItem(item: 0, neighbors: 3)
        print(results as Any)
    }
    
    public func loadImageEncoder(path: String) async -> Bool {
        let resourceURL = URL(fileURLWithPath: path)
    
        do {
            let startingTime = Date()
            let imageEncoder = try ImageEncoder(resourcesAt: resourceURL)
            // 8.542439937591553 seconds used for loading image encoder
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for loading image encoder")
            self.imageEncoder = imageEncoder
            return true
        } catch let error {
            print("Failed to load model: \(error.localizedDescription)")
        }
        
        return false
    }
    
    public func loadTextEncoder(path: String) async -> Bool {
        let resourceURL = URL(fileURLWithPath: path)
    
        do {
            let startingTime = Date()
            let textEncoder = try TextEncoder(resourcesAt: resourceURL)
            // 8.542439937591553 seconds used for loading text encoder
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for loading text encoder")
            self.textEncoder = textEncoder
            return true
        } catch let error {
            print("Failed to load model: \(error.localizedDescription)")
        }
        
        return false
    }
}
