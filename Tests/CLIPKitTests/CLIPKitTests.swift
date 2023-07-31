import XCTest
@testable import CLIPKit

final class CLIPKitTests: XCTestCase {
    func testExample() async throws {
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods
        
        let kit = CLIPKit()
        await kit.loadImageEncoder(path: "\(Bundle.main.bundlePath)/Models/ImageEncoder_float32.mlmodelc")
    }
}
