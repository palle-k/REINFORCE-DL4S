//
//  Distributions.swift
//  Reinforce
//
//  Created by Palle Klewitz on 28.07.19.
//  Copyright (c) 2019 Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

import Foundation
import DL4S

protocol Distribution {
    associatedtype Element
    
    func sample() -> Element
}


struct Categorical<E: NumericType, Device: DeviceType>: Distribution {
    typealias Element = Int
    
    var densities: Tensor<E, Device>
    var cumulative: Tensor<E, Device>
    
    init(densities: Tensor<E, Device>) {
        self.densities = densities
        cumulative = Tensor<E, Device>(repeating: 0, shape: densities.shape)
        var sum = Tensor<E, Device>(repeating: 0, shape: Array(densities.shape[1...]))
        for i in 0 ..< densities.shape[0] {
            sum += densities[i].detached()
            cumulative[i] = sum
        }
    }
    
    func sample() -> Int {
        let random = E(Float.random(in: 0 ... 1))
        
        for i in 0 ..< cumulative.shape[0] {
            if random < cumulative[i].item {
                return i
            }
        }
        
        return cumulative.shape[0] - 1
    }
    
    func logProb(_ i: Int) -> Tensor<E, Device> {
        densities[i]
    }
}
