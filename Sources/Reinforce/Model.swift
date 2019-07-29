//
//  Model.swift
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


func PolicyModel(width: Int, height: Int) -> Sequential<Float, CPU> {
    Sequential<Float, CPU>(
        Flatten().asAny(),
        Dense(inputFeatures: width * height * 2, outputFeatures: width * height).asAny(),
        Relu().asAny(),
        Dense(inputFeatures: width * height, outputFeatures: 4).asAny(),
        Softmax().asAny()
    )
}

func encode<E, D>(location: (Int, Int), world: World) -> Tensor<E, D> {
    let (x, y) = location
    //return Tensor([E(x) / E(world.width), E(y) / E(world.height)])
    let t = Tensor<E, D>(repeating: 0, shape: 2, world.height, world.width)
    
    for y in 0 ..< world.height {
        for x in 0 ..< world.width {
            t[0, y, x] = world.tiles[y][x] == .obstacle ? 1 : 0
        }
    }
    t[1, y, x] = 1
    return t
}
