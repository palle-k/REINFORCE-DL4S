//
//  main.swift
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

import DL4S
import Foundation

let epochs = 30_000
let iterations = 100
let width = 10
let height = 10
let obstacleCount = 10

let model = PolicyModel(width: width, height: height)
var optimizer = Adam(model: model, learningRate: 0.0003)

var finishCount = 0
var completeCount = 0
var lossSum: Float = 0
var avgPathLen: Float = 0

for epoch in 1 ... epochs {
    
    let obstacles = (0 ..< obstacleCount).map { _ in
        (Int.random(in: 0 ..< width), Int.random(in: 0 ..< height))
    }

    let world = World(width: width, height: height, exit: (0, 0), obstacles: obstacles)
    
    let initialState = State(world: world, agentPosition: (Int.random(in: 0 ..< world.width), Int.random(in: 0 ..< world.height)))
    let (stateSequence, loss) = run(model: optimizer.model, from: initialState, decay: 0.9, maxSteps: 100, explore: 0.1)
    
    let grads = loss.gradients(of: optimizer.model.parameters)
    optimizer.update(along: grads)
    
    lossSum += loss.item / 100
    avgPathLen += Float(stateSequence.count) / 100
    
    if stateSequence.last?.isCompleted ?? false {
        finishCount += 1
    }
    
    if epoch.isMultiple(of: 100) {
        if finishCount == 100 {
            completeCount += 1
        } else if finishCount < 95 {
            completeCount -= 1
            completeCount = max(0, completeCount)
        }
        
        if completeCount == 100 {
            // Finish after 100% accuracy has been achieved 100 times
            break
        }

        print("[\(epoch)/\(epochs)] loss: \(lossSum), completed: \(finishCount)%, avg pathlen: \(avgPathLen)")
        // print("==================================")
        finishCount = 0
        lossSum = 0
        avgPathLen = 0
    }
}

while true {
    let obstacles = (0 ..< obstacleCount).map { _ in
        (Int.random(in: 0 ..< width), Int.random(in: 0 ..< height))
    }

    let world = World(width: width, height: height, exit: (0, 0), obstacles: obstacles)
    
    let initialState = State(world: world, agentPosition: (Int.random(in: 0 ..< world.width), Int.random(in: 0 ..< world.height)))
    let (stateSequence, _) = run(model: optimizer.model, from: initialState, decay: 0.9, maxSteps: 100, explore: 0.0)
    
    printPath(stateSequence, delay: 0.5)
    sleep(2)
}
