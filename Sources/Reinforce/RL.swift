//
//  RL.swift
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


func step<L: Layer>(state: State, model: L, explore: Double) -> (State, Tensor<L.Element, L.Device>, L.Element) {
    let x: Tensor<L.Input, L.Device> = encode(location: state.agentPosition, world: state.world)
    let y = model.forward(x.view(as: 1, -1)).view(as: -1)
    let distribution = Categorical(densities: y)
    
    let a: Int
    
    if Double.random(in: 0 ... 1) <= explore {
        a = Int.random(in: 0 ..< 3)
    } else {
        a = distribution.sample()
    }
    let action = Action(rawValue: a)!
    
    let reward = state.reward(for: action)
    let newState = state.next(applying: action)
    
    return (newState, distribution.logProb(a), L.Element(reward))
}

func run<L: Layer>(model: L, from initialState: State, decay: L.Element = 0.95, maxSteps: Int = 100, explore: Double = 0.1) -> ([State], Tensor<L.Element, L.Device>) {
    var results: [(State, Tensor<L.Element, L.Device>, L.Element)] = []
    var state = initialState
    
    for _ in 0 ..< maxSteps {
        if state.isCompleted {
            break
        }
        let (n, lp, r) = step(state: state, model: model, explore: explore)
        results.append((n, lp, r))
        state = n
    }
    
    var reward: L.Element = 0
    var totalLoss: Tensor<L.Element, L.Device> = 0
    for (_, lp, r) in results.reversed() {
        reward = r + reward * decay
        totalLoss += -lp * Tensor(reward)
    }
    
    return (results.map {$0.0}, totalLoss / Tensor(L.Element(maxSteps)))
}

