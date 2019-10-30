//
//  World.swift
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


struct World {
    enum Tile: Int, Hashable {
        case empty = 0
        case exit = 1
        case obstacle = 2
        case agent = 3
    }
    
    var tiles: [[Tile]]
}

extension World.Tile: CustomStringConvertible {
    var description: String {
        return [" ", "O", "X", "•"][rawValue]
    }
}

extension World {
    var width: Int {
        tiles.first?.count ?? 0
    }
    
    var height: Int {
        tiles.count
    }
    
    init(width: Int, height: Int, exit: (Int, Int), obstacles: [(Int, Int)]) {
        var tiles = [[Tile]](repeating: [Tile](repeating: .empty, count: width), count: height)
        
        for (x, y) in obstacles {
            tiles[y][x] = .obstacle
        }
        
        tiles[exit.1][exit.0] = .exit
        
        self.init(
            tiles: tiles
        )
    }
    
    init(width: Int, height: Int, exit: (Int, Int), obstacles: (Int, Int)...) {
        self.init(width: width, height: height, exit: exit, obstacles: obstacles)
    }
}

extension World: CustomStringConvertible {
    func generateDescription(agentPosition: (Int, Int)) -> String {
        var copy = tiles
        copy[agentPosition.1][agentPosition.0] = .agent
        return World(tiles: copy).description
    }
    
    var description: String {
        let hBorder = " \(repeatElement("-", count: width).joined()) "
        let rows = tiles.map { row -> String in
            "|\(row.map {$0.description}.joined())|"
        }.joined(separator: "\n")
        
        return """
        \(hBorder)
        \(rows)
        \(hBorder)
        """
    }
}

enum Action: Int, Hashable {
    case left
    case right
    case up
    case down
    
    func apply(to position: (Int, Int)) -> (Int, Int) {
        let (x, y) = position
        
        switch self {
        case .left:
            return (x - 1, y)
        case .right:
            return (x + 1, y)
        case .up:
            return (x, y - 1)
        case .down:
            return (x, y + 1)
        }
    }
}

struct State {
    var world: World
    var steps = 0
    var agentPosition: (Int, Int)
    
    var isCompleted: Bool {
        world.tiles[agentPosition.1][agentPosition.0] == .exit
    }
    
    func reward(for action: Action) -> Float {
        let (x, y) = action.apply(to: agentPosition)

        if !(0 ..< world.width ~= x) || !(0 ..< world.height ~= y) {
            return -10
        }
        switch world.tiles[y][x] {
        case .empty:
            return -1
        case .exit:
            return 100
        case .obstacle:
            return -10
        default:
            return 0
        }
    }
    
    func next(applying action: Action) -> State {
        let (x, y) = action.apply(to: agentPosition)
        let (x2, y2) = (max(min(world.width - 1, x), 0), max(min(world.height - 1, y), 0))
        if world.tiles[y2][x2] == .obstacle {
            return self
        }
        return State(world: world, steps: steps + 1, agentPosition: (x2, y2))
    }
}

extension State: CustomStringConvertible {
    var description: String {
        world.generateDescription(agentPosition: agentPosition)
    }
}
