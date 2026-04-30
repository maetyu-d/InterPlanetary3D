# MineyCraft

Small C++ voxel prototype using OpenGL, GLFW, and GLAD.

## Features

- First-person camera with collision, gravity, and jumping
- Multi-chunk voxel terrain with visible-face meshing
- Chunk-local remeshing when blocks change
- Procedural red-black texture atlas
- Block placement and removal
- Targeted block outline for clearer interaction
- Animated infernal sky and heavier red fog for a hellish look
- Collectible fuel blocks that keep the satellite feed alive
- Toxic-yellow plutonium blocks used to arm atomic bombs

## Build

```bash
cd /Users/user/Documents/MineyCraft
cmake -S . -B build
cmake --build build
```

## Run

```bash
cd /Users/user/Documents/MineyCraft
./build/mineycraft
```

## Controls

- `W A S D`: move
- `Space`: jump
- `Left Shift`: sprint
- `- / =`: rotate the satellite's polar orbit west/east
- `[ / ]`: slow down or speed up the satellite
- `P`: drop an atomic bomb from the satellite
- `Hold Left Click`: dig and remove block
- `Hold Right Click`: place block
- `1 / 2 / 3`: switch block type
- `Esc`: release mouse

Break glowing cyan fuel blocks to refill the satellite feed. Break toxic-yellow plutonium blocks to stockpile bomb material. Each atomic bomb costs `2` plutonium. When fuel runs out, the satellite camera collapses into static.
