# InterPlanetary3D

InterPlanetary3D is a split-screen voxel combat prototype by `matd.space`, built with C++, OpenGL, GLFW, and GLAD.

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
- Player health with hazard and blast damage
- Vertical split-screen local two-player mode

## Build

```bash
cd /Users/user/Documents/InterPlanetary3D
cmake -S . -B build
cmake --build build
```

## Run

```bash
cd /Users/user/Documents/InterPlanetary3D
./build/InterPlanetary3D
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

Player 2 uses a control pad on the right screen:
- `Left stick`: move
- `Right stick`: look
- `A`: jump
- `Left stick press`: sprint
- `B`: drop an atomic bomb from player 2's satellite
- `Left trigger`: place block
- `Right trigger`: dig block
- `Left / right bumper`: rotate player 2's satellite orbit
- `X / Y`: slow down or speed up player 2's satellite
- `D-pad left / up / right`: switch block type

Break glowing cyan fuel blocks to refill the satellite feed. Break toxic-yellow plutonium blocks to stockpile bomb material. Each atomic bomb costs `2` plutonium. When fuel runs out, the satellite camera collapses into static.
Nearby atomic blasts hurt, and touching the forcefield is instantly fatal.
