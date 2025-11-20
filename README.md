# video-pose-annotation

## How to Run
This depends on `https://github.com/jdibenes/hl2ss`

you must set up your folder structure so that `hl2ss` and this repository are in
the same parent folder. The correct structure should be:

```
parent/
├── hl2ss/
│   └── ...
└── video-pose-annotation/
    └── ...
```

Then you can run
```
python video-annotation.py path/to/data
```

## Keyboard Controls

The following keys are used to adjust the **6D pose** (translation + rotation) of the object, control step size, and manage frame writing.

### Translation
- **q / a** : Increase / decrease translation along **X-axis**
- **w / s** : Increase / decrease translation along **Y-axis**
- **e / d** : Increase / decrease translation along **Z-axis**

### Rotation
- **r / f** : Increase / decrease rotation around **X-axis**
- **t / g** : Increase / decrease rotation around **Y-axis**
- **y / h** : Increase / decrease rotation around **Z-axis**

### Step Size Adjustment
- **o** : Double the translation and rotation step size  
- **l** : Halve the translation and rotation step size  

### Frame Writing Control
- **b** : Decrement `write_index` (next write will overwrite bad frames)  
- **m** : Skip writing this frame (`should_write = False`)  
- **spacebar** : write this frame   



### Exit
- **Esc** : Exit the program immediately
