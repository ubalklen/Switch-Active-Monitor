# Switch Active Monitor

<p align="center"><img src="gif/demo.gif"/></p>

Proof of concept of automatic active monitor switching in a dual-monitor computer using a webcam. Position of user's nose and shoulders are tracked using [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet). Switching occurs when the nose gets closer to one of the shoulders.

Uses TensorFlow 1.15.3. Works only on Windows.

PoseNet Python porting by [zg9uagfv](https://github.com/zg9uagfv/tf_posenet).

## Usage
1. Install all the dependences.
2. `python posenet.py`