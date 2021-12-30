## Hand Gesture Recognition Through ArUco Marker Video Detection

The simple code to this hand gesture recognition program.

Here comes the file tree:

```
src - aruco_detection.py
 \  - model - mlp_net.py
```
where aruco_detection.py works for aruco detection and mlp_net.py defines the mlp net

The `videos/` file contains testing videos and `weights/` file contains pretrained weights.

### Requirments:

opencv-contrib-python
pytorch

### Usage:

running `python3 main.py $video_filename $weight_filename` under this direction, while video_filename is the video to be classified and weight_filename is the pretrained weights.