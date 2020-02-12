# Gap-Crossing Video Analysis

This project presents a system comprised of two machine learning models to identify locations of gap-crossing events. The project them employs computer vision techniques to identify the center of mass and the direction of the fly during these gap-crossing events. The system achieves an optimal accuracy of 84% on identifying frames with attempts and is used to analyze the decision making processes of flies across various genotypes. The goal of this project is to apply this classification system to existing videos to gain meaningful analysis of different genotypes that affect decision-making processes in the fruit fly.


## Getting Started

This project uses Python 3.6. Install the packages in the requirements.txt file using

```
$pip install -r requirements.txt
```

### Usage

Standard usage for running classification model on video file and extracting analysis from locations that are predicted to be gap-crossing events.
```
$python analysis.py [-h] [--saveimgs] [--display] [--savevid] [--framerate FRAMERATE] filename
```

To see the specific usage of each parameter, run:

```
$python analysis.py --help
```

### Example Usage

```
$python analysis.py sample-videos/cross1.avi --saveimgs --framerate 10
```

This command looks for gap-crossing events at 10 frame intervals in the cross1.avi file and saves the output analysis images at events. These saved images are the augmented optical flow image with the ellipse over the fly, the cumulative sum plot, and the 2D histogram of the magnitudes of the optical flow vectors across time.

```
$python analysis.py sample-videos/dir1.avi --display --savevid
```

This command looks for gap-crossing events in the dir1.avi file and displays the output analysis images using Matplotlib (note that displaying the images takes time). This will also save an augmented video cropped around the gap-crossing event with the predicted center of mass overlayed.


## Author

[**Vinay Bhaip**](https://github.com/vbhaip)

Mentors: Hannah Haberkern, Daniel Turner-Evans

This project was completed under the [Jayaraman Lab](https://janelia.org/lab/jayaraman-lab).
