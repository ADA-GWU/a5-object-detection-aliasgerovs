# Assignment 5. Object detection and document recognition

## Overview
In this task, we have been required to recognize the content of a small text document. A receipt from supermarket or electricity bill cheque should be fine.

### Setup

Prerequisites:
- Python 3.10 or higher
- Libraries: OpenCV, NumPy, Torch, torchvision, EasyOCR, Matplotlib
- A camera or an image file of a skewed document (e.g., receipt or bill)

### Installation:
1. Clone the repository:
```
   git clone https://github.com/ADA-GWU/a5-object-detection-aliasgerovs.git
   cd a5-object-detection-aliasgerovs
```

2. Install the required Python packages:
```
   pip3 install -r requirements.txt
```

### Running the Code

To run the text recognition:

1. Place your skewed image in the project directory or update the image path in the script.
2. Execute the script from the command line:

```
   python3 recognition.py

```
3. Follow the on-screen instructions to select the corners of the document in the displayed window.
4. Press "p" to process the image after selecting the corners.

### Features

- Perspective Transformation: Converts the trapezoidal shape of the document to a rectangular form using manually selected corners.
- Text Detection: Utilizes the MSER algorithm to detect regions likely to contain text.
- Text Recognition: Applies a pre-trained CNN to recognize the detected text.
- Visualization: Shows the processed images and detected text regions with recognized text.

### Code Structure
- text_recognition.py: Contains the main logic for image processing, text detection, and recognition.
- images folder: Contains images.
