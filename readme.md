Orthophoto Processing Pipeline

Overview
This Python-based application creates orthophotos by processing webcam imagery, applying geometric corrections, and generating uniformly scaled images suitable for precise measurements and mapping.

Features
Camera Calibration: Precise camera parameter estimation
Lens Distortion Correction: Removes optical aberrations
Orthographic Image Generation: Creates geometrically corrected imagery
Real-time Processing: Processes frames from webcam input
Detailed Logging: Comprehensive terminal output for each processing step

Prerequisites

Hardware
Webcam
Calibration checkerboard (9x6 internal corners recommended)

Software
Python 3.7+
OpenCV
NumPy
SciPy
Matplotlib

Installation

Clone the repository:
git clone https://github.com/Neel-XV/orthophoto-processor.git
cd orthophoto-processor

Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:
pip install -r requirements.txt

Usage
Camera Calibration
Print a checkerboard pattern (9x6 internal corners)
Run the script
Show the checkerboard to the camera during calibration
Follow on-screen instructions

Running the Script
python orthophoto_processor.py
Key Controls
c: Capture calibration frame
q: Quit the application

Workflow Stages
Camera Calibration
Detect checkerboard pattern
Compute camera matrix
Determine lens distortion parameters
Image Undistortion
Remove lens distortions
Create geometrically corrected base image
Orthophoto Generation
Apply geometric transformations
Interpolate image to correct perspective
Optional terrain model support

Limitations
Requires good lighting conditions
Accuracy depends on calibration quality
Limited terrain correction without detailed Digital Elevation Model (DEM)

Troubleshooting
Ensure even, well-lit environment for calibration
Use high-contrast checkerboard
Minimal camera movement during calibration
Check webcam compatibility

License
Distributed under the MIT License. See LICENSE for more information.