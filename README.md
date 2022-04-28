# Echo_Timing_ONNX
Conversion of trained Echo Timing tf model to ONNX file

Open Neural Network Exchange (ONNX) Runtime is an open source project that is designed to accelerate machine learning across a wide range of frameworks, operating systems, and hardware platforms. It enables acceleration of machine learning inferencing across all of your deployment targets using a single set of API. ONNX Runtime automatically parses through your model to identify optimisation opportunities and provides access to the best hardware acceleration available.

This repository is related to my echocasrdiographic phase detection project. For code [visit](https://intsav.github.io/phase_detection.html) and [here](https://github.com/intsav/EchoPhaseDetection) for a detailed description, including access to the dataset and published paper.

To access the model weights in ONNX format (converted from tf) click [here](https://drive.google.com/file/d/18pOJLL7bzt1_fKhlYOhBBzjbqgnPJVY-/view?usp=sharing).

### File descriptions:

`processing.py` - Data management class used during inference\
`onnx_inference.py` - Inference script to predict ED and ES frames from echocardiographic images of arbitrary length (using ONNX file)

For a simple tf to ONNX model tutorial for image classification, and to assist in understanding this code, visit [this](https://github.com/intsav/ONNX_Conversion) repository.
