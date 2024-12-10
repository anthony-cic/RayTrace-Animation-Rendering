# Project 3: Raytracing

Name: Anthony Cicardo   
PSUID: 991643254
Email: abc6181@psu.edu
Date of current submission:

## Submission Description

Snow texture source
https://www.freepik.com/free-photo/homemade-coconut-balls-isolated-white-background_21233471.htm#fromView=keyword&page=1&position=21&uuid=8efef700-9e0a-4551-b121-c481ff81d9f9

## Extra Credit Attempts

Provide a description of any extra credit attempts you've made.

### Installation

Follow the setup instructions of previous projects (setting up a conda environment, etc.). This project requires new packages which can be installed using the following command:

```bash
pip install -r requirements.txt
```

### Arguments

The file `Project3.py` is the entry point for the program. It accepts the following arguments:

- `ray_path`: Path to the scene file. Note in python the `.ray` extension has been replaced with `.json`.
- `width`: Width of the output image.
- `height`: Height of the output image.
- `save_progress`: If True, the program will save the image every Nth iteration where N can be configured.
- `max_depth`: Maximum recursion depth for the raytracing algorithm.

You are free to add/remove any of the arguments as you see fit.

### Quickstart

To run the code with a provided scene file, use the following command:

```bash
python Project3.py --ray_path Media/SampleScenes/reflectionTest.json
```

This will generate an image of size `width x height` and save it to `final/reflectionTest.png`.
