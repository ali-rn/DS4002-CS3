# UNet Reproduction & Workflow Guide
This markdown provides a directory to this folder and how to reproduce the unet results to guide your implementation of a new CNN. Play around with these scripts to learn how these models work

1. The original model was run using Jupyter Notebook through a virtual Rivanna session. Using a super-computer is essential as training these kinds of models is computationally intense.
2. Use instructions in data_import_process.md to import your dataset
3. Run unet.py to train the model
4. Run unet_test.py to test the model
5. Run analysis.ipynb to compare this model to DeepLabV3+
6. Other files:
* Results.jpg visualize example results of the two base models
* deeplab.py shows the DeepLabV3+ model
* nnUNet_architecture.jpg shows the general structure of a UNet (note: the model in the code uses a two-layer UNet)
