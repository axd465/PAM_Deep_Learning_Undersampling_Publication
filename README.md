# PAM_Deep_Learning_Undersampling_Publication
 This project contains the continued work to perfect a model for upsampling undersampled Photoacoustic Microscopy (PAM) images.
 
 **Abstract**—One primary technical challenge in photoacoustic microscopy (PAM) is the necessary compromise between spatial resolution and imaging speed. In this study, we propose a novel application of deep learning principles to reconstruct undersampled PAM images and transcend the trade-off between spatial resolution and imaging speed. We compared various convolutional neural network (CNN) architectures, and selected a fully dense U-net (FD U-net) model that produced the best results. To mimic various undersampling conditions in practice, we artificially downsampled fully-sampled PAM images of mouse brain vasculature at different ratios. This allowed us to not only definitively establish the ground truth, but also train and test our deep learning model at various imaging conditions. Our results and numerical analysis have collectively demonstrated the robust performance of our model to reconstruct PAM images with as few as 2% of the original pixels, which may effectively shorten the imaging time without substantially sacrificing the image quality.
 
**Anaconda Environment**

* To download and duplicate the Anaconda environment used in this project, go to the Anaconda terminal and use the following commands:

```
conda env create axd465/tf
source activate tf
```
