# CVMVT
Complex-valued Multi-scale Vision Transformer on Space Target Recognition by ISAR image sequence
# Abstract
In recent years, researches on the recognition for Inverse Synthetic Aperture Radar (ISAR) images continue to deepen, while most methods only use the real half of the complex-valued (CV) data. In addition, higher-order terms in the received signals for maneuvering space targets will cause defocus on the ISAR images, which affects the recognition task. To this end, this letter proposes an end-to-end recognition framework in the CV domain based on the transformer model, called CVMVT. It uses ISAR image sequences as input, which can obtain richer feature information of the space targets, and uses a CV attention mechanism to capture information between image frames from the input sequences. Besides, the multi-scale feature extraction method is combined with CV-CNN to better extract local features of ISAR images. Finally, a CV Layernorm (LN) layer is innovatively used to forward the defocused higher-order term information by computing the covariance of the real and imaginary parts to facilitate the recognition. In the experimental part, the proposed model is tested with real data and simulated data, and CVMVT obtains more accurate recognition results in comparison with similar methods.   
![img](https://github.com/yhx-hit/cv_gnn/blob/main/heart.gif)
# Dataset
The self-built dataset of defocused ISAR image sequences can be downloaded from  
https://drive.google.com/file/d/1KlgOx7Y3G6EOL9aHswSBdZ-az6W2MloA/view?usp=sharing  
The dataset exists in the .mat format.  
# Read dataset
The ISAR image data is in the form of complex-valued. We can get the content by:  
```
  import h5py
  input_dict = h5py.File(dataset)
  img = input_dict['s3']
  img_real = img['real'].astype(np.float32)
  img_imag = img['imag'].astype(np.float32)
