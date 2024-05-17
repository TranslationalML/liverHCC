# DETECT loss
Implementation of the DETECT loss presented in the paper **DiffErentiable TEmporal ContrasT (DETECT) Loss for Liver Cancer Screening in 4D Dynamic Contrast-Enhanced MRI**.

## Abstract
Causing approximately 12M years of life lost worldwide, Hepatocellular Carcinoma (HCC) is an important cause of death which would benefit from improved screening. The Liver Reporting and Data System (LI-RADS) guidelines recommend ultrasound (US) screening for patients at-risk of developping HCC. However, US has a limited sensitivity and abbreviated MRI (AMRI) could be a potential alternative. But HCC screening within AMRI is time-consuming to implement for radiologists, making it an impactful target for automation. In this study, we introduce a novel differentiable temporal contrast (DETECT) loss function based on LI-RADS guidelines to optimize the detection of HCC in 4D dynamic contrast-enhanced (DCE) MRI data  within a screening context. The dataset included 79 patients with diagnosed HCC and 82 healthy patients collected retrospectively between 2014 and 2019. A U-Net model was trained to segment lesions in the liver and was fine-tuned with our loss to detect lesions with temporal malignancy patterns. The U-Net fine-tuned with our custom DETECT loss detected HCC with a higher F1 score and F2 score(F1=0.65+-0.06, F2=0.64+-0.07) compared to the nnU-Net(F1=0.60+-0.08 and F2=0.55+-0.09) in our validation set. Similarly, our custom loss function outperformed the nnU-Net (F1=0.53+-0.03, F2=0.61+-0.03 vs F1=0.29, F2=0.42) with the internal test set. Our results demonstrate that radiological domain expertise can be effectively combined with traditional segmentation loss functions to increase detection performance for screening applications.
## Description
The DETECT loss is designed to improve the detection of HCC in 4D T1WI Dynamic Contrast-Enhanced (DCE) MRI including the 
native, arterial, venous and delayed phases. 
It is composed of a temporal contrastive loss and a segmentation loss.
The segmentation loss is set by default to the Tversky loss. 
The temporal loss is a custom loss function that is designed to capture the temporal dynamics of HCC in DCE-MRI.
The below parameters can be tuned to adjust the weight of each loss and the behavior of the temporal loss.

## Parameters
- `lambda` ($\lambda$): The lambda parameter can be tuned between 0 and 1 to balance the weight of each loss in DETECT loss $L_{DETECT}= \lambda \cdot L_{seg} + (1-\lambda) \cdot L_{temp}$
- `art_factor, del_factor, art_del_factor` ($\gamma, \delta, \epsilon$): Weights for the temporal loss $L_T = \gamma \cdot L_{a} + \delta \cdot L_{d} + \epsilon \cdot L_{a-d}$
- `alpha, beta` ($\alpha, \beta$): Parameters controlling the trade-off between false positives and false negatives in the Tversky loss.
- `min_lesion_size`: Minimum number of voxels of a component in the prediction mask to apply the temporal loss $L_T$.
- `inside_liver_ratio`: Minimum overlap between the lesion and the liver.
- `dilation_kernel`: Size of the kernel used for dilation of lesion masks. Higher values increase the surrounding area of the lesion.
- `k_art_del, k_art, k_del` ($k$): Constants adjusting steepness in the sigmoid functions for temporal loss $L_T$.
- `teta_art_del, teta_art, teta_del` ($\theta$): Constants adjusting the offset of the sigmoid functions for temporal loss $L_T$.
- `cca_num_iterations`: Number of iterations for connected component analysis.
- `threshold`: Threshold applied to the output probability map before applying connected component analysis.
- `art_inside_low_bound, del_inside_low_bound, del_inside_high_bound` : Percentile bounds for determining inside lesion pixel intensities $In^a$ in arterial phase and $In^d$ in delayed phase.
- `outside_low_bound, outside_high_bound`: Percentile bounds for selecting the pixels for the surrounding lesions $Out^a$ in arterial phase and $Out^d$ in delayed phase.
- `min_outside_bound`: That threshold allows to remove remaining extrem small values in the surrounding of a prediction when the prediction is located in a dark region.



## Technologies
- Python 3.8.16
- PyTorch 1.13.1
- MONAI 1.2.0
