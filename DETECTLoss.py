import torch
import torch.nn.functional as F
from tools import connected_components_with_gradients
from monai.losses import TverskyLoss


class DETECTLoss(torch.nn.modules.loss._Loss):
    """
        DETECTLoss implements a custom loss function combining a Tversky loss and a custom Temporal loss to improve the
        detection of Hepatocellular Carcinoma (HCC) in 4D T1WI Dynamic Contrast Enhanced (DCE) MRI composed of native,
        arterial, venous and delayed phases.

        The Tversky loss is used to compute the loss for the binary segmentation output of the model. The temporal loss
        is used to compute the loss for the temporal contrast of the lesion components identified in the output of the
        model for each patch. Differentiable connected component analysis is used to identify significant lesion
        components in the output of the model for each patch. The loss is then computed for each of these components.
        The mean Temporal loss is computed across all patches in the batch. The final DETECT loss is a weighted sum of
        the Tversky loss and the Temporal loss.

        Connected component analysis is performed using the modified Kornia library function from the ICI loss project.
        Repository URL: https://github.com/BrainImageAnalysis/ICI-loss

        Original paper: DiffErentiable TEmporal ContrasT (DETECT) Loss for Liver Cancer Screening in 4D Dynamic Contrast-Enhanced MRI
        Openreview URL: https://openreview.net/forum?id=yxRRB4QJul&noteId=yxRRB4QJul

        Parameters:
        - art_factor (float): Weighting factor for arterial contrast loss.
        - del_factor (float): Weighting factor for delay phase loss.
        - art_del_factor (float): Weighting factor for arterial-delay loss phase.
        - lambda_factor (float): Balancing factor between Tversky loss and temporal loss.
        - alpha (float), beta (float): Parameters controlling the trade-off between false positives
          and false negatives in the Tversky loss.
        - min_lesion_size (int): Minimum size of a lesion to be considered significant.
        - inside_liver_ratio (float): Minimum ratio of lesion area inside the liver to total lesion area.
        - dilation_kernel (tuple): Size of the kernel used for dilation operations.
        - k_art_del, k_art, k_del (float): Constants adjusting steepness in the sigmoid functions for loss calculation.
        - teta_art_del, teta_art, teta_del (float): Constant parameters adjusting the offset of the sigmoid functions.
        - cca_num_iterations (int): Number of iterations for connected component analysis.
        - cca_threshold (float): Threshold for connected component analysis.
        - art_inside_low_bound, del_inside_low_bound, del_inside_high_bound (float): Quantile bounds
          for determining high/low-intensity pixels within lesions.
        - outside_low_bound, outside_high_bound (float): Quantile bounds for analyzing the intensity
          of pixels surrounding lesions.
        - min_outside_bound (float): Minimum acceptable median intensity for pixels outside lesions.
    """
    def __init__(self,
                 art_factor: float,
                 del_factor: float,
                 art_del_factor: float,
                 lambda_factor: float,
                 alpha: float,
                 beta: float,
                 min_lesion_size: int = 30,
                 inside_liver_ratio: float = 0.5,
                 dilation_kernel: tuple = (11, 11, 11),
                 k_art_del: float = 50,
                 k_art: float = 50,
                 k_del: float = 50,
                 theta_art_del: float = 0.2,
                 theta_art: float = 1.1,
                 theta_del: float = 1.2,
                 cca_num_iterations: int = 75,
                 cca_threshold: float = 0.75,
                 art_inside_low_bound: float = 0.5,
                 del_inside_low_bound: float = 0.1,
                 del_inside_high_bound: float = 0.5,
                 outside_low_bound: float = 0.25,
                 outside_high_bound: float = 0.75,
                 min_outside_bound: float = 0.1
                 ):
        super().__init__()
        self.art_factor = art_factor
        self.del_factor = del_factor
        self.art_del_factor = art_del_factor
        self.k_art_del = k_art_del
        self.k_art = k_art
        self.k_del = k_del
        self.theta_art_del = theta_art_del
        self.theta_art = theta_art
        self.theta_del = theta_del
        self.lambda_factor = lambda_factor
        self.min_lesion_size = min_lesion_size
        self.inside_liver_ratio = inside_liver_ratio
        self.dilation_kernel = dilation_kernel
        self.cca_num_iterations = cca_num_iterations
        self.cca_threshold = cca_threshold
        self.art_inside_low_bound = art_inside_low_bound
        self.del_inside_low_bound = del_inside_low_bound
        self.del_inside_high_bound = del_inside_high_bound
        self.outside_low_bound = outside_low_bound
        self.outside_high_bound = outside_high_bound
        self.minimum_outside_bound = min_outside_bound
        self.tversky_loss_function = TverskyLoss(sigmoid=True, include_background=False, alpha=alpha, beta=beta)


    def compute_loss_per_component(self,
                                   patch: int,
                                   inputs: torch.Tensor,
                                   output_binary_patch: torch.Tensor,
                                   cc3d_mask: torch.Tensor,
                                   counter: dict,
                                   liver_label: torch.Tensor
                                   ):
        """
           Computes the loss for each lesion component identified in the output of the model.

           Parameters:
           - patch (int): Index of the current patch being processed.
           - inputs (Tensor): Input tensor containing the image data for the current batch.
           - output_binary_patch (Tensor): Binary segmentation output for the current patch.
           - cc3d_mask (Tensor): Mask obtained from connected component analysis.
           - counter (dict): Dictionary containing the count of pixels for each labeled component.
           - liver_label (Tensor): Binary mask indicating liver regions.

           Returns:
           - Tensor: Mean loss computed across all significant lesion components in the patch.
        """
        total_losses_per_patch_tensor = torch.tensor(0, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
        lesion_count = 0
        for idx, label in enumerate(counter.keys()):
            lesion_count += 1
            output_cc_binary_ar = torch.where(cc3d_mask == label, output_binary_patch, 0)
            inside_liver_ratio = (liver_label[patch, 0, :, :,
                                  :] * output_cc_binary_ar).sum() / output_cc_binary_ar.sum()
            if not self.min_lesion_size < counter[label] or inside_liver_ratio < self.inside_liver_ratio:
                art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                total_loss = self.art_factor * art_loss + self.del_factor * del_loss + self.art_del_factor * art_del_loss
                total_losses_per_patch_tensor += total_loss
                continue

            art_inside_tumor = output_cc_binary_ar * inputs[patch, 1, :, :, :]
            non_zero_art_val = art_inside_tumor[art_inside_tumor > 0].flatten()
            del_inside_tumor = output_cc_binary_ar * inputs[patch, 3, :, :, :]
            non_zero_del_val = del_inside_tumor[del_inside_tumor > 0].flatten()
            if torch.sum(non_zero_art_val) != 0 and torch.sum(non_zero_del_val) != 0:
                art_quant = torch.quantile(non_zero_art_val, self.art_inside_low_bound)
                art_inten_pixels = torch.where(art_inside_tumor >= art_quant,
                                               torch.tensor(1, dtype=torch.float32, requires_grad=True).to(output_binary_patch.device),
                                               torch.tensor(0, dtype=torch.float32, requires_grad=True).to(output_binary_patch.device))
                del_quant_up = torch.quantile(non_zero_del_val, self.del_inside_high_bound)
                del_quant_low = torch.quantile(non_zero_del_val, self.del_inside_low_bound)
                del_condition = torch.logical_and(del_quant_low < del_inside_tumor,
                                                  del_inside_tumor < del_quant_up)
                del_inten_pixels = torch.where(del_condition,
                                               torch.tensor(1, dtype=torch.float32, requires_grad=True).to(output_binary_patch.device),
                                               torch.tensor(0, dtype=torch.float32, requires_grad=True).to(output_binary_patch.device))
            else:
                art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                total_loss = self.art_factor * art_loss + self.del_factor * del_loss + self.art_del_factor * art_del_loss
                total_losses_per_patch_tensor += total_loss
                continue

            medians_inside = torch.empty(4, dtype=torch.float).to(output_binary_patch.device)
            for i, tag in enumerate(['nat', 'art', 'ven', 'del']):
                inside_tumor = output_cc_binary_ar * inputs[patch, i, :, :, :]
                if tag == 'art':
                    high_intensities = art_inten_pixels * inside_tumor
                elif tag == 'del':
                    high_intensities = del_inten_pixels * inside_tumor
                else:
                    continue
                medians_inside[i] = torch.median(high_intensities[high_intensities > 0])

            padding = ((self.dilation_kernel[0] - 1) // 2, (self.dilation_kernel[1] - 1) // 2, (self.dilation_kernel[2] - 1) // 2)
            dilated_tensor_small = F.max_pool3d(output_cc_binary_ar.view(1, 1, art_inten_pixels.size(0),
                                                                         art_inten_pixels.size(1),
                                                                         art_inten_pixels.size(2)),
                                                kernel_size=self.dilation_kernel, stride=1, padding=padding)
            dilated_label_binary_int = torch.where(dilated_tensor_small > torch.tensor(0.0, dtype=torch.float32,
                                                                                       device=dilated_tensor_small.device,
                                                                                       requires_grad=True),
                                                   torch.tensor(1.0, dtype=torch.float32,
                                                                device=dilated_tensor_small.device, requires_grad=True),
                                                   torch.tensor(0.0, dtype=torch.float32,
                                                                device=dilated_tensor_small.device, requires_grad=True))
            around_tumor = dilated_label_binary_int - output_cc_binary_ar

            around_tumor_inten_pixels = {}
            for i, tag in enumerate(['nat', 'art', 'ven', 'del']):
                if tag == 'art' or tag == 'del':
                    around_tumor_region = around_tumor[0, 0, :, :, :] * inputs[patch, i, :, :, :]
                    non_zero_around = around_tumor_region[around_tumor_region > 0].flatten()

                    if torch.sum(non_zero_around) == 0:
                        continue
                    else:
                        around_quant_high = torch.quantile(non_zero_around, self.outside_high_bound)
                        around_quant_low = torch.quantile(non_zero_around, self.outside_low_bound)
                        condition = torch.logical_and(around_quant_low < around_tumor_region,
                                                      around_tumor_region < around_quant_high)
                        around_tumor_inten_pixels[tag] = torch.where(condition,
                                                                     torch.tensor(1, dtype=torch.float32,
                                                                                  requires_grad=True).to(output_binary_patch.device),
                                                                     torch.tensor(0, dtype=torch.float32,
                                                                                  requires_grad=True).to(output_binary_patch.device))

            if len(around_tumor_inten_pixels.keys()) != 2:
                art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(output_binary_patch.device)
                total_loss = self.art_factor * art_loss + self.del_factor * del_loss + self.art_del_factor * art_del_loss
                total_losses_per_patch_tensor += total_loss
                continue

            contrasts = torch.empty(4, dtype=torch.float).to(output_binary_patch.device)
            medians_outside = torch.empty(4, dtype=torch.float).to(output_binary_patch.device)
            for i, tag in enumerate(['nat', 'art', 'ven', 'del']):
                if tag == 'art' or tag == 'del':
                    min_value = 0
                    outside_tumor = around_tumor_inten_pixels[tag] * inputs[patch, i, :, :, :]
                    number_list = outside_tumor[outside_tumor > min_value].flatten()
                    outside_median = torch.median(number_list)

                    if outside_median < self.minimum_outside_bound:
                        number_list = outside_tumor[outside_tumor > outside_median].flatten()
                        outside_median = torch.median(number_list)

                    medians_outside[i] = outside_median
                    computed_contrast = medians_inside[i] / outside_median

                    if torch.isnan(computed_contrast):
                        contrasts[i] = 0
                    else:
                        contrasts[i] = computed_contrast

            art_del_cont_diff = contrasts[1] - contrasts[3]
            art_del_loss = (-1 / (1 + torch.exp(self.k_art_del * (-art_del_cont_diff + self.theta_art_del)))) + 1
            art_loss = (-1 / (1 + torch.exp(self.k_art * (-contrasts[1] + self.theta_art)))) + 1
            del_loss = 1 / (1 + torch.exp(self.k_del * (-contrasts[3] + self.theta_del)))
            total_loss = self.art_factor * art_loss + self.del_factor * del_loss + self.art_del_factor * art_del_loss
            total_losses_per_patch_tensor += total_loss

        total_losses_per_patch_tensor_mean = total_losses_per_patch_tensor/len(counter.keys())
        return total_losses_per_patch_tensor_mean

    def compute_temporal_loss(self,
                              outputs: torch.Tensor,
                              inputs: torch.Tensor,
                              liver_label: torch.Tensor
                              ):
        """
            Calculates the temporal loss for the entire batch by aggregating losses from individual patches.

            Parameters:
            - outputs (Tensor): Output tensor for the current batch.
            - inputs (Tensor): Input tensor containing the image data for the current batch.
            - liver_label (Tensor): Binary mask indicating liver regions.

            Returns:
            - Tensor: Mean temporal loss computed across all patches in the batch.
        """

        out_binary_map = torch.sigmoid(outputs)
        temporal_loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(outputs.device)
        patch_nbr = 0
        for patch in range(inputs.shape[0]):
            output_binary_patch = out_binary_map[patch, 0, :, :, :]
            cc3d_mask = connected_components_with_gradients(output_binary_patch.unsqueeze(0).unsqueeze(0),
                                                            num_iterations=self.cca_num_iterations,
                                                            threshold=self.cca_threshold)
            cc3d_mask = cc3d_mask.squeeze(0).squeeze(0)

            component_labels, count_per_label = torch.unique(cc3d_mask, sorted=True, return_counts=True)
            component_count = {element.item(): count.item() for element, count in zip(component_labels, count_per_label)
                               if element.item() != 0}
            if 0 < len(component_count.keys()):
                total_losses_per_patient_tensor = self.compute_loss_per_component(patch, inputs, output_binary_patch,
                                                                                  cc3d_mask, component_count,
                                                                                  liver_label)
                temporal_loss += total_losses_per_patient_tensor
                patch_nbr += 1

        if patch_nbr != 0:
            mean_temporal_loss = temporal_loss/patch_nbr
        else:
            mean_temporal_loss = 1

        return mean_temporal_loss

    def forward(self,
                outputs: torch.Tensor,
                inputs: torch.Tensor,
                labels: torch.Tensor,
                liver_label: torch.Tensor
                ):
        """
            Forward method of the DETECTLoss, combining the Tversky loss and the Temporal loss.

            Parameters:
            - outputs (Tensor): Output tensor for the current batch.
            - inputs (Tensor): Input tensor containing the image data for the current batch.
            - labels (Tensor): Ground truth labels for the current batch.
            - liver_label (Tensor): Binary mask indicating liver regions.

            Returns:
            - Tensor: Final DETECT loss for the batch.
        """
        temporal_loss = self.compute_temporal_loss(outputs, inputs, liver_label)
        tversky_loss = self.tversky_loss_function(outputs, labels)
        return (1-self.lambda_factor)*temporal_loss + self.lambda_factor*tversky_loss
