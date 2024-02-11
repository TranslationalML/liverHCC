import torch
import torch.nn.functional as F
from tools import connected_components_with_gradients


def computeLossPerComponent(patch, inputs, output_binary_patch, cc3d_mask, counter, liver_label, art_factor, del_factor,
                            art_del_factor, device):
    total_losses_per_patch_tensor = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)
    cnt_lesions = 0
    for idx, label in enumerate(counter.keys()):
        cnt_lesions += 1
        output_cc_binary_ar = torch.where(cc3d_mask == label, output_binary_patch, 0)
        if not 30 < counter[label]:
            art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            total_loss = art_factor * art_loss + del_factor * del_loss + art_del_factor * art_del_loss
            total_losses_per_patch_tensor += total_loss
            continue

        inside_liver_ratio = (liver_label[patch, 0, :, :, :] * output_cc_binary_ar).sum() / output_cc_binary_ar.sum()
        if inside_liver_ratio < 0.5:
            art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            total_loss = art_factor * art_loss + del_factor * del_loss + art_del_factor * art_del_loss
            total_losses_per_patch_tensor += total_loss
            continue

        art_inside_tumor = output_cc_binary_ar * inputs[patch, 1, :, :, :]
        non_zero_art_val = art_inside_tumor[art_inside_tumor > 0].flatten()
        del_inside_tumor = output_cc_binary_ar * inputs[patch, 3, :, :, :]
        non_zero_del_val = del_inside_tumor[del_inside_tumor > 0].flatten()
        if torch.sum(non_zero_art_val) != 0 and torch.sum(non_zero_del_val) != 0:
            art_quant = torch.quantile(non_zero_art_val, 0.5)
            art_inten_pixels = torch.where(art_inside_tumor >= art_quant,
                                           torch.tensor(1, dtype=torch.float32, requires_grad=True).to(device),
                                           torch.tensor(0, dtype=torch.float32, requires_grad=True).to(device))
            del_quant_up = torch.quantile(non_zero_del_val, 0.5)
            del_quant_low = torch.quantile(non_zero_del_val, 0.1)
            del_condition = torch.logical_and(del_quant_low < del_inside_tumor,
                                              del_inside_tumor < del_quant_up)
            del_inten_pixels = torch.where(del_condition,
                                           torch.tensor(1, dtype=torch.float32, requires_grad=True).to(device),
                                           torch.tensor(0, dtype=torch.float32, requires_grad=True).to(device))
        else:
            art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            total_loss = art_factor * art_loss + del_factor * del_loss + art_del_factor * art_del_loss
            total_losses_per_patch_tensor += total_loss
            continue

        medians_inside = torch.empty(4, dtype=torch.float).to(device)
        for i, tag in enumerate(['nat', 'art', 'ven', 'del']):
            inside_tumor = output_cc_binary_ar * inputs[patch, i, :, :, :]
            if tag == 'art':
                high_intensities = art_inten_pixels * inside_tumor
            elif tag == 'del':
                high_intensities = del_inten_pixels * inside_tumor
            else:
                continue
            medians_inside[i] = torch.median(high_intensities[high_intensities > 0])

        large_kernel = (11, 11, 11)
        padding = ((large_kernel[0] - 1) // 2, (large_kernel[1] - 1) // 2, (large_kernel[2] - 1) // 2)
        dilated_tensor_small = F.max_pool3d(output_cc_binary_ar.view(1, 1, art_inten_pixels.size(0),
                                                                     art_inten_pixels.size(1),
                                                                     art_inten_pixels.size(2)),
                                            kernel_size=large_kernel, stride=1, padding=padding)
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
                    around_quant_high = torch.quantile(non_zero_around, 0.75)
                    around_quant_low = torch.quantile(non_zero_around, 0.25)
                    condition = torch.logical_and(around_quant_low < around_tumor_region,
                                                  around_tumor_region < around_quant_high)
                    around_tumor_inten_pixels[tag] = torch.where(condition,
                                                                 torch.tensor(1, dtype=torch.float32,
                                                                              requires_grad=True).to(device),
                                                                 torch.tensor(0, dtype=torch.float32,
                                                                              requires_grad=True).to(device))

        if len(around_tumor_inten_pixels.keys()) != 2:
            art_del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            art_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            del_loss = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
            total_loss = art_factor * art_loss + del_factor * del_loss + art_del_factor * art_del_loss
            total_losses_per_patch_tensor += total_loss
            continue

        contrasts = torch.empty(4, dtype=torch.float).to(device)
        medians_outside = torch.empty(4, dtype=torch.float).to(device)
        for i, tag in enumerate(['nat', 'art', 'ven', 'del']):
            if tag == 'art' or tag == 'del':
                min_value = 0
                outside_tumor = around_tumor_inten_pixels[tag] * inputs[patch, i, :, :, :]
                number_list = outside_tumor[outside_tumor > min_value].flatten()
                outside_median = torch.median(number_list)

                if outside_median < 0.1:
                    number_list = outside_tumor[outside_tumor > outside_median].flatten()
                    outside_median = torch.median(number_list)
                    if outside_median < 0.1:
                        number_list = outside_tumor[outside_tumor > outside_median].flatten()
                        outside_median = torch.median(number_list)
                medians_outside[i] = outside_median
                computed_contrast = medians_inside[i] / outside_median

                if torch.isnan(computed_contrast):
                    contrasts[i] = 0
                else:
                    contrasts[i] = computed_contrast

        art_del_cont_diff = contrasts[1] - contrasts[3]
        art_del_loss = (-1 / (1 + torch.exp(50 * (-art_del_cont_diff + 0.2)))) + 1
        art_loss = (-1 / (1 + torch.exp(50 * (-contrasts[1] + 1.1)))) + 1
        del_loss = 1 / (1 + torch.exp(50 * (-contrasts[3] + 1.2)))
        total_loss = art_factor * art_loss + del_factor * del_loss + art_del_factor * art_del_loss
        total_losses_per_patch_tensor += total_loss

    total_losses_per_patch_tensor_mean = total_losses_per_patch_tensor/len(counter.keys())
    return total_losses_per_patch_tensor_mean


def applyCCAndComputeLoss(outputs, inputs, liver_label, device, art_factor, del_factor, art_del_factor):
    range_values = torch.sigmoid(outputs)
    mean_total_losses_per_batch = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)
    patch_nbr = 0
    for patch in range(inputs.shape[0]):
        output_binary_patch = range_values[patch, 0, :, :, :]
        cc3d_mask = connected_components_with_gradients(output_binary_patch.unsqueeze(0).unsqueeze(0),
                                                        num_iterations=75, threshold=0.75)
        cc3d_mask = cc3d_mask.squeeze(0).squeeze(0)

        labels_count, count = torch.unique(cc3d_mask, sorted=True, return_counts=True)
        counter = {element.item(): count.item() for element, count in zip(labels_count, count) if element.item() != 0}
        if 0 < len(counter.keys()):
            total_losses_per_patient_tensor = computeLossPerComponent(patch, inputs, output_binary_patch,
                                                                       cc3d_mask, counter, liver_label, art_factor,
                                                                       del_factor, art_del_factor, device)
            mean_total_losses_per_batch += total_losses_per_patient_tensor
            patch_nbr += 1

    if patch_nbr != 0:
        total_losses_mean = mean_total_losses_per_batch/patch_nbr
    else:
        total_losses_mean = 1

    return total_losses_mean
