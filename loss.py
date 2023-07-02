import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def SSIM(prediction, ground_truth):
    window_size = 11
    sigma = 1.5
    gaussian = torch.Tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gaussian = torch.unsqueeze(gaussian / gaussian.sum(), dim=0)
    gaussian_t = gaussian.t()
    gaussian_filter = torch.unsqueeze(torch.unsqueeze(torch.matmul(gaussian_t, gaussian), dim=0), dim=0)
    gaussian_filter = gaussian_filter.to(prediction.device)
    mu1 = F.conv2d(prediction, gaussian_filter, padding=window_size // 2)
    mu2 = F.conv2d(ground_truth, gaussian_filter, padding=window_size // 2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(prediction * prediction, gaussian_filter, padding=window_size // 2) - mu1 ** 2
    sigma2_sq = F.conv2d(ground_truth * ground_truth, gaussian_filter, padding=window_size // 2) - mu2 ** 2
    sigma12 = F.conv2d(prediction * ground_truth, gaussian_filter, padding=window_size // 2) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    contrast = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

    n1 = 2 * mu12 + C1
    n2 = 2 * sigma12 + C2
    d1 = mu1 ** 2 + mu2 ** 2 + C1
    d2 = sigma1_sq + sigma2_sq + C2
    ssim = ((n1 * n2) / (d1 * d2)).mean()
    L_ssim = (1 - ssim) / 2
    return L_ssim


def Loss(prediction, ground_truth):
    L1 = nn.L1Loss()
    L1_loss = L1(prediction, ground_truth)
    L_ssim = SSIM(prediction, ground_truth)
    return L1_loss, L_ssim
