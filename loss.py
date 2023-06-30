import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gradient_x(img):
    gx = img[:, :, :-1] - img[:, :, 1:]
    return gx


def gradient_y(img):
    gy = img[:, :-1, :] - img[:, 1:, :]
    return gy


def get_image_gradient(image):
    return gradient_x(image), gradient_y(image)


def depth_loss(prediction, ground_truth):
    L1_loss = nn.L1Loss()
    L_depth = L1_loss(prediction, ground_truth)
    return L_depth


def grad_loss(prediction, ground_truth):
    L1_loss = nn.MSELoss()
    dx_prediction, dy_prediction = get_image_gradient(prediction)
    dx_ground_truth, dy_ground_truth = get_image_gradient(ground_truth)
    L_grad_x = L1_loss(dx_prediction, dx_ground_truth)
    L_grad_y = L1_loss(dy_prediction, dy_ground_truth)
    L_grad = L_grad_x + L_grad_y
    return L_grad


def TV_loss(prediction):
    dx, dy = get_image_gradient(prediction)
    dx = torch.abs(dx)
    dy = torch.abs(dy)
    L_TV = torch.sum(dx) + torch.sum(dy)
    return L_TV


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


def contrast_loss(prediction, ground_truth):
    mse_loss = nn.MSELoss()
    prediction_std = torch.std(prediction)
    ground_truth_std = torch.std(ground_truth)
    return mse_loss(ground_truth_std, prediction_std)


def SIMSE(prediction, ground_truth, validity_map):
    simse_image = torch.abs(prediction - ground_truth) / (ground_truth + 1e-6)
    simse_image = simse_image * validity_map
    return torch.mean(simse_image)


def laplacian_loss(prediction, ground_truth):
    L1_loss = nn.L1Loss()
    dx_prediction, dy_prediction = get_image_gradient(prediction)
    dx_sq_prediction, dx_dy_prediction = get_image_gradient(dx_prediction)
    dy_dx_prediction, dy_sq_prediction = get_image_gradient(dy_prediction)
    laplacian_prediction = dx_sq_prediction + dy_sq_prediction

    dx_ground_truth, dy_ground_truth = get_image_gradient(ground_truth)
    dx_sq_ground_truth, dy_dx_ground_truth = get_image_gradient(dx_ground_truth)
    dy_dx_ground_truth, dy_sq_ground_truth = get_image_gradient(dy_ground_truth)
    laplacian_ground_truth = dx_sq_ground_truth + dy_sq_ground_truth
    L_laplacian = L1_loss(laplacian_prediction, laplacian_ground_truth)
    return L_laplacian


def Loss(prediction, ground_truth, validity_map):
    L_depth = depth_loss(prediction, ground_truth)
    L_ssim = SSIM(prediction, ground_truth)
    L_SIMSE = SIMSE(prediction, ground_truth, validity_map)
    return L_depth, L_ssim, L_SIMSE
