# coding=utf-8
import numpy as np
import torch
import os
import cv2 as cv
from train.train import UNetModel

def test(unet, root_dir, results_dir):
    unet = UNetModel().cuda()
    unet.load_state_dict(torch.load('train\\unet_road_model.pt'))
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        h, w = image.shape
        img = np.float32(image) /255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view( 1, 1, h, w)
        probs = unet(x_input.cuda())
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        _, output = m_label_out_.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()
        result = cv.resize(np.uint8(predic_), (w, h))
        result_filename = os.path.join(results_dir, f"{f}")
        cv.imwrite(result_filename, result)