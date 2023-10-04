import os
import torch
from tqdm import tqdm
from exps.example.yolox_voc.yolox_voc_s import Exp
from scripts.utils import *

if __name__ == '__main__':

    option = 'val'   # val or test
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IMG_H = 1080
    IMG_W = 1920
    conf_threshold = 0.85
    nms_threshold = 0.95
    
    VOCdevkit = os.path.join('datasets', 'VOCdevkit')
    filename_loc = os.path.join(VOCdevkit, 'VOC2022', 'ImageSets', 'Main')
    image_loc = os.path.join(VOCdevkit, 'VOC2022', 'JPEGImages')
    pth_loc = os.path.join('YOLOX_outputs', 'yolox_voc_s', 'best_ckpt.pth')
    output_loc = os.path.join('datasets', 'GTA_dataset', option+'_preds')

    exp = Exp()
    ckpt = torch.load(pth_loc, map_location=device)
    model = exp.get_model().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    filename_list = read_filename_list(filename_loc, option)

    for filename in tqdm(filename_list):
        img = load_image(image_loc, filename)
        img = img.to(device)
        with torch.no_grad():
            output = model(img).cpu()

        mask_pos_x = torch.logical_and(output[0,:,0] > 0, output[0,:,0] < IMG_W)
        mask_pos_y = torch.logical_and(output[0,:,1] > 0, output[0,:,1] < IMG_H)
        mask_foreground = output[0,:,4] > conf_threshold
        mask = torch.logical_and(torch.logical_and(mask_pos_x, mask_pos_y), mask_foreground)
        
        bboxes = output[0,mask,:5].tolist()
        bboxes = nms(bboxes, threshold=nms_threshold)
        Write_Inference_txt(output_loc, filename, bboxes)
