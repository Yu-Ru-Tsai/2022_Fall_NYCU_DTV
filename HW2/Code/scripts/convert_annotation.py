import os
from tqdm import tqdm
from scripts.utils import (read_filename_list,
                           read_annotation_txt,
                           write_annotation_xml)

if __name__ == '__main__':
    options = ['train', 'val']
    
    VOCdevkit = os.path.join('datasets', 'VOCdevkit')
    GTAdataset = os.path.join('datasets','GTA_dataset')
    

    # For train and val
    for option in options:
        filename_loc = os.path.join(VOCdevkit, 'VOC2022', 'ImageSets', 'Main')
        anno_xml_loc = os.path.join(VOCdevkit, 'VOC2022', 'Annotations')
        anno_txt_loc = os.path.join(GTAdataset, option+'_labels')
        filename_list = read_filename_list(filename_loc, option)
        for filename in tqdm(filename_list): 
            bboxes = read_annotation_txt(anno_txt_loc, filename)
            write_annotation_xml(anno_xml_loc, filename, bboxes)
