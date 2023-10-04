import os

def output_pred_result(preds_loc, name, output_loc):

    with open(os.path.join(preds_loc, name +'.txt'), mode='r') as f:
        filename_list = [line.strip().split(' ') for line in f.readlines()]
        for filename in filename_list:
            _, score, x, y, w, h = map(float, filename)
            filename = f"{int(name)}.txt"
            filepath = os.path.join(output_loc, filename)
            with open(filepath, mode='a') as f:
                f.write(f"{0} {score} {int((x - w/2)*1920)} {int((y - h/2)*1080)} {int((x + w/2)*1920)} {int((y + h/2)*1080)} " + '\n')


def read_filename_list(path, file):
    with open(os.path.join(path, file+'.txt'), mode='r') as fp:
        filename_list = [filename.rstrip() for filename in fp.readlines()]
    return filename_list

if __name__ == '__main__':   

    option = 'test'  
    filename_loc = os.path.join('datasets', 'VOCdevkit', 'VOC2022', 'ImageSets', 'Main')
    preds_loc = os.path.join('datasets', 'GTA_dataset', 'test_preds')
    output_loc = os.path.join('..', '..', 'SE') 

    filename_list = read_filename_list(filename_loc, option)
    for filename in filename_list:
        output_pred_result(preds_loc, filename, output_loc)

    
