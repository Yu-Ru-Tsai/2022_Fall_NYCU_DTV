import os
import csv
from tqdm import tqdm
from model import *
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor

TestImgTform = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
TestImgDir = os.path.join('dataset', 'test')
save_path = os.path.join('HW1_311652010.pt')
# test step
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_file = open(os.path.join('HW1_311652010.csv'),mode='w',newline='')
    writer = csv.writer(result_file)
    writer.writerow(['names','label'])

    model = CNN_model()
    ckpt = torch.load(save_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()     
    with torch.no_grad():
        model.eval()
        for file_name in tqdm(os.listdir(TestImgDir),desc='Testing'):
            image = Image.open(os.path.join(TestImgDir,file_name))
            image = TestImgTform(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            pred = model(image)
            pred = pred.argmax(dim=1).item()
            writer.writerow([file_name,pred])
        result_file.close()