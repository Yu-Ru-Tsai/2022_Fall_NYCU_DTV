import torch.nn as nn
import torchvision.transforms as transforms
import os
import csv
import random
#import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import pandas as pd
from torch.optim import lr_scheduler
from plot import *
from model import *
from torchsummary import summary
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-bs',type=int,default=32,help='batch size')
parser.add_argument('-ep',type=int,default=200,help='epoch')
parser.add_argument('--model',type=str,default='CNN_model',help='model')
parser.add_argument('--lr',type=float,default='5e-4',help='learning rate')
parser.add_argument('--val_period',type=int,default=5,help='model')

opt = parser.parse_args()

def read_csv(annotations_file):
    with open(annotations_file, newline='') as csvfile:
        rows = csv.reader(csvfile)
        return list(rows)[1:]

class SportDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = read_csv(annotations_file) # list[ [Filename(str), Lable(str)] ]
        self.img_dir = img_dir # string
        self.transform = transform # transform for image
        self.target_transform = target_transform # transform for label

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # setup image path and read image
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = read_image(img_path)

        # read the corresponding label
        label = torch.tensor(int(self.img_labels[idx][1])).long()

        # Some transformation on image
        if self.transform:
            image = self.transform(image)
        
        # Some transformation on label
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class ImgTo01:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.float() / 255
        return img

class RandomGaussianNoise:
    def __init__(self, p=0.1, sig=0.01):
        self.sig = sig
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
                img += self.sig * torch.randn(img.shape)

        return img

# dir and path
TrainAnno = os.path.join('dataset', 'train.csv')
TrainImgDir = os.path.join('dataset', 'train')
ValidAnno = os.path.join('dataset', 'val.csv')
ValidImgDir = os.path.join('dataset', 'val')

# Transform
TrainImgTform = transforms.Compose([
    ImgTo01(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    RandomGaussianNoise(),
    transforms.RandomGrayscale(),
    transforms.RandomPerspective(distortion_scale=0.3,p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
])
ValidImgTform = transforms.Compose([
    ImgTo01(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Dataset
TrainDataset = SportDataset(TrainAnno, TrainImgDir, transform=TrainImgTform)
TrainLoader = DataLoader(TrainDataset, opt.bs, shuffle=True)
ValidDataset = SportDataset(ValidAnno, ValidImgDir, transform=ValidImgTform)
ValidLoader = DataLoader(ValidDataset, batch_size=1, shuffle=False)

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_acc = 0.75
    model = CNN_model()
    summary(model,(3,224,224))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
    save_path = os.path.join('HW1_311652010.pt')

    # Training step
    train_record = {'loss': [], 'accuracy': []} 
    valid_record = {'loss': [], 'accuracy': []} 
    for epoch in range(1, opt.ep+1):
        model.train()
        num, correct, total_loss = 0, 0, 0
        train_bar = tqdm(TrainLoader, desc=f'Training {epoch}')
        for data in train_bar:
            train_bar.update()
            image, label = data
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(image)
            t_loss = criterion(pred, label)
            t_loss.backward()
            optimizer.step()
            num += image.shape[0]
            correct += (pred.argmax(dim=1) == label).sum()
            T_acc = correct/num
            total_loss += t_loss*image.shape[0]
            T_loss = total_loss/num
            train_bar.set_postfix({
                'Loss' :T_loss.item(), 'Acc' : T_acc.item()
            })
        train_bar.close()
        train_acc = correct / num
        train_loss = total_loss / num
        train_record['loss'].append(train_loss.detach().cpu().item())
        train_record['accuracy'].append(train_acc.detach().cpu().item())
        scheduler.step()

    # valid step
        if(epoch % opt.val_period == 0):
            num, correct, total_loss= 0, 0, 0
            model.eval()
            with torch.no_grad():
                valid_bar = tqdm(ValidLoader, desc=f'Valid {epoch}')
                for data in valid_bar:
                    image, label = data
                    image = image.to(device)
                    label = label.to(device)
                    pred = model(image)
                    v_loss = criterion(pred, label)
                    num += image.shape[0]
                    correct += (pred.argmax(dim=1) == label).sum()
                    V_acc = correct/num
                    total_loss += v_loss*image.shape[0]
                    V_loss = total_loss/num
                    valid_bar.set_postfix({
                        'Loss' : V_loss.item(), 'Acc' : V_acc.item()
                    })
                valid_bar.close()
                valid_acc = correct/ num
                valid_loss = total_loss / num
                if(valid_acc > max_acc):
                    max_acc = valid_acc
                    torch.save(model.state_dict(), save_path)
            valid_record['loss'].append(valid_loss.detach().cpu().item())
            valid_record['accuracy'].append(valid_acc.detach().cpu().item()) 

        path= os.path.join('train_record.csv')
        file=open(path, 'w')
        writer=csv.writer(file)
        df = pd.DataFrame(data=train_record)
        df.to_csv("train_record.csv")
        file.close()

        path= os.path.join('valid_record.csv')
        file=open(path, 'w')
        writer=csv.writer(file)
        df = pd.DataFrame(data=valid_record)
        df.to_csv("valid_record.csv")
        file.close()

    # plot
    train_data = pd.read_csv('train_record.csv')
    valid_data = pd.read_csv('valid_record.csv')
    plot_learning_curve(train_data["loss"], valid_data["loss"], title='CNN model')
    plot_accuracy_curve(train_data["accuracy"], valid_data["accuracy"], title='CNN model')  
