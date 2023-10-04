import torch.nn as nn

class CNN_model(nn.Module):     #(3 x 224 x 224)
    def __init__(self, numClasses=10, dropout: float = 0.5):
        super(CNN_model, self).__init__()
        self.CNN = nn.Sequential( 
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
                
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   

            nn.Conv2d(128, 256, 3, padding=1),   
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           

            nn.Conv2d(256, 256, 1, padding=0),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, numClasses),
        )

    def forward(self, x):
        x = self.CNN(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


        