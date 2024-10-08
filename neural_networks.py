from torch import nn

class Lenet5_Based_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(#cnn layers
            nn.Conv2d(1, 6, 5, padding=2),
            nn.LeakyReLU(),#than tanh
            nn.AvgPool2d(2, stride=2),
            
            nn.Conv2d(6, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=2),
            
            nn.Flatten(),#transform to linear
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, X):
        X = self.layers(X)
        return X
    
