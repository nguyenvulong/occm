import torch
from models.ssl import SSLModel
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # Compute attention weights
        weights = self.conv(x)
        weights = nn.functional.sigmoid(weights)
        
        # Apply attention to the input feature map
        x = x * weights
        
        return x

class cnn_net_with_attention(nn.Module):
    def __init__(self):
        super(cnn_net_with_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)  # Batch normalization after the first convolution
        self.attention1 = SpatialAttention(in_channels=8)  # Attention layer after the first convolution
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization after the second convolution
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Additional convolutional layer
        self.bn3 = nn.BatchNorm2d(32)  # Batch normalization after the third convolution
        self.attention3 = SpatialAttention(in_channels=32)  # Attention layer after the third convolution
        
        desired_width = 256
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, desired_width))  # Specify the desired output size
        
        self.fc1 = nn.Linear(32 * desired_width, 128)  # Adjust the input size based on your input dimensions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)    # 2 output classes for binary classification
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 0.5 probability
        
    def forward(self, x):
        # Apply convolutions
        x = nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.attention1(x)  # Apply attention
        
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))  # Additional convolutional layer
        x = self.bn3(x)
        x = self.attention3(x)  # Apply attention
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        # Apply fully connected layers with dropout
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class cnn_net_complex(nn.Module):
    def __init__(self):
        super(cnn_net_complex, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)  # Batch normalization after the first convolution
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)  # Batch normalization after the second convolution
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # Additional convolutional layer
        self.bn3 = nn.BatchNorm2d(16)  # Batch normalization after the third convolution
        
        desired_width = 256
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, desired_width))  # Specify the desired output size
        
        self.fc1 = nn.Linear(16 * desired_width, 128)  # Adjust the input size based on your input dimensions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)    # 2 output classes for binary classification
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 0.5 probability
        
    def forward(self, x):
        # Apply convolutions
        x = nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))  # Additional convolutional layer
        x = self.bn3(x)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Calculate the expected size for the view operation
        batch_size = x.size(0)
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)

        # Reshape the tensor for fully connected layers
        x = x.reshape(batch_size, num_channels * height * width)

        # Apply fully connected layers with dropout
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class cnn_net_basic(nn.Module):
    def __init__(self):
        super(cnn_net_basic, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        desired_width = 4096
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, desired_width))  # Specify the desired output size
        
        self.fc1 = nn.Linear(16 * desired_width, 128)  # Adjust the input size based on your input dimensions
        self.fc2 = nn.Linear(128, 64)  # Add an additional fc layer
        self.fc3 = nn.Linear(64, 2)    # 2 output classes for binary classification
        
    def forward(self, x):
        # Apply convolutions
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))  # Apply the additional fc layer
        x = self.fc3(x)
        
        return x
class cnn_net(nn.Module):
    def __init__(self):
        super(cnn_net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)  # Batch normalization after the first convolution
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization after the second convolution
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Additional convolutional layer
        self.bn3 = nn.BatchNorm2d(32)  # Batch normalization after the third convolution
        
        desired_width = 256
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, desired_width))  # Specify the desired output size
        
        self.fc1 = nn.Linear(32 * desired_width, 128)  # Adjust the input size based on your input dimensions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)    # 2 output classes for binary classification
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 0.5 probability
        
    def forward(self, x):
        # Apply convolutions
        x = nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))  # Additional convolutional layer
        x = self.bn3(x)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        # Apply fully connected layers with dropout
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class total_cnn_net(nn.Module):
    def __init__(self, device): 
        super(total_cnn_net, self).__init__()
        self.frontend = SSLModel(device) 
        self.cnn_net = cnn_net()

    def forward(self, x):
        """combine the ssl and cnn net

        Args:
            x (Tensor): (batch_size, time series)
        """
        # print("x.shape", x.shape)
        x = self.frontend.extract_feat(x) # (batch_size, frame, 1024)
        # print("x.shape", x.shape)
        x = x.unsqueeze(1) # (batch_size, 1, frame, 1024)
        x = self.cnn_net(x) # (batch_size, 2)
        return x
        
        


if __name__ == "__main__":
    # Instantiate the model
    model = cnn_net_with_attention()
    # model = cnn_net()

    # Calculate and print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:", total_params)
    for p in model.parameters():
        print(p.numel())
