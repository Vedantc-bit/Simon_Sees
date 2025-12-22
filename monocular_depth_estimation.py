import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
  #ananya ko credit do ananya ko naye do
  
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class MonocularDepthEstimationModel(nn.Module):
   
    def __init__(self):
        super(MonocularDepthEstimationModel, self).__init__()

        # Encoder (Down-sampling path)
        # This part of the network captures the context of the image.
        self.enc1 = conv_block(3, 64)   #input 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck (the deepest part of the network)
        self.bottleneck = conv_block(256, 512)

        # Decoder (Up-sampling path)
        # This part reconstructs the image to create the depth map.
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256) # 256 from upconv + 256 from enc3 skip connection
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128) # 128 from upconv + 128 from enc2 skip connection
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)  # 64 from upconv + 64 from enc1 skip connection
        
        # Output Layer
        # A final 1x1 convolution to map the features to a single-channel depth map.
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder 
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder with Skip Connections
        # Skip connections are crucial: they combine high-level features from the
        # decoder with detailed, low-level features from the encoder.
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        # Output
        output = self.output_conv(d1)
        return output