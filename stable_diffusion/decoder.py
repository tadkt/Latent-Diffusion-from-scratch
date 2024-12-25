# import torch
# from torch import nn
# from torch.nn import functional as F
# from attention import SelfAttention

# class VAE_AttentionBlock(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.groupnorm = nn.GroupNorm(32, channels)
#         self.attention = SelfAttention(1, channels)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # B, F, H, W

#         residue = x

#         n, c, h, w = x.shape

#         # B, F, H, W -> B, F, H*W
#         x = x.view(n, c, h*w)

#         # B, F, H*W -> B, H*W, F
#         x = x.transpose(-1, -2)

#         # B, H*W, F -> B, H*W, F
#         x = self.attention(x)
        
#         # B, H*W, F -> B, F, H*W
#         x = x.transpose(-1, -2)

#         # B, F, H*W -> B, F, H, W
#         x = x.view((n, c, h, w))

#         x += residue

#         return x

# # Math: Normalization y = \frac{x - E[x]}{\sqrt{Var[x]+\epsilon}}*\gamma + \beta

# class VAE_ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.groupnorm_1 = nn.GroupNorm(32, in_channels)
#         self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

#         self.groupnorm_2 = nn.GroupNorm(32, out_channels)
#         self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

#         if in_channels == out_channels:
#             self.residual_layer = nn.Identity()
#         else:
#             self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: B, IC, H, W
#         residue = x

#         x = self.groupnorm_1(x)

#         x = F.silu(x)

#         x = self.conv_1(x)

#         x = self.groupnorm_2(x)

#         x = F.silu(x)

#         x = self.conv_2(x)

#         return x + self.residual_layer(residue)
    
# class VAE_decoder(nn.Sequential):
#     def __init__(self):
#         super().__init__(
#             nn.Conv2d(4, 4, kernel_size=1, padding=0),
#             nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
#             VAE_ResidualBlock(512, 512),

#             VAE_AttentionBlock(512),

#             VAE_ResidualBlock(512, 512),

#             VAE_ResidualBlock(512, 512),

#             VAE_ResidualBlock(512, 512),

#             # B, 512, H/8, W/8 -> B, 512, H/8, W/8
#             VAE_ResidualBlock(512, 512),

#             # B, 512, H/8, W/8 -> B, 512, H/4, W/4
#             nn.Upsample(scale_factor=2),

#             nn.Conv2d(512, 512, kernel_size=3, padding=1),

#             VAE_ResidualBlock(512, 512),
#             VAE_ResidualBlock(512, 512),
#             VAE_ResidualBlock(512, 512),

#             # B, 512, H/4, W/4 -> B, 512, H/2, W/2
#             nn.Upsample(scale_factor=2),

#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             VAE_ResidualBlock(512, 256),
#             VAE_ResidualBlock(256, 256),
#             VAE_ResidualBlock(256, 256),

#             # B, 256, H/2, W/2 -> B, 256, H, W
#             nn.Upsample(scale_factor=2),

#             nn.Conv2d(256, 256, kernel_size=3, padding=1),

#             VAE_ResidualBlock(256, 128),
#             VAE_ResidualBlock(128, 128),
#             VAE_ResidualBlock(128, 128),

#             nn.GroupNorm(32, 128),

#             nn.SiLU(),

#             # B, 128, H, W -> B, 3, H, W
#             nn.Conv2d(128, 3, kernel_size=3, padding=1)
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x /= 0.18215
        
#         for module in self:
#             x = module(x)

#         return x

### TRY CODE ###
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)

        residue = x 

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x = x.transpose(-1, -2)
        
        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
        x += residue

        # (Batch_Size, Features, Height, Width)
        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)

        residue = x

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.SiLU(), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x

