import torch
import torchvision
import torch.nn as nn
import numpy as np

from utils.hrnet import hrnet_w32

# Add transformer imports
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class Encoder(nn.Module):
    def __init__(self, encoder='hrnet', pretrained=True):
        super(Encoder, self).__init__()

        if encoder == 'swin':
            '''Swin Transformer encoder'''
            self.encoder = torchvision.models.swin_b(weights='DEFAULT')
            self.encoder.head = nn.GELU()
        elif encoder == 'hrnet':
            '''HRNet encoder'''
            self.encoder = hrnet_w32(pretrained=pretrained)
        else:
            raise NotImplementedError('Encoder not implemented')

    def forward(self, x):
        out = self.encoder(x)
        return out  

class Self_Attn(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, out_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.query_conv = nn.Conv1d(in_channels = in_dim, out_channels = out_dim, kernel_size = 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim, out_channels = out_dim, kernel_size = 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim, out_channels = out_dim, kernel_size = 1)
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, q, k, v):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height = q.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(q.permute(0, 2, 1))
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(k.permute(0, 2, 1))
        # transpose check, energy: B x N x N, N = H x W
        energy =  torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(v.permute(0, 2, 1))
        # out: B x C x N
        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, C, height)
        out = out/np.sqrt(self.channel_in)
        
        return out

class Cross_Att(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Cross_Att, self).__init__()

        self.cross_attn_1 = Self_Attn(in_dim, out_dim)
        self.cross_attn_2 = Self_Attn(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm([1, in_dim])

    def forward(self, sem_seg, part_seg):
        cross1 = self.cross_attn_1(sem_seg, part_seg, part_seg)
        cross2 = self.cross_attn_1(part_seg, sem_seg, sem_seg)

        out = cross1 * cross2
        out = self.layer_norm(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, encoder='hrnet'):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        if encoder == 'swin':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        elif encoder == 'hrnet':
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_dim),
                # nn.ReLU(),
                # nn.ConvTranspose2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.BatchNorm2d(out_dim),
                nn.Softmax(1)
            )
        else:
            raise NotImplementedError('Decoder not implemented')

    def forward(self, x):
        out = self.upsample(x)
        return out


class ObjectQueryDecoder(nn.Module):
    """
    ObjectQueryDecoder Class:
    Takes the cross-attended feature tensor F_c as input ("memory")
    Uses 80 learnable query vectors (one per object class)
    Processes queries through transformer decoder layers
    Outputs contact probabilities for each object-vertex pair
    """
    def __init__(self, d_model=480, nhead=8, num_decoder_layers=3, dim_feedforward=2048, 
                 num_queries=80, num_vertices=6890, dropout=0.1):
        super(ObjectQueryDecoder, self).__init__()
        
        # Object queries - learnable parameters for each object category
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
        
        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Final MLP to predict vertex contact probabilities for each object class
        self.vertex_predictor = nn.Linear(d_model, num_vertices)
        self.sigmoid = nn.Sigmoid()
        
        # Store dimensions
        self.num_queries = num_queries
        self.num_vertices = num_vertices
        
    def forward(self, memory):
        """
        Args:
            memory: Cross-attended feature tensor F_c [B, 1, D]
        Returns:
            Object-vertex contact probability tensor [B, C, V]
        """
        batch_size = memory.shape[0]
        
        # Repeat object queries for each item in batch
        query = self.query_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, C, D]
        
        # Reshape memory to expected format if needed
        if memory.shape[1] == 1:
            memory = memory.squeeze(1)  # [B, D]
            memory = memory.unsqueeze(1)  # [B, 1, D]
        
        # Apply transformer decoder
        # query: [B, C, D], memory: [B, 1, D]
        tgt = query
        memory = memory.repeat(1, self.num_queries, 1)  # [B, C, D]
        
        # Run transformer decoder
        decoder_output = self.transformer_decoder(tgt, memory)  # [B, C, D]
        
        # Predict vertex contact probabilities for each object class
        contact_logits = self.vertex_predictor(decoder_output)  # [B, C, V]
        contact_probs = self.sigmoid(contact_logits)
        
        return contact_probs

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim=6890, num_object_classes=80, use_transformer=True):
        super(Classifier, self).__init__()
        
        self.out_dim = out_dim
        self.use_transformer = use_transformer
        self.num_object_classes = num_object_classes
        
        if use_transformer:
            # DETR-style object query decoder
            self.object_decoder = ObjectQueryDecoder(
                d_model=in_dim,
                num_queries=num_object_classes,
                num_vertices=out_dim
            )
        else:
            # Original MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, 4096, True), 
                nn.ReLU(),
                nn.Linear(4096, out_dim, True),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.use_transformer:
            # x shape: [B, 1, D]
            # Output shape: [B, C, V]
            contact_probs = self.object_decoder(x)
            return contact_probs
        else:
            # Original implementation
            out = self.classifier(x)
            return out.reshape(-1, self.out_dim)
