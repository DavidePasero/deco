import torch
import torchvision
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoModel

from utils.hrnet import hrnet_w32

class Encoder(nn.Module):
    def __init__(self, encoder='hrnet', pretrained=True, device="cuda"):
        super(Encoder, self).__init__()

        self.encoder_name = encoder

        if encoder == 'swin':
            '''Swin Transformer encoder'''
            self.encoder = torchvision.models.swin_b(weights='DEFAULT')
            self.encoder.head = nn.GELU()
        elif encoder == 'hrnet':
            '''HRNet encoder'''
            self.encoder = hrnet_w32(pretrained=pretrained)
        elif "dinov2" in encoder:
            #self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            assert encoder in ("dinov2-giant", "dinov2-large", "dinov2-small", "dinov2-base"), "Encoder name must be in dinov2-giant, dinov2-large, dinov2-small, dinov2-base"
            self.encoder = AutoModel.from_pretrained(f'facebook/{encoder}').to(device)
        else:
            raise NotImplementedError('Encoder not implemented')

    def forward(self, x=None, **kwargs):
        if "dinov2" in self.encoder_name:
            # LoRA passes input_ids instead of x.
            x = x if x is not None else kwargs.get('input_ids')
            outputs = self.encoder(x)
            last_hidden_states = outputs.last_hidden_state
            cls_token_embedding = last_hidden_states[:, 0]
            return cls_token_embedding

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
        elif "dinov2" in encoder:
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
        else:
            raise NotImplementedError("Decoder not implemented!")

    def forward(self, x):
        out = self.upsample(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim, num_vertices=6890):
        super(Classifier, self).__init__()

        self.num_vertices = num_vertices

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 4096, True), 
            nn.ReLU(),
            nn.Linear(4096, num_vertices, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.classifier(x)
        return out.reshape(-1, self.num_vertices)


class SemanticClassifier(nn.Module):
    """
    Classifier for semantic contact prediction - predicts which object class
    each vertex is in contact with
    """
    def __init__(self, in_dim, num_classes=70, num_vertices=6890):
        super(SemanticClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.num_vertices = num_vertices
        
        # First transform features to higher dimension
        self.feature_transform = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU()
        )
        
        # Then predict class probabilities for each vertex
        self.vertex_classifiers = nn.Linear(1024, num_vertices * num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input features [B, in_dim]
        Returns:
            Class logits [B, num_classes, num_vertices]
        """
        batch_size = x.shape[0]
        
        # Transform features
        features = self.feature_transform(x)
        
        # Predict class probabilities for all vertices
        vertex_preds = self.vertex_classifiers(features)
        
        # Reshape to [B, num_vertices, num_classes]
        vertex_preds = vertex_preds.view(batch_size, self.num_vertices, self.num_classes)
        
        # Transpose to [B, num_classes, num_vertices] to match expected output format
        vertex_preds = vertex_preds.transpose(1, 2)
        
        return vertex_preds

class SharedSemanticClassifier(nn.Module):
    """
    Shared classifier that predicts the contact class for every mesh vertex.
    The same (image-conditioned) MLP is applied to all 6890 vertices in
    parallel.  Global image features are broadcast across vertices and
    concatenated with a learnable positional embedding for each vertex.
    """
    def __init__(self, features_dim: int, num_classes: int = 70, num_vertices: int = 6890):
        super().__init__()

        self.num_vertices = num_vertices
        self.num_classes = num_classes

        # Per‑vertex positional embedding
        self.pos_embedding = nn.Embedding(num_vertices, features_dim)

        # Two‑layer MLP that is shared by all vertices
        self.mlp = nn.Sequential(
            nn.Linear(features_dim * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)           # → logits for each contact class
        )

    def forward(self, features, vertex_indices=None):
        """
        Args:
            features (Tensor | Dict): If Tensor, shape [B, F] with the same
                cross-attention image features for every vertex.  If Dict, it
                must contain:
                    - "features": Tensor [B, F]
                    - "vertex_pos": LongTensor [B, V] with vertex indices.
            vertex_indices (LongTensor, optional): Explicit vertex indices
                [B, V].  If omitted and `features` is a Dict, the key
                "vertex_pos" is used.  If still None, all vertices
                (0 … 6889) are assumed.
        Returns:
            logits: [B, num_classes, V]
        """
        # Support both the old dict interface and the new positional argument
        if isinstance(features, dict):
            vertex_indices = features["vertex_pos"]
            features = features["features"]

        B, F = features.shape
        if vertex_indices is None:
            # Use the full mesh if no indices are provided
            vertex_indices = torch.arange(self.num_vertices,
                                          device=features.device).unsqueeze(0).expand(B, -1)  # [B, V]

        # [B, V, F] positional embeddings
        pos_emb = self.pos_embedding(vertex_indices)

        # Broadcast global image features to every vertex: [B, 1, F] → [B, V, F]
        global_feat = features.unsqueeze(1).expand(-1, pos_emb.size(1), -1)

        # Concatenate and run the shared MLP
        x = torch.cat([global_feat, pos_emb], dim=-1)             # [B, V, 2F]
        logits = self.mlp(x)                                      # [B, V, C]

        # Rearrange to [B, C, V] expected by downstream code
        return logits.permute(0, 2, 1).contiguous()


class TextFeatureAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super(TextFeatureAggregator, self).__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float16)) #TODO after 1 optim step this becomes -inf
        self.linear = nn.Linear(hidden_dim, hidden_dim).to(torch.float16)

    def forward(self, text_features):
        """
        Args:
            text_features (Tensor): Shape [N, 2048], where N is the number of tokens.

        Returns:
            aggregated_features (Tensor): Shape [1, 2048]
        """
        # Compute attention scores (dot product with query vector)
        attention_scores = torch.matmul(text_features, self.query)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)

        # Compute the weighted sum of the text features
        aggregated_features = (text_features * attention_weights).sum(dim=1).squeeze()

        return aggregated_features.to(torch.float32)
