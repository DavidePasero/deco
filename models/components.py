import torch
import torchvision
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List
import torch.nn.functional as F


from utils.hrnet import hrnet_w32

class Encoder(nn.Module):
    def __init__(self, encoder='hrnet', pretrained=True, device="cuda",
                 return_cls_token: bool = True):
        super(Encoder, self).__init__()

        self.encoder_name = encoder
        self.return_cls_token = return_cls_token

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
            if not self.return_cls_token:
                return last_hidden_states[:, 1:]
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


class EfficientSelfAttention(nn.Module):
    """Self-attention layer using PyTorch's optimized scaled_dot_product_attention"""

    def __init__(self, in_dim, out_dim, num_heads=8):
        super(EfficientSelfAttention, self).__init__()
        self.channel_in = in_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, in_dim)

        self.scale = (self.head_dim) ** -0.5

    def forward(self, x):
        """
        Args:
            x: input features [B, C, L] where L is sequence length
        Returns:
            out: self attention output [B, C, L]
        """
        # Reshape input: [B, C, L] -> [B, L, C]
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [B, L, C] -> [B, L, H, D] -> [B, H, L, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's optimized attention implementation
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # Reshape back: [B, H, L, D] -> [B, L, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Final projection
        output = self.out_proj(attn_output)

        # Return in original format [B, C, L]
        return output


class VisualTextCrossAttention(nn.Module):
    """Cross-attention between visual and text features using PyTorch's optimized attention"""

    def __init__(self, visual_dim, text_dim, num_heads=8):
        super(VisualTextCrossAttention, self).__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        assert visual_dim % num_heads == 0, "visual_dim must be divisible by num_heads"

        # Visual features as queries
        self.q_proj = nn.Linear(visual_dim, visual_dim)

        # Text features as keys and values (with projection to match visual dimension)
        self.k_proj = nn.Linear(text_dim, visual_dim)
        self.v_proj = nn.Linear(text_dim, visual_dim)

        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.layer_norm = nn.LayerNorm(visual_dim)

    def forward(self, v_features, t_features):
        """
        Args:
            visual_features: Visual features [B, L_v, C_v]
            text_features: Text features [B, L_t, C_t]
        Returns:
            out: Cross-attended features in visual feature space [B, L_v, C_v]
        """
        # Reshape inputs: [B, C, L] -> [B, L, C]

        batch_size, v_len, _ = v_features.shape
        _, t_len, _ = t_features.shape

        # Project to queries, keys, values
        q = self.q_proj(v_features)  # [B, L_v, C_v]
        k = self.k_proj(t_features)  # [B, L_t, C_v]
        v = self.v_proj(t_features)  # [B, L_t, C_v]

        # Reshape for multi-head attention
        q = q.view(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_v, D]
        k = k.view(batch_size, t_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_t, D]
        v = v.view(batch_size, t_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_t, D]

        # Use PyTorch's optimized attention implementation
        attn_output = F.scaled_dot_product_attention(q, k, v)  # [B, H, L_v, D]

        # Reshape back: [B, H, L_v, D] -> [B, L_v, C_v]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, v_len, -1)

        # Final projection
        output = self.out_proj(attn_output)  # [B, L_v, C_v]

        # Apply layer norm
        output = self.layer_norm(output)

        # Return in original format [B, C_v, L_v]
        return output


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
        elif "dinov2" in encoder and "vlm" in encoder:
            # Input feature map is of size 18,18,1024
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
    def __init__(self, in_dim, out_dim=6890):
        super(Classifier, self).__init__()

        self.out_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 4096, True), 
            nn.ReLU(),
            nn.Linear(4096, out_dim, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.classifier(x)
        return out.reshape(-1, self.out_dim)


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



class VLMTextEncoder(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.tokenizer = self.model.tokenizer

    def forward(self, texts: List[str], **kwargs):
        #x = x if x is not None else kwargs.get('input_ids')
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        features = self.model(tokens)
        return features[("token_embeddings")]


class AttentivePooling(nn.Module):
    def __init__(self, embed_dim,):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x):
        # x: [B, seq_len, embed_dim]
        attn_scores = torch.einsum('d,bsd->bs', self.query, x)  # [B, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, seq_len]
        attended = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, embed_dim]
        return attended