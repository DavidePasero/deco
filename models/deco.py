from models.components import (Encoder, Cross_Att, Decoder, Classifier, SemanticClassifier, SharedSemanticClassifier,
                               VLMTextEncoder, EfficientSelfAttention, VisualTextCrossAttention, AttentivePooling)
import torch.nn as nn
import torch
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType
import torchvision.transforms as tt
import torch.nn.functional as F

from models.vlm import *

@dataclass
class DINOv2DIM:
    LARGE: int = 1024
    GIANT: int = 1536


DINOv2NAME_TO_HIDDEN_DIM = {
    "dinov2-large": DINOv2DIM.LARGE,
    "dinov2-giant": DINOv2DIM.GIANT
}

SMOL_HIDDEN_DIM = 768  # hidden size of SmolVLM‑Base


def shared_semantic_classifier(self, att, cont):
    # Use SharedSemanticClassifier
    batch_size = att.shape[0]
    num_vertices = 6890
    num_classes = self.semantic_classif.num_classes

    # Initialize output tensor
    semantic_cont = torch.zeros(batch_size, num_classes, num_vertices).to(self.device)

    # Get contact vertices (where cont > threshold)
    contact_mask = (cont > 0.5)

    if contact_mask.sum() > 0:
        # Process vertices with contact in parallel using torch.nn.utils.parametrize.cached
        with torch.nn.utils.parametrize.cached():
            for b in range(batch_size):
                # Get vertices with contact for this batch item
                batch_contacts = contact_mask[b].squeeze()

                if batch_contacts.sum() > 0:
                    # Get indices of contact vertices
                    contact_vertices = batch_contacts.nonzero(as_tuple=True)[0]

                    # Create position embeddings for all contact vertices at once
                    vertex_positions = torch.arange(num_vertices).to(self.device)
                    vertex_pos_embeddings = self.semantic_classif.pos_embedding(vertex_positions)

                    # Get embeddings only for contact vertices
                    contact_pos_embeddings = vertex_pos_embeddings[contact_vertices]

                    # Repeat attention features for each contact vertex
                    repeated_features = att[b].expand(len(contact_vertices), -1, -1)

                    # Concatenate features with position embeddings
                    combined_features = torch.cat([
                        repeated_features.squeeze(1),
                        contact_pos_embeddings
                    ], dim=1)

                    # Apply classifier to all vertices at once
                    class_predictions = self.semantic_classif.feature_transform(combined_features)

                    # Place predictions in output tensor
                    semantic_cont[b, :, contact_vertices] = class_predictions.t()

    return semantic_cont

class DECO(nn.Module):
    def __init__(self,
                 encoder,
                 context,
                 device,
                 classifier_type='shared',
                 num_encoders: int = 1,
                 use_vlm: bool = False,
                 train_backbone: bool = False,
                 lora_r: int = 8,
                 lora_alpha: int = 32,
                 train_vlm_text_encoder = False,
                 patch_cross_attention=False):
        super(DECO, self).__init__()
        self.encoder_type = encoder
        self.context = context
        self.classifier_type = classifier_type
        self.train_backbone = train_backbone
        self.use_vlm = use_vlm
        self.train_vlm_text_encoder = train_vlm_text_encoder
        self.num_encoders = num_encoders
        self.patch_cross_attention = patch_cross_attention

        if self.encoder_type == 'hrnet':
            self.encoder_sem = Encoder(encoder=encoder).to(device)
            self.encoder_part = Encoder(encoder=encoder).to(device)
            if self.context:
                self.decoder_sem = Decoder(480, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(480, 26, encoder=encoder).to(device)
            self.sem_pool = nn.AdaptiveAvgPool2d((1))
            self.part_pool = nn.AdaptiveAvgPool2d((1))
            self.cross_att = Cross_Att(480, 480).to(device)
            self.classif = Classifier(480).to(device)
            if self.classifier_type == 'shared':
                self.semantic_classif = SharedSemanticClassifier(480).to(device)
            else:
                # Add semantic classifier with correct input dimension for hrnet
                self.semantic_classif = SemanticClassifier(480).to(device)
        elif self.encoder_type == 'swin':
            self.encoder_sem = Encoder(encoder=encoder).to(device)
            self.encoder_part = Encoder(encoder=encoder).to(device)
            self.correction_conv = nn.Conv1d(768, 1024, 1).to(device)
            if self.context:
                self.decoder_sem = Decoder(1, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(1, 26, encoder=encoder).to(device)
            self.cross_att = Cross_Att(1024, 1024).to(device)
            self.classif = Classifier(1024).to(device)
            # Add semantic classifier with correct input dimension for swin
            if self.classifier_type == 'shared':
                self.semantic_classif = SharedSemanticClassifier(1024).to(device)
            else:
                self.semantic_classif = SemanticClassifier(1024).to(device)

        elif "dinov2" in self.encoder_type:
            self.num_encoders = num_encoders
            hidden_dim = DINOv2NAME_TO_HIDDEN_DIM[self.encoder_type]

            if self.num_encoders > 1:
                self.encoder_sem = Encoder(encoder=self.encoder_type, return_cls_token=not patch_cross_attention).to(device)
                self.encoder_part = Encoder(encoder=self.encoder_type, return_cls_token=not patch_cross_attention).to(device)
            else:
                self.encoder = Encoder(encoder=self.encoder_type, device=device, return_cls_token=not patch_cross_attention).to(device)
                self.scene_projector = nn.Linear(hidden_dim, 1024).to(device)
                self.contact_projector = nn.Linear(hidden_dim, 1024).to(device)

            # Freeze DINOv2 backbone parameters
            if self.num_encoders > 1:
                for p in self.encoder_sem.parameters():
                    p.requires_grad = False
                for p in self.encoder_part.parameters():
                    p.requires_grad = False
            else:
                for p in self.encoder.parameters():
                    p.requires_grad = False

            self.correction_conv = nn.Conv1d(hidden_dim, 1024, 1).to(device)

            if  self.context:
                self.decoder_sem = Decoder(1, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(1, 26, encoder=encoder).to(device)

            if patch_cross_attention:
                self.cross_att = VisualTextCrossAttention(hidden_dim, hidden_dim).to(device)
                self.self_att = EfficientSelfAttention(hidden_dim, hidden_dim).to(device)
                self.pooling = AttentivePooling(hidden_dim).to(device)
            else:
                self.cross_att = Cross_Att(hidden_dim, hidden_dim).to(device)

            self.classif = Classifier(hidden_dim).to(device)
            # Add semantic classifier with correct input dimension for swin
            if self.classifier_type == 'shared':
                self.semantic_classif = SharedSemanticClassifier(hidden_dim).to(device)
            else:
                self.semantic_classif = SemanticClassifier(hidden_dim ).to(device)

            # ---- LoRA adaptation on DINOv2 backbone ----
            if self.train_backbone and 'dinov2' in self.encoder_type:
                print("--------------------------------------------- Training Backbone with LoRA  ---------------------------------------------")
                lora_cfg = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=[
                        "attention.attention.query",
                        "attention.attention.key",
                        "attention.attention.value",
                        "attention.output.dense"
                    ]
                )
                if self.num_encoders > 1:
                    # wrap both encoders with LoRA
                    self.encoder_sem = get_peft_model(self.encoder_sem, lora_cfg)
                    self.encoder_part = get_peft_model(self.encoder_part, lora_cfg)
                else:
                    # wrap single shared encoder with LoRA
                    self.encoder = get_peft_model(self.encoder, lora_cfg)

        else:
            NotImplementedError('Encoder type not implemented')

        # -----------------------  SmolVLM branch  ------------------------
        self.use_vlm = use_vlm
        if self.use_vlm:
            print("--------------------------------------------- Using VLM  ---------------------------------------------")
            if encoder == "hrnet":
                num_channels = 480
            else:
                num_channels = DINOv2NAME_TO_HIDDEN_DIM[encoder]

            self.vlm_text_encoder = VLMTextEncoder(device=device)

            if self.train_vlm_text_encoder:
                print("--------------------------------------------- Training VLM Text Encoder with LoRA  ---------------------------------------------")
                lora_cfg = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q", "k", "v", "o"]
                )
                self.vlm_text_encoder = get_peft_model(self.vlm_text_encoder, lora_cfg)

            self.text_projector = nn.Linear(768, num_channels).to(device)

            if self.num_encoders == 2:
                self.cross_att_text_sem = VisualTextCrossAttention(num_channels, 768).to(device)
                self.sem_self_att = EfficientSelfAttention(num_channels, num_channels).to(device)
                self.cross_att_text_part = VisualTextCrossAttention(num_channels, 768).to(device)
                self.part_self_att = EfficientSelfAttention(num_channels, num_channels).to(device)
                self.sem_proj = nn.Linear(num_channels, num_channels).to(device)
                self.part_proj = nn.Linear(num_channels, num_channels).to(device)
                self.pooling = AttentivePooling(num_channels).to(device)
            else:
                self.cross_att_text = VisualTextCrossAttention(num_channels, 768).to(device)
                self.self_att = EfficientSelfAttention(num_channels, num_channels).to(device)

            self.dec_proj_sem = nn.Linear(num_channels, 1024).to(device)
            self.dec_proj_part = nn.Linear(num_channels, 1024).to(device)

        self.device = device

    def get_encoder_params(self):
        """
        Returns a list of encoder parameters that require gradients.
        Handles both single-encoder and two-encoder setups.
        """
        if self.num_encoders == 1:
            return [p for p in self.encoder.parameters() if p.requires_grad]
        else:
            return ([p for p in self.encoder_sem.parameters() if p.requires_grad] +
                    [p for p in self.encoder_part.parameters() if p.requires_grad])

    def get_semantic_branch_params(self):
        """
        Returns parameters for the semantic segmentation head:
        scene_projector (if present), decoder_sem (if context), and correction_conv.
        """
        params = []
        if self.context:
            if self.num_encoders == 1:
                params += list(self.scene_projector.parameters())
            params += list(self.decoder_sem.parameters())
            params += list(self.correction_conv.parameters()) if hasattr(self, "correction_conv") else []
            params += list(self.dec_proj_sem.parameters())
        return params

    def get_part_branch_params(self):
        """
        Returns parameters for the part segmentation head:
        contact_projector (if present), decoder_part (if context), and correction_conv.
        """
        params = []
        if self.context:
            if self.num_encoders == 1:
                params += list(self.contact_projector.parameters())
            params += list(self.decoder_part.parameters())
            params += list(self.correction_conv.parameters()) if hasattr(self, "correction_conv") else []
            params += list(self.dec_proj_part.parameters())

        return params

    def get_contact_branch_params(self):
        """
        Returns parameters for the contact head: cross-attention, classifier, and correction_conv.
        """
        base_params = list(self.cross_att.parameters()) + list(self.classif.parameters()) + list(self.semantic_classif.parameters())

        if self.patch_cross_attention:
            base_params += list(self.pooling.parameters())

        return base_params

    def get_vlm_params(self):
        """
        Returns parameters for the VLM text encoder and text projector.
        """
        if not self.use_vlm:
            return []

        base_params = list()

        if self.num_encoders == 2:
            base_params += list(self.cross_att_text_sem.parameters())  + list(self.sem_self_att.parameters()) + list(self.sem_proj.parameters())
            base_params += list(self.cross_att_text_part.parameters()) + list(self.part_self_att.parameters()) + list(self.part_proj.parameters())
        else:
            base_params += list(self.cross_att_text.parameters()) + list(self.self_att.parameters()) + list(self.proj.parameters())

        if self.train_vlm_text_encoder:
            base_params += list(self.vlm_text_encoder.parameters())

        if self.patch_cross_attention:
            base_params += list(self.pooling.parameters())

        return base_params

    def forward(self, img, vlm_feats = None):
        if self.encoder_type == 'hrnet':
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            if self.use_vlm:
                raise ValueError("VLM does not support HRNet yet")
                #sem_enc_out = self._apply_vlm(vlm_feats, visual_features=sem_enc_out, target_branch="sem")
                #part_enc_out = self._apply_vlm(vlm_feats, visual_features=part_enc_out, target_branch="part")

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_enc_out)
                part_mask_pred = self.decoder_part(part_enc_out)

            sem_enc_out = self.sem_pool(sem_enc_out)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.squeeze(2)
            sem_enc_out = sem_enc_out.unsqueeze(1)

            part_enc_out = self.part_pool(part_enc_out)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.squeeze(2)
            part_enc_out = part_enc_out.unsqueeze(1)

            att = self.cross_att(sem_enc_out, part_enc_out)

            # --- Optional VLM conditioning ---
            if self.use_vlm:
                att = self._apply_vlm(vlm_feats, visual_features=att)

            cont = self.classif(att)

        elif "dinov2" in self.encoder_type:
            if self.num_encoders > 1:
                out = self._dinov2_forward_pass_two_encoders(img, vlm_feats)
            else:
                out = self._dinov2_forward_pass_shared_encoder(img, vlm_feats)

            if self.context:
                att, cont, sem_mask_pred, part_mask_pred = out
            else:
                att, cont = out

        else:
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

            sem_seg = torch.reshape(sem_enc_out, (-1, 768, 1))
            part_seg = torch.reshape(part_enc_out, (-1, 768, 1))

            sem_seg = self.correction_conv(sem_seg)
            part_seg = self.correction_conv(part_seg)

            sem_seg = torch.reshape(sem_seg, (-1, 1, 32, 32))
            part_seg = torch.reshape(part_seg, (-1, 1, 32, 32))

            if self.context:
                sem_mask_pred = self.decoder_sem(sem_seg)
                part_mask_pred = self.decoder_part(part_seg)

            sem_enc_out = torch.reshape(sem_seg, (-1, 1, 1024))
            part_enc_out = torch.reshape(part_seg, (-1, 1, 1024))

            att = self.cross_att(sem_enc_out, part_enc_out)
            cont = self.classif(att)

        if self.classifier_type == 'shared':
            # ------------------------------------------------------------------
            # Vectorised semantic‑contact prediction without per‑sample loops.
            #
            #   att   : [B, 1, F]   (global image features)
            #   cont  : [B, 1, 6 890]  (contact logits – not probabilities!)
            #
            # Strategy:
            #   • Run the shared classifier on *all* vertices in parallel
            #     → logits_all : [B, C, 6890]
            # ------------------------------------------------------------------
            feats = att.squeeze(1)  # [B, F]
            logits_all = self.semantic_classif(feats)  # [B, C, 6890]
        else: #TODO implement correctly
            # Semantic contact prediction with the separate (non‑shared) classifier
            semantic_cont = self.semantic_classif(att)

        if self.context:
            return cont, sem_mask_pred, part_mask_pred, logits_all

        return cont, logits_all

    def _apply_vlm(self, vlm_texts, visual_features, target_branch: str = "default"):
        text_feature = self.vlm_text_encoder(texts=vlm_texts)
        if len(text_feature.shape) == 1:   #[B,1024]
            text_feature = text_feature.unsqueeze(0)

        if target_branch == "default":
            att = self.cross_att_text(visual_features, text_feature)
            att = self.self_att(att)
        elif target_branch == "sem":
            att = self.cross_att_text_sem(visual_features, text_feature)
            att = self.sem_self_att(att)
        elif target_branch == "part":
            att = self.cross_att_text_part(visual_features, text_feature)
            att = self.part_self_att(att)
        else:
            raise ValueError(f"Unknown target branch: {target_branch}")

        return att


    def _dinov2_forward_pass_shared_encoder(self, img, vlm_feats = None):
        if self.train_backbone:
            features = self.encoder(img)
        else:
            with torch.no_grad():
                features = self.encoder(img)

        sem_enc_out = self.scene_projector(features)
        part_enc_out = self.contact_projector(features)

        if self.context:
            dec_feats = self.dec_proj_sem(sem_enc_out)
            part_feats = self.dec_proj_part(part_enc_out)
            sem_seg = torch.reshape(dec_feats, (-1, 1, 32, 32))
            part_seg = torch.reshape(part_feats, (-1, 1, 32, 32))
            sem_mask_pred = self.decoder_sem(sem_seg)
            part_mask_pred = self.decoder_part(part_seg)

        att = self.cross_att(sem_enc_out.unsqueeze(1), part_enc_out.unsqueeze(1))

        if self.use_vlm:
            att = self._apply_vlm(vlm_feats, visual_features=att)

        if self.patch_cross_attention:
            att = self.pooling(att)

        cont = self.classif(att)

        if self.context:
            return att, cont, sem_mask_pred, part_mask_pred

        return att, cont

    def _dinov2_forward_pass_two_encoders(self, img, vlm_feats = None):
        sem_enc_out = self.encoder_sem(img)
        part_enc_out = self.encoder_part(img)

        if self.use_vlm:
            sem_enc_out = self._apply_vlm(vlm_feats, visual_features=sem_enc_out, target_branch="sem")
            part_enc_out = self._apply_vlm(vlm_feats, visual_features=part_enc_out, target_branch="part")


        if self.context:
            dec_feats = self.dec_proj_sem(sem_enc_out)
            part_feats = self.dec_proj_part(part_enc_out)
            sem_mask_pred = self.decoder_sem(dec_feats.mean(axis=1).reshape(-1, 1, 32, 32))
            part_mask_pred = self.decoder_part(part_feats.mean(axis=1).reshape(-1, 1, 32, 32))

       # sem_enc_out = torch.reshape(sem_seg, (-1, 1, 1024))
       # part_enc_out = torch.reshape(part_seg, (-1, 1, 1024))

        att = self.cross_att(sem_enc_out, part_enc_out)
        if self.patch_cross_attention:
            att = self.pooling(att)

        cont = self.classif(att)

        if self.context:
            return att, cont, sem_mask_pred, part_mask_pred

        return att, cont


class DINOContact(nn.Module):
    def __init__(
        self, device: str = "cuda", encoder_name: str = "dinov2-large",
        classifier_type: str = "shared", train_backbone: bool = False,
        lora_r: int = 8, lora_alpha: int = 32,
        use_vlm: bool = False, train_vlm_text_encoder: bool = False,
        *args, **kwargs
    ):
        super(DINOContact, self).__init__()
        self.device = device
        self.use_vlm = use_vlm
        self.train_vlm_text_encoder = train_vlm_text_encoder
        self.train_backbone = train_backbone

        # --- Encoder + LoRA for backbone ---
        self.encoder = Encoder(encoder=encoder_name, return_cls_token= not use_vlm).to(device)
        hidden_dim = DINOv2NAME_TO_HIDDEN_DIM[encoder_name]
        if self.train_backbone:
            print("[DINOContact] Training backbone with LoRA")
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "attention.attention.query",
                    "attention.attention.key",
                    "attention.attention.value",
                    "attention.output.dense"
                ]
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        # --- VLM branch ---
        if self.use_vlm:
            print("[DINOContact] Using VLM branch")
            self.vlm_text_encoder = VLMTextEncoder(device=device)
            if self.train_vlm_text_encoder:
                print("[DINOContact] Training VLM Text Encoder with LoRA")
                vlm_lora_cfg = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q", "k", "v", "o"]
                )
                self.vlm_text_encoder = get_peft_model(self.vlm_text_encoder, vlm_lora_cfg)
            self.text_projector = nn.Linear(768, hidden_dim).to(device)
            self.cross_att_text = VisualTextCrossAttention(hidden_dim, 768).to(device)
            self.self_att = EfficientSelfAttention(hidden_dim, hidden_dim).to(device)
            self.pooling = AttentivePooling(hidden_dim).to(device)

        # --- Classifiers ---
        self.classifier = Classifier(hidden_dim).to(device)
        self.semantic_classif = SharedSemanticClassifier(hidden_dim).to(device) if classifier_type == "shared" else None
        self.pooling = AttentivePooling(hidden_dim).to(device)

    def forward(self, x, vlm_feats=None):
        if self.train_backbone:
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)

        # --- VLM branch ---
        if self.use_vlm and vlm_feats is not None:
            text_feature = self.vlm_text_encoder(texts=vlm_feats)
            if len(text_feature.shape) == 1:
                text_feature = text_feature.unsqueeze(0)
            att = self.cross_att_text(features, text_feature)
            att = self.self_att(att)
            att = self.pooling(att)  # [B, hidden_dim]
        else:
            att = self.pooling(features)  # [B, hidden_dim]

        cont = self.classifier(att.unsqueeze(1))  # expects [B, 1, F] or [B, F]

        logits_all = None
        if self.semantic_classif is not None:
            feats = att  # already pooled [B, F]
            logits_all = self.semantic_classif(feats)  # [B, C, 6890]

        return (cont, logits_all) if logits_all is not None else cont

    def get_params(self):
        base_params = list(self.encoder.parameters()) + list(self.classifier.parameters())

        if self.use_vlm:
            if self.train_vlm_text_encoder:
                base_params += list(self.vlm_text_encoder.parameters())

            base_params += list(self.text_projector.parameters()) + list(self.cross_att_text.parameters()) + list(self.self_att.parameters()) +  list(self.pooling.parameters())

        if self.semantic_classif is not None:
            base_params += list(self.semantic_classif.parameters())

        return base_params




