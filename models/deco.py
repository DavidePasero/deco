from models.components import Encoder, Cross_Att, Decoder, Classifier, SemanticClassifier, SharedSemanticClassifier
import torch.nn as nn
import torch
from dataclasses import dataclass

@dataclass
class DINOv2DIM:
    LARGE: int = 1024
    GIANT: int = 1536

DINOv2NAME_TO_HIDDEN_DIM = {
    "dinov2-large": DINOv2DIM.LARGE,
    "dinov2-giant": DINOv2DIM.GIANT
}


class DECO(nn.Module):
    def __init__(self, encoder, context, device, classifier_type='shared',
                 train_backbone: bool = False, num_encoders: int = 1):
        super(DECO, self).__init__()
        self.encoder_type = encoder
        self.context = context
        self.classifier_type = classifier_type
        self.train_backbone = train_backbone

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
            self.num_encoder = num_encoders
            hidden_dim = DINOv2NAME_TO_HIDDEN_DIM[self.encoder_type]

            if self.num_encoder > 1:
                self.encoder_sem = Encoder(encoder=self.encoder_type).to(device)
                self.encoder_part = Encoder(encoder=self.encoder_type).to(device)
            else:
                self.encoder = Encoder(encoder=self.encoder_type, device=device)
                self.scene_projector = nn.Linear(hidden_dim, 1024).to(device)
                self.contact_projector = nn.Linear(hidden_dim, 1024).to(device)

            self.correction_conv = nn.Conv1d(hidden_dim, 1024, 1).to(device)

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

        else:
            NotImplementedError('Encoder type not implemented')

        self.device = device

    def forward(self, img):
        if self.encoder_type == 'hrnet':
            sem_enc_out = self.encoder_sem(img)
            part_enc_out = self.encoder_part(img)

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
            cont = self.classif(att)

        elif  "dinov2" in self.encoder_type:
            if self.num_encoders > 1:
                out = self._dinov2_forward_pass_two_encoders(img)
            else:
                out = self._dinov2_forward_pass_shared_encoder(img)

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
        else:  
            # Semantic contact prediction
            semantic_cont = self.semantic_classif(att)

        if self.context: 
            return cont, sem_mask_pred, part_mask_pred, semantic_cont

        return cont, semantic_cont

    def _dinov2_forward_pass_shared_encoder(self, img):
        if self.train_backbone:
            features = self.encoder(img)
        else:
            with torch.no_grad():
                features = self.encoder(img)

        sem_enc_out = self.scene_projector(features)
        part_enc_out = self.contact_projector(features)

        if self.context:
            sem_seg = torch.reshape(sem_enc_out, (-1, 1, 32, 32))
            part_seg = torch.reshape(part_enc_out, (-1, 1, 32, 32))
            sem_mask_pred = self.decoder_sem(sem_seg)
            part_mask_pred = self.decoder_part(part_seg)

        att = self.cross_att(sem_enc_out.unsqueeze(1), part_enc_out.unsqueeze(1))
        cont = self.classif(att)

        if self.context:
            return att, cont, sem_mask_pred, part_mask_pred

        return att, cont

    def _dinov2_forward_pass_two_encoders(self, img):
        sem_enc_out = self.encoder_sem(img)
        part_enc_out = self.encoder_part(img)

        sem_seg = torch.reshape(sem_enc_out, (-1, DINOv2NAME_TO_HIDDEN_DIM[self.encoder_type], 1))
        part_seg = torch.reshape(part_enc_out, (-1, DINOv2NAME_TO_HIDDEN_DIM[self.encoder_type], 1))

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

        if self.context:
            return att, cont, sem_mask_pred, part_mask_pred

        return att, cont


class DINOContact(nn.Module):
    def __init__(self, device: str = "cuda", encoder_name: str = "dinov2-large",
                 classifier_type: str = "shared", train_backbone: bool = False,
                 *args, **kwargs):
        super(DINOContact, self).__init__()
        self.device = device
        self.encoder = Encoder(encoder=encoder_name)
        hidden_dim = DINOv2NAME_TO_HIDDEN_DIM[encoder_name]
        self.classifier = Classifier(hidden_dim).to(device)
        self.semantic_classif = SemanticClassifier(hidden_dim).to(device) if classifier_type is "shared" else None
        self.train_backbone = train_backbone

    def forward(self, x):
        if self.train_backbone:
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)

        cont = self.classifier(features)

        if self.semantic_classif is not None:
            sem_cont = self.semantic_classif(features)
            return cont, sem_cont

        return cont
