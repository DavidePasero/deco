from models.components import Encoder, Cross_Att, Decoder, Classifier, SemanticClassifier, SharedSemanticClassifier
import torch.nn as nn
import torch

class DECO(nn.Module):
    def __init__(self, encoder, context, device, classifier_type='shared'):
        super(DECO, self).__init__()
        self.encoder_type = encoder
        self.context = context
        self.classifier_type = classifier_type

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

        elif self.encoder_type == "dinov2":
            self.correction_conv = nn.Conv1d(1536, 1024, 1).to(device)

            self.encoder = Encoder(encoder="dinov2", device=device)
            if self.context:
                self.decoder_sem = Decoder(1, 133, encoder=encoder).to(device)
                self.decoder_part = Decoder(1, 26, encoder=encoder).to(device)

            self.scene_projector = nn.Linear(1536, 1024).to(device)
            self.contact_projector = nn.Linear(1536, 1024).to(device)
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

        elif self.encoder_type == "dinov2":
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



class DINOContact(nn.Module):
    def __init__(self, device: str = "cuda", classifier_type: str = "shared", *args, **kwargs):
        super(DINOContact, self).__init__()
        self.device = device
        self.classifier = Classifier(1536).to(device)
        self.semantic_classif = SemanticClassifier(1536).to(device) if classifier_type is "shared" else None
        self.encoder = Encoder(encoder="dinov2")

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)

        cont = self.classifier(features)

        if self.semantic_classif is not None:
            sem_cont = self.semantic_classif(features)
            return cont, sem_cont

        return cont
