from utils.loss import sem_loss_function, class_loss_function, pixel_anchoring_function, MultiClassContactLoss
import torch
import os
import time

"""
Manages the training process
Contains the TrainStepper class that handles optimization, loss calculation, and evaluation
Processes model outputs and computes appropriate losses
"""

class TrainStepper():
    def __init__(self, deco_model, context, learning_rate, loss_weight, pal_loss_weight, device, 
                 use_transformer=True, num_object_classes=80):
        self.device = device
        self.model = deco_model
        self.context = context
        self.use_transformer = use_transformer
        self.num_object_classes = num_object_classes

        # Initialize optimizers
        if self.context:
            self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
                                                lr=learning_rate, weight_decay=0.0001)
            self.optimizer_part = torch.optim.Adam(
                params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()), lr=learning_rate,
                weight_decay=0.0001)
        self.optimizer_contact = torch.optim.Adam(
            params=list(self.model.encoder_sem.parameters()) + list(self.model.encoder_part.parameters()) + list(
                self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=learning_rate, weight_decay=0.0001)

        # Initialize loss functions
        if self.context: 
            self.sem_loss = sem_loss_function().to(device)
        
        # Use different loss functions based on whether we're using the transformer
        if self.use_transformer:
            self.class_loss = MultiClassContactLoss(
                contact_weight=1.0, 
                class_weight=0.5,
                dist_weight=0.2
            ).to(device)
        else:
            self.class_loss = class_loss_function().to(device)
            
        self.pixel_anchoring_loss_smplx = pixel_anchoring_function(model_type='smplx').to(device)
        self.pixel_anchoring_loss_smpl = pixel_anchoring_function(model_type='smpl').to(device)
        self.lr = learning_rate
        self.loss_weight = loss_weight
        self.pal_loss_weight = pal_loss_weight

    def optimize(self, batch):
        """
        Performs one optimization step with the given batch of data
        
        Args:
            batch: Dictionary containing training data
        
        Returns:
            losses: Dictionary of computed losses
            output: Dictionary of model outputs and ground truth for visualization
        """
        self.model.train()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        # Get binary contact labels
        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        
        # Get semantic contact labels if available
        if 'semantic_contact' in batch:
            semantic_contact = batch['semantic_contact'].to(self.device)
            has_semantic_contact = batch['has_semantic_contact'].to(self.device) if 'has_semantic_contact' in batch else torch.zeros(img.shape[0]).to(self.device)
        else:
            # Create dummy semantic contact with one class
            semantic_contact = gt_contact_labels_3d.unsqueeze(1)
            has_semantic_contact = torch.zeros(img.shape[0]).to(self.device)

        if self.context:
            sem_mask_gt = batch['sem_mask'].to(self.device)
            part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Forward pass
        if self.context:
            cont, sem_mask_pred, part_mask_pred = self.model(img)
        else:
            cont = self.model(img)    

        # Calculate losses
        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        
        # Use different loss calculation based on whether we're using the transformer
        if self.use_transformer:
            # For transformer model with multi-class output
            # Use semantic contact labels if available, otherwise use binary contacts
            if has_semantic_contact.sum() > 0:
                loss_cont, loss_details = self.class_loss(cont, semantic_contact)
            else:
                # If no semantic contacts, use binary contacts
                loss_cont, loss_details = self.class_loss(cont, gt_contact_labels_3d.unsqueeze(1))
        else:
            # For original model with binary output
            loss_cont = self.class_loss(cont, gt_contact_labels_3d)
        
        valid_polygon_contact_2d = has_polygon_contact_2d

        # PAL loss calculation remains the same
        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0:
            # For transformer case, we need to collapse the class dimension
            if self.use_transformer:
                # Take max across class dimension to get binary contact
                cont_binary = torch.max(cont, dim=1)[0]
                smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                    'transl': transl[is_smplx == 0],
                                    'has_smpl': has_smpl[is_smplx == 0]}
                loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(
                    cont_binary[is_smplx == 0],
                    smpl_body_params,
                    cam_k[is_smplx == 0],
                    img_scale_factor[is_smplx == 0],
                    polygon_contact_2d[is_smplx == 0],
                    valid_polygon_contact_2d[is_smplx == 0]
                )
            else:
                # Original implementation
                smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                    'transl': transl[is_smplx == 0],
                                    'has_smpl': has_smpl[is_smplx == 0]}
                loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(
                    cont[is_smplx == 0],
                    smpl_body_params,
                    cam_k[is_smplx == 0],
                    img_scale_factor[is_smplx == 0],
                    polygon_contact_2d[is_smplx == 0],
                    valid_polygon_contact_2d[is_smplx == 0]
                )
            
            # weigh the smpl loss based on the number of smpl sample
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        # Combine losses
        if self.context: 
            loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring
        else: 
            loss = self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        # Optimization step
        if self.context:
            self.optimizer_sem.zero_grad()
            self.optimizer_part.zero_grad()
        self.optimizer_contact.zero_grad()

        loss.backward()

        if self.context:
            self.optimizer_sem.step()
            self.optimizer_part.step()
        self.optimizer_contact.step()

        # Prepare loss dictionary
        if self.use_transformer:
            losses = {
                'sem_loss': loss_sem if self.context else 0,
                'part_loss': loss_part if self.context else 0,
                'cont_loss': loss_cont,
                'binary_loss': loss_details['binary_loss'],
                'class_loss': loss_details['class_loss'],
                'dist_loss': loss_details['dist_loss'],
                'pal_loss': loss_pix_anchoring,
                'total_loss': loss
            }
        else:
            if self.context:
                losses = {'sem_loss': loss_sem,
                        'part_loss': loss_part,
                        'cont_loss': loss_cont,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}
            else:
                losses = {'cont_loss': loss_cont,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}         

        # Prepare output dictionary
        if self.context:
            output = {
                'img': img,
                'sem_mask_gt': sem_mask_gt,
                'sem_mask_pred': sem_mask_pred,
                'part_mask_gt': part_mask_gt,
                'part_mask_pred': part_mask_pred,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}
        else:
            output = {
                'img': img,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}   

        return losses, output

    @torch.no_grad()
    def evaluate(self, batch):
        self.model.eval()

        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        # Get binary contact labels
        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        
        # Get semantic contact labels if available
        if 'semantic_contact' in batch:
            semantic_contact = batch['semantic_contact'].to(self.device)
            has_semantic_contact = batch['has_semantic_contact'].to(self.device) if 'has_semantic_contact' in batch else torch.zeros(img.shape[0]).to(self.device)
        else:
            # Create dummy semantic contact with one class
            semantic_contact = gt_contact_labels_3d.unsqueeze(1)
            has_semantic_contact = torch.zeros(img.shape[0]).to(self.device)

        if self.context:
            sem_mask_gt = batch['sem_mask'].to(self.device)
            part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Forward pass
        initial_time = time.time()
        if self.context: 
            cont, sem_mask_pred, part_mask_pred = self.model(img)
        else: 
            cont = self.model(img)
        time_taken = time.time() - initial_time

        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        
        # Use different loss calculation based on whether we're using the transformer
        if self.use_transformer:
            # For transformer model with multi-class output
            # Use semantic contact labels if available, otherwise use binary contacts
            if has_semantic_contact.sum() > 0:
                loss_cont, loss_details = self.class_loss(cont, semantic_contact)
            else:
                # If no semantic contacts, use binary contacts
                loss_cont, loss_details = self.class_loss(cont, gt_contact_labels_3d.unsqueeze(1))
        else:
            # For original model with binary output
            loss_cont = self.class_loss(gt_contact_labels_3d, cont)
        
        valid_polygon_contact_2d = has_polygon_contact_2d

        # PAL loss calculation remains the same
        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0:
            # For transformer case, we need to collapse the class dimension
            if self.use_transformer:
                # Take max across class dimension to get binary contact
                cont_binary = torch.max(cont, dim=1)[0]
                smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                    'transl': transl[is_smplx == 0],
                                    'has_smpl': has_smpl[is_smplx == 0]}
                loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(
                    cont_binary[is_smplx == 0],
                    smpl_body_params,
                    cam_k[is_smplx == 0],
                    img_scale_factor[is_smplx == 0],
                    polygon_contact_2d[is_smplx == 0],
                    valid_polygon_contact_2d[is_smplx == 0]
                )
            else:
                # Original implementation
                smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                    'transl': transl[is_smplx == 0],
                                    'has_smpl': has_smpl[is_smplx == 0]}
                loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(
                    cont[is_smplx == 0],
                    smpl_body_params,
                    cam_k[is_smplx == 0],
                    img_scale_factor[is_smplx == 0],
                    polygon_contact_2d[is_smplx == 0],
                    valid_polygon_contact_2d[is_smplx == 0]
                )
            
            # weigh the smpl loss based on the number of smpl sample
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        # Combine losses
        if self.context: 
            loss = loss_sem + loss_part + self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring
        else: 
            loss = self.loss_weight * loss_cont + self.pal_loss_weight * loss_pix_anchoring

        # Prepare loss dictionary
        if self.use_transformer:
            losses = {
                'sem_loss': loss_sem if self.context else 0,
                'part_loss': loss_part if self.context else 0,
                'cont_loss': loss_cont,
                'binary_loss': loss_details['binary_loss'],
                'class_loss': loss_details['class_loss'],
                'dist_loss': loss_details['dist_loss'],
                'pal_loss': loss_pix_anchoring,
                'total_loss': loss
            }
        else:
            if self.context:
                losses = {'sem_loss': loss_sem,
                        'part_loss': loss_part,
                        'cont_loss': loss_cont,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}
            else:
                losses = {'cont_loss': loss_cont,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}         

        # Prepare output dictionary
        if self.context:
            output = {
                'img': img,
                'sem_mask_gt': sem_mask_gt,
                'sem_mask_pred': sem_mask_pred,
                'part_mask_gt': part_mask_gt,
                'part_mask_pred': part_mask_pred,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}
        else:
            output = {
                'img': img,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont}   

        return losses, output, time_taken

    def save(self, ep, f1, model_path):
        # create model directory if it does not exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if self.context:
            torch.save({
                'epoch': ep,
                'deco': self.model.state_dict(),
                'f1': f1,
                'sem_optim': self.optimizer_sem.state_dict(),
                'part_optim': self.optimizer_part.state_dict(),
                'contact_optim': self.optimizer_contact.state_dict()
            },
                model_path)
        else:
            torch.save({
                'epoch': ep,
                'deco': self.model.state_dict(),
                'f1': f1,
                'sem_optim': self.optimizer_sem.state_dict(),
                'part_optim': self.optimizer_part.state_dict(),
                'contact_optim': self.optimizer_contact.state_dict()
            },
                model_path)    

    def load(self, model_path):
        print(f'~~~ Loading existing checkpoint from {model_path} ~~~')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['deco'], strict=True)

        if self.context:
            self.optimizer_sem.load_state_dict(checkpoint['sem_optim'])
            self.optimizer_part.load_state_dict(checkpoint['part_optim'])
        self.optimizer_contact.load_state_dict(checkpoint['contact_optim'])
        epoch = checkpoint['epoch']
        f1 = checkpoint['f1']
        return epoch, f1

    def update_lr(self, factor=2):
        if factor:
            new_lr = self.lr / factor

        if self.context:
            self.optimizer_sem = torch.optim.Adam(params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()),
                                                lr=new_lr, weight_decay=0.0001)
            self.optimizer_part = torch.optim.Adam(
                params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()), lr=new_lr, weight_decay=0.0001)
        self.optimizer_contact = torch.optim.Adam(
            params=list(self.model.encoder_sem.parameters()) + list(self.model.encoder_part.parameters()) + list(
                self.model.cross_att.parameters()) + list(self.model.classif.parameters()), lr=new_lr, weight_decay=0.0001)

        print('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr
