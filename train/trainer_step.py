from utils.loss import sem_loss_function, class_loss_function, pixel_anchoring_function, MultiClassContactLoss
import torch
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainStepper():
    def __init__(self, deco_model, context, learning_rate, loss_weight, pal_loss_weight, device, log_dir="runs",
                 run_name: str = ""):
        self.device = device
        self.global_step = 0
        self.model = deco_model
        self.context = context

        self._setup_optims(deco_model, learning_rate)

        if self.context: self.sem_loss = sem_loss_function().to(device)
        self.class_loss = class_loss_function().to(device)
        self.pixel_anchoring_loss_smplx = pixel_anchoring_function(model_type='smplx').to(device)
        self.pixel_anchoring_loss_smpl = pixel_anchoring_function(model_type='smpl').to(device)
        self.lr = learning_rate
        self.loss_weight = loss_weight
        self.pal_loss_weight = pal_loss_weight
        self.semantic_contact_loss = MultiClassContactLoss(pos_weight=6.451).to(device)
        self.semantic_loss_weight = 0.1  # Weight for semantic loss (between contact and pixel anchoring)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"train_{timestamp}_{run_name}"))

    def _log(self, mode, losses, step):
        for key, value in losses.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                value = value.item() if isinstance(value, torch.Tensor) else value
                self.writer.add_scalar(f"{mode}/{key}", value, step)

    def _setup_optims(self, deco_model: torch.nn.Module, learning_rate: float):

        if deco_model.__class__.__name__ == 'DINOContact':  # Check if we are using simply DINOContact
            self.optimizer_contact = torch.optim.Adam(
                params=list(self.model.classifier.parameters()) +
                       (list(self.model.encoder.parameters()) if self.model.train_backbone else []),
                lr=learning_rate,
                weight_decay=0.0001)
        elif deco_model.__class__.__name__ == 'DECO':
            if self.context:
                if "dinov2" in self.model.encoder_type and self.model.num_encoder == 1:
                    self.optimizer_sem = torch.optim.Adam(
                        params=list(self.model.scene_projector.parameters()) +
                               list(self.model.decoder_sem.parameters()) +
                               (list(self.model.encoder.parameters()) if self.model.train_backbone else []),
                        lr=learning_rate, weight_decay=0.0001)
                    self.optimizer_part = torch.optim.Adam(
                        params=list(self.model.decoder_part.parameters()) +
                               list(self.model.contact_projector.parameters()) +
                               (list(self.model.encoder.parameters()) if self.model.train_backbone else []),
                        lr=learning_rate,
                        weight_decay=0.0001)
                else:
                    self.optimizer_sem = torch.optim.Adam(
                        params=list(self.model.encoder_sem.parameters()) + list(self.model.decoder_sem.parameters()) +
                               (list(self.model.correction_conv.parameters()) if hasattr(self.model, "correction_conv") else []),
                        lr=learning_rate, weight_decay=0.0001)
                    self.optimizer_part = torch.optim.Adam(
                        params=list(self.model.encoder_part.parameters()) + list(self.model.decoder_part.parameters()) +
                        (list(self.model.correction_conv.parameters()) if hasattr(self.model, "correction_conv") else []),
                        lr=learning_rate,
                        weight_decay=0.0001)

            if "dinov2" in self.model.encoder_type:  # Check if DECO has DINO encoder, if so we need to use the feature projectors in the optimizer
                self.optimizer_contact = torch.optim.Adam(
                    params=list(self.model.scene_projector.parameters()) + list(
                        self.model.contact_projector.parameters()) + list(
                        self.model.cross_att.parameters()) + list(self.model.classif.parameters()) + (list(
                        self.model.encoder.parameters()) if self.model.train_backbone else []) +
                        (list(self.model.correction_conv.parameters()) if hasattr(self.model, "correction_conv") else []),
                    lr=learning_rate,
                    weight_decay=0.0001)
            else:
                self.optimizer_contact = torch.optim.Adam(
                    params=list(self.model.encoder_sem.parameters()) + list(
                        self.model.encoder_part.parameters()) + list(
                        self.model.cross_att.parameters()) + list(self.model.classif.parameters())
                    + (list(self.model.correction_conv.parameters()) if hasattr(self.model, "correction_conv") else []),
                    lr=learning_rate,
                    weight_decay=0.0001)

        else:
            raise NotImplementedError(f"The model {deco_model.__class__.__name__} is not supported")

    def optimize(self, batch):
        self.model.train()
        losses = {}
        img_paths = batch['img_path']
        img = batch['img'].to(self.device)

        img_scale_factor = batch['img_scale_factor'].to(self.device)

        pose = batch['pose'].to(self.device)
        betas = batch['betas'].to(self.device)
        transl = batch['transl'].to(self.device)
        has_smpl = batch['has_smpl'].to(self.device)
        is_smplx = batch['is_smplx'].to(self.device)

        cam_k = batch['cam_k'].to(self.device)

        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        has_contact_3d = batch['has_contact_3d'].to(self.device)

        if self.context:
            sem_mask_gt = batch['sem_mask'].to(self.device)
            part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Add semantic contact labels
        semantic_contact_labels = batch['semantic_contact'].to(self.device)
        has_semantic_contact = batch['has_semantic_contact'].to(self.device)

        # Forward pass
        if self.context:
            cont, sem_mask_pred, part_mask_pred, semantic_logits = self.model(img)
        else:
            cont, semantic_logits = self.model(img)

        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        valid_polygon_contact_2d = has_polygon_contact_2d

        # CONTACT LOSS CALCULATION
        total_cont_loss, loss_cont, loss_semantic, loss_dist = self.semantic_contact_loss(cont, semantic_logits, semantic_contact_labels)


        if self.pal_loss_weight > 0 and (is_smplx == 0).sum() > 0:
            smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                'transl': transl[is_smplx == 0],
                                'has_smpl': has_smpl[is_smplx == 0]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[is_smplx == 0],
                                                                                                  smpl_body_params,
                                                                                                  cam_k[is_smplx == 0],
                                                                                                  img_scale_factor[
                                                                                                      is_smplx == 0],
                                                                                                  polygon_contact_2d[
                                                                                                      is_smplx == 0],
                                                                                                  valid_polygon_contact_2d[
                                                                                                      is_smplx == 0])
            # weigh the smpl loss based on the number of smpl sample
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)
        
        if self.context:
            loss_sem_dict = {
                'sem_loss': loss_sem.item(),
                'part_loss': loss_part.item(),
            }
            loss = loss_sem + loss_part + self.loss_weight * total_cont_loss + self.pal_loss_weight * loss_pix_anchoring
        else:
            loss_sem_dict = {}
            loss = self.loss_weight * total_cont_loss + self.pal_loss_weight * loss_pix_anchoring
        
        losses = {
            **{
            'cont_loss': loss_cont.item(),
            'semantic_loss': loss_semantic.item(),
            'dist_loss': loss_dist.item(),
            'pal_loss': loss_pix_anchoring.item(),
            'total_loss': loss.item()},
            **loss_sem_dict
        }
        self._log("train", losses, self.global_step)
        self.global_step += 1
        
        if self.context:
            self.optimizer_sem.zero_grad()
            self.optimizer_part.zero_grad()
        self.optimizer_contact.zero_grad()

        loss.backward()

        if self.context:
            self.optimizer_sem.step()
            self.optimizer_part.step()
        self.optimizer_contact.step()

        if self.context:
            losses = {  'sem_loss': loss_sem,
                        'part_loss': loss_part,
                        'cont_loss': loss_cont,
                        'semantic_loss': loss_semantic,
                        'dist_loss': loss_dist,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}
        else:
            losses = {  'cont_loss': loss_cont,
                        'semantic_loss': loss_semantic,
                        'dist_loss': loss_dist,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}

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
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont,
                'semantic_contact_pred': semantic_logits,
                'semantic_contact_gt': semantic_contact_labels}
        else:
            output = {
                'img': img,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont,
                'semantic_contact_pred': semantic_logits,
                'semantic_contact_gt': semantic_contact_labels}

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

        gt_contact_labels_3d = batch['contact_label_3d'].to(self.device)
        has_contact_3d = batch['has_contact_3d'].to(self.device)

        if self.context:
            sem_mask_gt = batch['sem_mask'].to(self.device)
            part_mask_gt = batch['part_mask'].to(self.device)

        polygon_contact_2d = batch['polygon_contact_2d'].to(self.device)
        has_polygon_contact_2d = batch['has_polygon_contact_2d'].to(self.device)

        # Add semantic contact labels
        semantic_contact_labels = batch['semantic_contact'].to(self.device)
        has_semantic_contact = batch['has_semantic_contact'].to(self.device)

        # Forward pass
        initial_time = time.time()
        if self.context:
            cont, sem_mask_pred, part_mask_pred, semantic_logits = self.model(img)
        else:
            cont, semantic_logits = self.model(img)
        time_taken = time.time() - initial_time

        if self.context:
            loss_sem = self.sem_loss(sem_mask_gt, sem_mask_pred)
            loss_part = self.sem_loss(part_mask_gt, part_mask_pred)
        valid_contact_3d = has_contact_3d
        loss_cont = self.class_loss(gt_contact_labels_3d, cont, valid_contact_3d)
        valid_polygon_contact_2d = has_polygon_contact_2d

        # CONTACT LOSS CALCULATION
        total_cont_loss, loss_cont, loss_semantic, loss_dist = self.semantic_contact_loss(cont, semantic_logits, semantic_contact_labels)

        if self.pal_loss_weight > 0 and (
                is_smplx == 0).sum() > 0:  # PAL loss only on 2D contacts in HOT which only has SMPL
            smpl_body_params = {'pose': pose[is_smplx == 0], 'betas': betas[is_smplx == 0],
                                'transl': transl[is_smplx == 0],
                                'has_smpl': has_smpl[is_smplx == 0]}
            loss_pix_anchoring_smpl, contact_2d_pred_rgb_smpl, _ = self.pixel_anchoring_loss_smpl(cont[is_smplx == 0],
                                                                                                  smpl_body_params,
                                                                                                  cam_k[is_smplx == 0],
                                                                                                  img_scale_factor[
                                                                                                      is_smplx == 0],
                                                                                                  polygon_contact_2d[
                                                                                                      is_smplx == 0],
                                                                                                  valid_polygon_contact_2d[
                                                                                                      is_smplx == 0])
            # weight the smpl loss based on the number of smpl samples
            contact_2d_pred_rgb = contact_2d_pred_rgb_smpl
            loss_pix_anchoring = loss_pix_anchoring_smpl * (is_smplx == 0).sum() / len(is_smplx)
        else:
            loss_pix_anchoring = 0
            contact_2d_pred_rgb = torch.zeros_like(polygon_contact_2d)

        if self.context:
            loss = loss_sem + loss_part + self.loss_weight * total_cont_loss + self.pal_loss_weight * loss_pix_anchoring
        else:
            loss = self.loss_weight * total_cont_loss + self.pal_loss_weight * loss_pix_anchoring

        if self.context:
            losses = {  'sem_loss': loss_sem,
                        'part_loss': loss_part,
                        'cont_loss': loss_cont,
                        'semantic_loss': loss_semantic,
                        'dist_loss': loss_dist,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}
        else:
            losses = {  'cont_loss': loss_cont,
                        'semantic_loss': loss_semantic,
                        'dist_loss': loss_dist,
                        'pal_loss': loss_pix_anchoring,
                        'total_loss': loss}
        self._log("eval", losses, self.global_step)
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
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont,
                'semantic_contact_pred': semantic_logits,
                'semantic_contact_gt': semantic_contact_labels}
        else:
            output = {
                'img': img,
                'has_contact_2d': has_polygon_contact_2d,
                'contact_2d_gt': polygon_contact_2d,
                'contact_2d_pred_rgb': contact_2d_pred_rgb,
                'has_contact_3d': has_contact_3d,
                'contact_labels_3d_gt': gt_contact_labels_3d,
                'contact_labels_3d_pred': cont,
                'semantic_contact_pred': semantic_logits,
                'semantic_contact_gt': semantic_contact_labels}

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
                'deco' if self.model.__class__.__name__ == 'DECO' else "dinoContact": self.model.state_dict(),
                'f1': f1,
                'sem_optim': self.optimizer_sem.state_dict() if hasattr(self, "optimizer_sem") else None,
                'part_optim': self.optimizer_part.state_dict() if hasattr(self, "part_optim") else None,
                'contact_optim': self.optimizer_contact.state_dict() if hasattr(self, "contact_optim") else None
            },
                model_path)

    def load(self, model_path):
        print(f'~~~ Loading existing checkpoint from {model_path} ~~~')
        checkpoint = torch.load(model_path, weights_only=False)
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

        self._setup_optims(self.model, new_lr)

        print('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr