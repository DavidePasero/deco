from tqdm import tqdm
from utils.metrics import metric, precision_recall_f1score, det_error_metric
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No X needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PIL.Image as pil_img
from inference import snapshot_matplotlib, create_scene_cpu
import trimesh
import os
from common import constants  # Import constants from the common module

def gen_render(output, normalize=True):
    """
    Generate renders using matplotlib instead of pyrender
    """
    img = output['img']
    cont_pred = output['cont_pred']
    cont_gt = output['cont_gt']
    
    # Get the first image from the batch
    img_np = img[0].detach().cpu().numpy().transpose(1, 2, 0)
    if normalize:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Create meshes with predicted and ground truth contacts
    pred_mesh = create_mesh_with_contacts(cont_pred[0])
    gt_mesh = create_mesh_with_contacts(cont_gt[0])
    
    # Use the matplotlib-based rendering
    pred_render = create_scene_cpu(pred_mesh, img_np)
    gt_render = create_scene_cpu(gt_mesh, img_np)
    
    return {'pred': pred_render, 'gt': gt_render}

def create_mesh_with_contacts(contacts):
    """
    Create a trimesh with contacts highlighted
    """
    # Load the SMPL model
    smpl_path = os.path.join(constants.SMPL_MODEL_DIR, 'smpl_neutral_tpose.ply')
    mesh = trimesh.load(smpl_path, process=False)
    
    # Set default color for the mesh
    for vert in range(mesh.visual.vertex_colors.shape[0]):
        mesh.visual.vertex_colors[vert] = [130, 130, 130, 255]
    
    # Highlight contact vertices
    cont_smpl = []
    contacts_np = contacts.detach().cpu().numpy()
    for idx, val in enumerate(contacts_np):
        if val >= 0.5:
            cont_smpl.append(idx)
    
    mesh.visual.vertex_colors[cont_smpl] = [0, 255, 0, 255]
    
    return mesh


def trainer(epoch, train_loader, solver, hparams, compute_metrics=False):

    total_epochs = hparams.TRAINING.NUM_EPOCHS
    print('Training Epoch {}/{}'.format(epoch, total_epochs))

    length = len(train_loader)
    iterator = tqdm(enumerate(train_loader), total=length, leave=False, desc=f'Training Epoch: {epoch}/{total_epochs}')
    for step, batch in iterator:
        losses, output = solver.optimize(batch)
    return losses, output

@torch.no_grad()
def evaluator(val_loader, solver, hparams, epoch=0, dataset_name='Unknown', normalize=True, return_dict=False):
    total_epochs = hparams.TRAINING.NUM_EPOCHS

    batch_size = val_loader.batch_size
    dataset_size = len(val_loader.dataset)
    print(f'Dataset size: {dataset_size}')

    # Original metrics for binary contact
    val_epoch_cont_pre = np.zeros(dataset_size)
    val_epoch_cont_rec = np.zeros(dataset_size)
    val_epoch_cont_f1 = np.zeros(dataset_size)
    val_epoch_fp_geo_err = np.zeros(dataset_size)
    val_epoch_fn_geo_err = np.zeros(dataset_size)
    
    # New metrics for semantic contact (object-specific)
    if hparams.TRAINING.USE_TRANSFORMER:
        val_epoch_sem_cont_pre = np.zeros(dataset_size)
        val_epoch_sem_cont_rec = np.zeros(dataset_size)
        val_epoch_sem_cont_f1 = np.zeros(dataset_size)
    
    if hparams.TRAINING.CONTEXT:
        val_epoch_sem_iou = np.zeros(dataset_size)
        val_epoch_part_iou = np.zeros(dataset_size)

    val_epoch_cont_loss = np.zeros(dataset_size)
    
    total_time = 0

    rend_images = []

    eval_dict = {}

    length = len(val_loader)
    iterator = tqdm(enumerate(val_loader), total=length, leave=False, desc=f'Evaluating {dataset_name.capitalize()} Epoch: {epoch}/{total_epochs}')
    for step, batch in iterator:
        curr_batch_size = batch['img'].shape[0]
        losses, output, time_taken = solver.evaluate(batch)

        val_epoch_cont_loss[step * batch_size:step * batch_size + curr_batch_size] = losses['cont_loss'].cpu().numpy()

        # compute metrics
        contact_labels_3d = output['contact_labels_3d_gt']
        contact_labels_3d_pred = output['contact_labels_3d_pred']
        
        if hparams.TRAINING.CONTEXT:
            sem_mask_gt = output['sem_mask_gt']
            sem_seg_pred = output['sem_mask_pred']
            part_mask_gt = output['part_mask_gt']
            part_seg_pred = output['part_mask_pred']

        # Evaluate binary contact (any object)
        if hparams.TRAINING.USE_TRANSFORMER:
            # For transformer case, collapse class dimension to get binary contact
            binary_gt = torch.max(contact_labels_3d, dim=1)[0]
            binary_pred = torch.max(contact_labels_3d_pred, dim=1)[0]
            
            # Calculate binary metrics
            cont_pre, cont_rec, cont_f1 = precision_recall_f1score(binary_gt, binary_pred)
            fp_geo_err, fn_geo_err = det_error_metric(binary_pred, binary_gt)
            
            # Calculate semantic contact metrics (object-specific)
            # Flatten batch and class dimensions for evaluation
            flat_gt = contact_labels_3d.reshape(-1, contact_labels_3d.shape[-1])
            flat_pred = contact_labels_3d_pred.reshape(-1, contact_labels_3d_pred.shape[-1])
            
            # Only consider vertices with actual contact in ground truth
            valid_mask = flat_gt.sum(dim=1) > 0
            if valid_mask.sum() > 0:
                sem_cont_pre, sem_cont_rec, sem_cont_f1 = precision_recall_f1score(
                    flat_gt[valid_mask], flat_pred[valid_mask]
                )
            else:
                sem_cont_pre = torch.zeros_like(cont_pre)
                sem_cont_rec = torch.zeros_like(cont_rec)
                sem_cont_f1 = torch.zeros_like(cont_f1)
        else:
            # Original binary metrics calculation
            cont_pre, cont_rec, cont_f1 = precision_recall_f1score(contact_labels_3d, contact_labels_3d_pred)
            fp_geo_err, fn_geo_err = det_error_metric(contact_labels_3d_pred, contact_labels_3d)
        
        if hparams.TRAINING.CONTEXT:
            sem_iou = metric(sem_mask_gt, sem_seg_pred)
            part_iou = metric(part_mask_gt, part_seg_pred)

        # Store metrics
        val_epoch_cont_pre[step * batch_size:step * batch_size + curr_batch_size] = cont_pre.cpu().numpy()
        val_epoch_cont_rec[step * batch_size:step * batch_size + curr_batch_size] = cont_rec.cpu().numpy()
        val_epoch_cont_f1[step * batch_size:step * batch_size + curr_batch_size] = cont_f1.cpu().numpy()
        val_epoch_fp_geo_err[step * batch_size:step * batch_size + curr_batch_size] = fp_geo_err.cpu().numpy()
        val_epoch_fn_geo_err[step * batch_size:step * batch_size + curr_batch_size] = fn_geo_err.cpu().numpy()
        
        if hparams.TRAINING.USE_TRANSFORMER:
            val_epoch_sem_cont_pre[step * batch_size:step * batch_size + curr_batch_size] = sem_cont_pre.cpu().numpy()
            val_epoch_sem_cont_rec[step * batch_size:step * batch_size + curr_batch_size] = sem_cont_rec.cpu().numpy()
            val_epoch_sem_cont_f1[step * batch_size:step * batch_size + curr_batch_size] = sem_cont_f1.cpu().numpy()
        
        if hparams.TRAINING.CONTEXT:
            val_epoch_sem_iou[step * batch_size:step * batch_size + curr_batch_size] = sem_iou.cpu().numpy()
            val_epoch_part_iou[step * batch_size:step * batch_size + curr_batch_size] = part_iou.cpu().numpy()
        
        total_time += time_taken

        # logging every summary_steps steps
        if step % hparams.VALIDATION.SUMMARY_STEPS == 0:
            if hparams.TRAINING.CONTEXT:
                # Prepare the output dictionary for gen_render
                render_output = {
                    'img': output['img'],
                    'cont_pred': binary_pred if hparams.TRAINING.USE_TRANSFORMER else output['contact_labels_3d_pred'],
                    'cont_gt': binary_gt if hparams.TRAINING.USE_TRANSFORMER else output['contact_labels_3d_gt']
                }
                rend = gen_render(render_output, normalize)
                rend_images.append(rend)

    # Compute average metrics
    eval_dict['cont_precision'] = np.sum(val_epoch_cont_pre) / dataset_size
    eval_dict['cont_recall'] = np.sum(val_epoch_cont_rec) / dataset_size
    eval_dict['cont_f1'] = np.sum(val_epoch_cont_f1) / dataset_size
    eval_dict['fp_geo_err'] = np.sum(val_epoch_fp_geo_err) / dataset_size
    eval_dict['fn_geo_err'] = np.sum(val_epoch_fn_geo_err) / dataset_size
    
    if hparams.TRAINING.USE_TRANSFORMER:
        eval_dict['sem_cont_precision'] = np.sum(val_epoch_sem_cont_pre) / dataset_size
        eval_dict['sem_cont_recall'] = np.sum(val_epoch_sem_cont_rec) / dataset_size
        eval_dict['sem_cont_f1'] = np.sum(val_epoch_sem_cont_f1) / dataset_size
    
    if hparams.TRAINING.CONTEXT:
        eval_dict['sem_iou'] = np.sum(val_epoch_sem_iou) / dataset_size
        eval_dict['part_iou'] = np.sum(val_epoch_part_iou) / dataset_size
        eval_dict['images'] = rend_images
    
    total_time /= dataset_size

    val_epoch_cont_loss = np.sum(val_epoch_cont_loss) / dataset_size
    if return_dict:
        return eval_dict, total_time
    return eval_dict['cont_f1']
