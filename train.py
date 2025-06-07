import torch
from torch.utils.data import DataLoader
import os
import warnings

from train.trainer_step import TrainStepper
from train.base_trainer import trainer, evaluator
from data.base_dataset import BaseDataset, VLMFeatureCollator
from data.mixed_dataset import MixedDataset
from models.deco import DECO, DINOContact
from utils.config import parse_args, run_grid_search_experiments


def _create_run_name(hparams):
    """
    Generates a unique run name for the experiment based on the model type, encoder, and current timestamp.
    """
    # Extract model type and encoder
    model_type = hparams.TRAINING.MODEL_TYPE
    encoder = hparams.TRAINING.ENCODER

    run_name = f"{model_type}_{encoder}"

    if "dinov2" in encoder:
        run_name += f"_train_backbone={hparams.TRAINING.TRAIN_BACKBONE}"
        run_name += f"_num_encoders={hparams.TRAINING.NUM_ENCODER}"

    run_name += f"use_vlm{hparams.TRAINING.USE_VLM}"
    run_name += f"train_vlm_text_encoder{hparams.TRAINING.TRAIN_VLM_TEXT_ENCODER}"

    return run_name


def train(hparams):
    if hparams.TRAINING.TRAIN_BACKBONE and "dino" in hparams.TRAINING.ENCODER:
        warnings.warn("Backbone will be trained. Make sure this behavior is wanted!")
    if hparams.TRAINING.MODEL_TYPE == 'deco':
        deco_model = DECO(
            encoder=hparams.TRAINING.ENCODER,
            context=hparams.TRAINING.CONTEXT,
            device=device,
            num_encoders=hparams.TRAINING.NUM_ENCODER,
            classifier_type=hparams.TRAINING.CLASSIFIER_TYPE,
            train_backbone=hparams.TRAINING.TRAIN_BACKBONE,
            train_vlm_text_encoder=hparams.TRAINING.TRAIN_VLM_TEXT_ENCODER,
            use_vlm=hparams.TRAINING.USE_VLM,
            patch_cross_attention=hparams.TRAINING.PATCH_CROSS_ATTENTION
            ) # set up DinoContact here
    elif hparams.TRAINING.MODEL_TYPE == 'dinoContact':
        deco_model = DINOContact(encoder_name=hparams.TRAINING.ENCODER,
                                 classifier_type=hparams.TRAINING.CLASSIFIER_TYPE,
                                 train_backbone=hparams.TRAINING.TRAIN_BACKBONE,
                                 train_vlm_text_encoder=hparams.TRAINING.TRAIN_VLM_TEXT_ENCODER,
                                 use_vlm=hparams.TRAINING.USE_VLM,
                                 )
    else:
        raise ValueError('Model type not supported')

    if isinstance(deco_model, DINOContact):
        hparams.TRAINING.CONTEXT = False

    solver = TrainStepper(deco_model, hparams.TRAINING.CONTEXT, hparams.OPTIMIZER.LR, hparams.TRAINING.LOSS_WEIGHTS,
                          hparams.TRAINING.PAL_LOSS_WEIGHTS, device, use_semantic_class_balanced_loss = hparams.TRAINING.USE_SEMANTIC_CLASS_BALANCED_LOSS, run_name=_create_run_name(hparams))

    vb_f1 = 0
    start_ep = 0
    num = 0
    k = True
    latest_model_path = hparams.TRAINING.BEST_MODEL_PATH.replace('best', 'latest')
    if os.path.exists(latest_model_path):
        _, vb_f1 = solver.load(hparams.TRAINING.BEST_MODEL_PATH)
        start_ep, _ = solver.load(latest_model_path)

    for epoch in range(start_ep + 1, hparams.TRAINING.NUM_EPOCHS + 1):
        # Train one epoch
        trainer(epoch, train_loader, solver, hparams)
        # Run evaluation
        vc_f1 = None
        for val_loader in val_loaders:
            dataset_name = val_loader.dataset.dataset
            vc_f1_dict, _ = evaluator(val_loader, solver, hparams, epoch, dataset_name,
                                      normalize=hparams.DATASET.NORMALIZE_IMAGES,
                                      return_dict=True)
            solver._log("epoch-end-eval", vc_f1_dict, epoch)
            vc_f1_ds = vc_f1_dict["cont_f1"]
            if dataset_name == hparams.VALIDATION.MAIN_DATASET:
                vc_f1 = vc_f1_ds
        if vc_f1 is None:
            raise ValueError('Main dataset not found in validation datasets')

        print('Learning rate: ', solver.lr)

        print('---------------------------------------------')
        print('---------------------------------------------')

        solver.save(epoch, vc_f1, latest_model_path)

        if epoch % hparams.TRAINING.CHECKPOINT_EPOCHS == 0:
            inter_model_path = latest_model_path.replace('latest', 'epoch_' + str(epoch).zfill(3))
            solver.save(epoch, vc_f1, inter_model_path)

        if vc_f1 < vb_f1:
            num += 1
            print('Not Saving model: Best Val F1 = ', vb_f1, ' Current Val F1 = ', vc_f1)
        else:
            num = 0
            vb_f1 = vc_f1
            print('Saving model...')
            solver.save(epoch, vb_f1, hparams.TRAINING.BEST_MODEL_PATH)

        if num >= hparams.OPTIMIZER.NUM_UPDATE_LR: solver.update_lr()
        if num >= hparams.TRAINING.NUM_EARLY_STOP:
            print('Early Stop')
            k = False

        if k:
            continue
        else:
            break


if __name__ == '__main__':
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script='train.py',
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataset = MixedDataset(hparams.TRAINING.DATASETS, 'train', dataset_mix_pdf=hparams.TRAINING.DATASET_MIX_PDF,
                                 normalize=hparams.DATASET.NORMALIZE_IMAGES, use_vlm=hparams.TRAINING.USE_VLM,
                                 transforms=hparams.DATASET.AUGMENTATION)

    val_datasets = []
    for ds in hparams.VALIDATION.DATASETS:
        if ds in ['rich', 'prox']:
            val_datasets.append(BaseDataset(ds, 'val', model_type='smpl', normalize=hparams.DATASET.NORMALIZE_IMAGES,
                                use_vlm=hparams.TRAINING.USE_VLM)),
        elif ds in ['damon']:
            val_datasets.append(BaseDataset(ds, 'val', model_type='smpl', normalize=hparams.DATASET.NORMALIZE_IMAGES,
                                use_vlm=hparams.TRAINING.USE_VLM))
        else:
            raise ValueError('Dataset not supported')

    train_loader = DataLoader(train_dataset, hparams.DATASET.BATCH_SIZE, shuffle=True,
                              num_workers=hparams.DATASET.NUM_WORKERS,)
    val_loaders = [DataLoader(val_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=False,
                              num_workers=hparams.DATASET.NUM_WORKERS,) for val_dataset in val_datasets]

    train(hparams)