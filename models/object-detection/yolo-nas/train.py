# !pip install super-gradients
#imports
import torch
import json
import time
import numpy as np
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm.auto import tqdm


#load model 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ARCH = 'yolo_nas_s'
#            'yolo_nas_l'
#            'yolo_nas_m'

#parameters 
EPOCHS = 25
BATCH_SIZE = 16
WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#directories
CHECKPOINT_DIR = 'checkpoints3orientation'
ROOT_DIR = './datasets/yolo_data2/yolo_data2'
train_imgs_dir = 'train/images'
train_labels_dir = 'train/labels'
val_imgs_dir = 'valid/images'
val_labels_dir = 'valid/labels'
test_imgs_dir = 'test/images'
test_labels_dir = 'test/labels'
# classes = ['Bus', 'Truck', 'Motorcycle', 'Car']
classes = ['car_back', 'car_side', 'car_front', 'bus_back', 'bus_side', 'bus_front', 'truck_back', 'truck_side', 'truck_front', 'motorcycle_back',
           'motorcycle_side', 'motorcycle_front', 'bicycle_back', 'bicycle_side', 'bicycle_front']

dataset_params = {
    'data_dir':ROOT_DIR,
    'train_images_dir':train_imgs_dir,
    'train_labels_dir':train_labels_dir,
    'val_images_dir':val_imgs_dir,
    'val_labels_dir':val_labels_dir,
    'test_images_dir':test_imgs_dir,
    'test_labels_dir':test_labels_dir,
    'classes':classes 
}

#establish the datasets
train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS,
        'pin_memory': True,
        'drop_last': True
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS,
        'pin_memory': True,
        'drop_last': True
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS,
        'pin_memory': True,
        'drop_last': True
    }
)

#test metrics
metrics=DetectionMetrics_050(score_thres=0.1, 
                            top_k_predictions=300, 
                            num_cls=len(dataset_params['classes']), 
                            normalize_targets=True, 
                            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                    nms_top_k=1000, 
                                                                                    max_predictions=300,                                                                              
                                                                                    nms_threshold=0.7)
                            )
metrics_50_95 = DetectionMetrics_050_095(
    score_thres=0.1, 
    top_k_predictions=300, 
    num_cls=len(dataset_params['classes']), 
    normalize_targets=True, 
    post_prediction_callback=PPYoloEPostPredictionCallback(
        score_threshold=0.01, 
        nms_top_k=1000, 
        max_predictions=300,                                                                              
        nms_threshold=0.7
    )
)
#modify to change training things
train_params = {
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 3e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.4,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.4,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.4,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.4,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50:0.95'
}

def convert_dict_values_to_float(d):
    for key, value in d.items():
        if isinstance(value, dict):
            convert_dict_values_to_float(value)
        elif isinstance(value, np.float32):
            d[key] = float(value)
    return d
    
#code to train
if __name__ == "__main__":
    if DEVICE =='cuda':
        torch.cuda.empty_cache()
        
    trainer = Trainer(
        experiment_name=MODEL_ARCH, 
        ckpt_root_dir=CHECKPOINT_DIR
    )

    model = models.get(
        MODEL_ARCH, 
        num_classes=len(dataset_params['classes']), 
        pretrained_weights="coco"
    )
    
    trainer.train(
        model=model, 
        training_params=train_params, 
        train_loader=train_data, 
        valid_loader=val_data
    )
    
    if DEVICE =='cuda':
        torch.cuda.empty_cache()
        
    start_time = time.time()
    results_2 = trainer.test(model=model, test_loader=test_data, test_metrics_list = [metrics, metrics_50_95])
    end_time = time.time()

    # Calculate the elapsed time
    results_2 = convert_dict_values_to_float(results_2)
    inference_time = end_time - start_time
    print("Results for weights:", results_2)
    print(f"Inference time: {inference_time:.4f} seconds")
    with open("test2_results.txt", "a") as f:
        f.write("adam_3e-4: ")
        f.write(json.dumps(results_2, indent=4)) 
        f.write(f"\nInference time: {inference_time:.4f} seconds\n")
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()