from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

BATCH_SIZE = 32
WORKERS = 8
# Initialize the Trainer
trainer = Trainer(experiment_name="yolonas_evaluation_comparison")

# Define dataset parameters
dataset_params = {
    "data_dir": 'C:/Users/User/CAM-R/models/object-detection/yolo_data',
    "images_dir": 'test/images',  # Directory containing validation images
    "labels_dir": 'test/labels',  # Directory containing YOLO format labels
    "classes": ['Bus', 'Truck', 'Motorcycle', 'Car'],  # List of class names
}

metrics=DetectionMetrics_050(score_thres=0.1, 
                                        top_k_predictions=300, 
                                        num_cls=len(dataset_params['classes']), 
                                        normalize_targets=True, 
                                        post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                nms_top_k=1000, 
                                                                                                max_predictions=300,                                                                              
                                                                                                nms_threshold=0.7)
                                        )

# Use the appropriate data format (YOLO or COCO)
data =  coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['images_dir'],
        'labels_dir': dataset_params['labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)

if __name__ == '__main__':
    # # Load and evaluate the first set of weights
    # model_1 = models.get('yolo_nas_l', pretrained_weights="coco")
    # results_1 = trainer.test(model=model_1, test_loader=data, test_metrics_list = metrics)
    # print("Results for first set of weights:", results_1)

    # Load and evaluate the second set of weights
    model_2 = models.get('yolo_nas_s', 
                         num_classes=len(dataset_params['classes']),
                         checkpoint_path="C:/Users/User/CAM-R/checkpoints/yolo_nas_s/RUN_20240902_210546_747020/ckpt_best.pth")
    results_2 = trainer.test(model=model_2, test_loader=data, test_metrics_list = metrics)
    print("Results for second set of weights:", results_2)
