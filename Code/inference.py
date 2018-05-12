python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_inception_v2_coco.config\
    --trained_checkpoint_prefix training/model.ckpt-200000\
    --output_directory InfraredGraph
