import argparse
import tensorflow as tf
import numpy as np
import pathlib
import cv2
import datetime

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Patches
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def run_inference_for_single_image(model, image):
    #Convert image to numpy array
    image = np.asarray(image)
    #Convert numpy array to tensor
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def visualize_inference(model, image_np):
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=3)
    return image_np



if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--input", required=True)
    arg.add_argument("--output")

    args = vars(arg.parse_args())

    model_dir = './Model/' #ADD MODEL DIR
    model = tf.saved_model.load(str(model_dir))
    # List of the strings that is used to add correct label for each box.

    label_path = './Model/label_map.pbtxt' ##ADD LABEL MAP
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    input_image_dir = pathlib.Path(args["input"])
    input_img_paths = sorted(list(input_image_dir.glob("*.jpg")))

    for image_path in input_img_paths:
        image_path = r"{}".format(image_path)
        img = cv2.imread(image_path)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
        image_np = np.array(img)
        output_img = visualize_inference(model, image_np)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Detection', output_img)
        file_name, _ = pathlib.Path(image_path).name.split(".")
        cv2.imwrite(args["output"] + f'/{file_name}_output.jpg', output_img)
    print("Done")
