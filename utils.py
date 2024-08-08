import os
import cv2
from PIL import Image

def process_yolo_predictions(predictions, input_image_path, output_dir='processed_images', target_size=(224, 224)):
    """
    Process YOLO predictions, crop and resize the target person bounding boxes, and save them in a specified directory.

    :param predictions: List of YOLO predictions.
    :param input_image_path: Path to the input image.
    :param output_dir: Directory to save the processed images.
    :param target_size: Desired size (width, height) of the cropped and resized images.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the input image
    image = Image.open(input_image_path)

    for i, pred in enumerate(predictions):
        # Extract the bounding box coordinates
        x1, y1, x2, y2, conf, class_id = pred[:6]

        # Crop the image based on the bounding box
        cropped_image = image.crop((x1, y1, x2, y2))

        # Resize the cropped image to the target size
        resized_image = cropped_image.resize(target_size, Image.ANTIALIAS)

        # Save the processed image
        output_path = os.path.join(output_dir, f'processed_image_{i}.jpg')
        resized_image.save(output_path)
        print(f'Saved processed image: {output_path}')

# Example usage
if __name__ == "__main__":
    # Assuming predictions are in the format [x1, y1, x2, y2, confidence, class_id]
    predictions = [
        [50, 50, 200, 200, 0.9, 0],  # Example bounding box
        [150, 150, 300, 300, 0.85, 0]
    ]
    input_image_path = 'path/to/input/image.jpg'
    process_yolo_predictions(predictions, input_image_path)
