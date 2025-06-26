Working fine(just light blur)
import boto3
import os
import tempfile
from PIL import Image, ImageFilter
import cv2
import numpy as np

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')

def lambda_handler(event, context):
    bucket = 'test-sample-reva-rv'
    
    # Get image file name from S3 event
    key = event['Records'][0]['s3']['object']['key']
    input_path = os.path.join(tempfile.gettempdir(), 'input.jpg')
    output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
    
    # Download image from input-images folder
    s3.download_file(bucket, key, input_path)

    # Call Rekognition to detect rash (assumed as 'Custom Labels')
    response = rekognition.detect_custom_labels(
        ProjectVersionArn='your-project-version-arn',  # Replace with actual ARN
        Image={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    
    # Load image using PIL
    image = Image.open(input_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = image.size

    mask = np.zeros((img_cv.shape[0], img_cv.shape[1]), dtype=np.uint8)

    for label in response['CustomLabels']:
        if label['Confidence'] >= 90 and 'Geometry' in label:
            box = label['Geometry']['BoundingBox']
            left = int(box['Left'] * img_cv.shape[1])
            top = int(box['Top'] * img_cv.shape[0])
            box_width = int(box['Width'] * img_cv.shape[1])
            box_height = int(box['Height'] * img_cv.shape[0])
            
            # Create white rectangle mask for rash region
            cv2.rectangle(mask, (left, top), (left + box_width, top + box_height), 255, -1)

    # Blur entire image
    blurred = cv2.GaussianBlur(img_cv, (21, 21), 0)
    
    # Use mask to blend rash region from original image
    focused = np.where(mask[:, :, None] == 255, img_cv, blurred)

    # Save output
    cv2.imwrite(output_path, focused)

    # Upload to output-images folder
    output_key = key.replace('input-images/', 'output-images/')
    s3.upload_file(output_path, bucket, output_key)

    return {
        'statusCode': 200,
        'body': f'Processed and uploaded to {output_key}'
    }

updated (for intense blur)


import cv2
import numpy as np
from PIL import Image

# Load image using PIL
image = Image.open(input_path).convert("RGB")
img_cv = np.array(image)[:, :, ::-1]  # RGB to BGR for OpenCV

# Create a mask and apply blur only to that area
mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)

# Assuming Rekognition returns bounding boxes
for label in response['CustomLabels']:
    if 'Geometry' in label:
        box = label['Geometry']['BoundingBox']
        height, width = img_cv.shape[:2]

        # Convert relative box to absolute pixel coords
        top = int(box['Top'] * height)
        left = int(box['Left'] * width)
        box_height = int(box['Height'] * height)
        box_width = int(box['Width'] * width)

        # Fill mask with white in the bounding box region
        mask[top:top + box_height, left:left + box_width] = 255

# Strong blur: increase kernel size for heavy blur
blurred = cv2.GaussianBlur(img_cv, (55, 55), 0)

# Blend original + blurred using mask
result = np.where(mask[:, :, None] == 255, blurred, img_cv)

# Save result
cv2.imwrite(output_path, result)

