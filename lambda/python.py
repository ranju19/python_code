file_bytes = np.fromfile(input_path, dtype=np.uint8)
img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)




for label in response['CustomLabels']:
    if label['Confidence'] > 60 and 'Geometry' in label:
        box = label['Geometry']['BoundingBox']
        
        top = int(box['Top'] * height)
        left = int(box['Left'] * width)
        box_height = int(box['Height'] * height)
        box_width = int(box['Width'] * width)

        # Ensure the coordinates stay within image bounds
        x1 = max(0, left)
        y1 = max(0, top)
        x2 = min(width, left + box_width)
        y2 = min(height, top + box_height)

        # Final draw - with all integer safe values
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)






BLACK OUT BLUR
import boto3
import json
import os
import tempfile
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')

def lambda_handler(event, context):
    # Get image file name from S3 event
    bucket = 'test-sample-reva-rv'
    key = event['Records'][0]['s3']['object']['key']
    
    # Paths for input and output
    input_path = os.path.join(tempfile.gettempdir(), 'input.jpg')
    output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
    
    # Download image from S3
    s3.download_file(bucket, key, input_path)
    
    # Call Rekognition to detect custom labels (eyes, nose, mouth, etc.)
    response = rekognition.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:161289594039:project/sample-test-rv/version/sample-test-rv.2025-07-10T13.46.54/1752169614968',
        Image={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    
    # Load image using PIL and convert to OpenCV BGR format
    image = Image.open(input_path).convert('RGB')
    img_cv = np.array(image)[..., ::-1]  # Convert to BGR
    height, width = img_cv.shape[:2]
    
    # Loop through detected regions and blackout those areas
    for label in response['CustomLabels']:
        if label['Confidence'] > 60 and 'Geometry' in label:
            box = label['Geometry']['BoundingBox']
            
            top = int(box['Top'] * height)
            left = int(box['Left'] * width)
            box_height = int(box['Height'] * height)
            box_width = int(box['Width'] * width)
            
            # Draw a filled black rectangle (blackout)
            cv2.rectangle(img_cv, 
                          (left, top), 
                          (left + box_width, top + box_height), 
                          (0, 0, 0), 
                          thickness=-1)

    # Save the final result
    cv2.imwrite(output_path, img_cv)

    # Upload to output-images/ folder
    output_key = key.replace('input-images/', 'output-images/')
    s3.upload_file(output_path, bucket, output_key)

    return {
        'statusCode': 200,
        'body': f'Processed and uploaded to {output_key}'
    }

-----------------------------------------------------------------------------------------------
import boto3
import json
import os
import tempfile
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')

def lambda_handler(event, context):
    # Get image file name from S3 event
    bucket = 'test-sample-reva-rv'
    key = event['Records'][0]['s3']['object']['key']
    
    # Paths for input and output
    input_path = os.path.join(tempfile.gettempdir(), 'input.jpg')
    output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
    
    # Download image from S3
    s3.download_file(bucket, key, input_path)
    
    # Call Rekognition to detect rash
    response = rekognition.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:161289594039:project/sample-test-rv/version/sample-test-rv.2025-07-10T13.46.54/1752169614968',
        Image={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    
    # Load image using PIL and convert to OpenCV BGR format
    image = Image.open(input_path).convert('RGB')
    img_cv = np.array(image)[..., ::-1]  # Convert to BGR
    height, width = img_cv.shape[:2]
    
    # Blur the entire image
    blurred = cv2.GaussianBlur(img_cv, (99, 99), 30)

    # Loop through detected rash regions and paste back the original rash area
    for label in response['CustomLabels']:
        if label['Confidence'] > 60 and 'Geometry' in label:
            box = label['Geometry']['BoundingBox']
            
            top = int(box['Top'] * height)
            left = int(box['Left'] * width)
            box_height = int(box['Height'] * height)
            box_width = int(box['Width'] * width)
            
            # Paste original rash region onto blurred image
            blurred[top:top + box_height, left:left + box_width] = img_cv[top:top + box_height, left:left + box_width]

    # Save the final result
    cv2.imwrite(output_path, blurred)

    # Upload to output-images/ folder
    output_key = key.replace('input-images/', 'output-images/')
    s3.upload_file(output_path, bucket, output_key)

    return {
        'statusCode': 200,
        'body': f'Processed and uploaded to {output_key}'
    }
----------------------------------------------------------------------------------------------
Fully working code with intense blur(Final)

import boto3
import os
import tempfile
from PIL import Image
import cv2
import numpy as np

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')

def lambda_handler(event, context):
    # Get image file name from S3 event
    bucket = 'test-sample-reva-rv'
    key = event['Records'][0]['s3']['object']['key']
    
    # Paths for input and output
    input_path = os.path.join(tempfile.gettempdir(), 'input.jpg')
    output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
    
    # Download image from S3
    s3.download_file(bucket, key, input_path)
    
    # Call Rekognition to detect rash
    response = rekognition.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:161289594039:project/sample-test-rv/version/sample-test-rv.2025-06-23T01.41.24/1750657284227',
        Image={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    
    # Load image using PIL and convert to OpenCV BGR format
    image = Image.open(input_path).convert('RGB')
    img_cv = np.array(image)[..., ::-1]  # Convert to BGR
    height, width = img_cv.shape[:2]
    
    # Blur the entire image
    blurred = cv2.GaussianBlur(img_cv, (55, 55), 0)

    # Loop through detected rash regions and paste back the original rash area
    for label in response['CustomLabels']:
        if label['Confidence'] > 90 and 'Geometry' in label:
            box = label['Geometry']['BoundingBox']
            
            top = int(box['Top'] * height)
            left = int(box['Left'] * width)
            box_height = int(box['Height'] * height)
            box_width = int(box['Width'] * width)
            
            # Paste original rash region onto blurred image
            blurred[top:top + box_height, left:left + box_width] = img_cv[top:top + box_height, left:left + box_width]

    # Save the final result
    cv2.imwrite(output_path, blurred)

    # Upload to output-images/ folder
    output_key = key.replace('input-images/', 'output-images/')
    s3.upload_file(output_path, bucket, output_key)

    return {
        'statusCode': 200,
        'body': f'Processed and uploaded to {output_key}'
    }

























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


draft 2 (for intense blur)

import boto3
import os
import tempfile
from PIL import Image
import cv2
import numpy as np

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')

def lambda_handler(event, context):
    # Get image file name from S3 event
    bucket = 'test-sample-reva'
    key = event['Records'][0]['s3']['object']['key']
    
    # Paths for input and output
    input_path = os.path.join(tempfile.gettempdir(), 'input.jpg')
    output_path = os.path.join(tempfile.gettempdir(), 'output.jpg')
    
    # Download image from S3
    s3.download_file(bucket, key, input_path)
    
    # Call Rekognition to detect rash
    response = rekognition.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:1611289594039:project/sample-test-rv/version/sample-test-rv.2024-05-01T12.12.20/1714561979389',
        Image={'S3Object': {'Bucket': bucket, 'Name': key}}
    )
    
    # Load image using PIL and convert to OpenCV BGR format
    image = Image.open(input_path).convert('RGB')
    img_cv = np.array(image)[..., ::-1]  # Convert to BGR
    height, width = img_cv.shape[:2]
    
    # Blur the entire image
    blurred = cv2.GaussianBlur(img_cv, (55, 55), 0)

    # Loop through detected rash regions and paste back the original rash area
    for label in response['CustomLabels']:
        if label['Confidence'] > 90 and 'Geometry' in label:
            box = label['Geometry']['BoundingBox']
            
            top = int(box['Top'] * height)
            left = int(box['Left'] * width)
            box_height = int(box['Height'] * height)
            box_width = int(box['Width'] * width)
            
            # Paste original rash region onto blurred image
            blurred[top:top + box_height, left:left + box_width] = img_cv[top:top + box_height, left:left + box_width]

    # Save the final result
    cv2.imwrite(output_path, blurred)

    # Upload to output-images/ folder
    output_key = key.replace('input-images/', 'output-images/')
    s3.upload_file(output_path, bucket, output_key)

    return {
        'statusCode': 200,
        'body': f'Processed and uploaded to {output_key}'
    }
