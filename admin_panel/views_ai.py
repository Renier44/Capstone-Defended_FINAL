import os
import io
import logging
import numpy as np
import cv2  # Must install: pip install opencv-python-headless
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from tensorflow.lite.python.interpreter import Interpreter

from .models import EyeScreening, Appointment

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 🔧 MODEL CONFIGURATION (V12 RESNET50)
# -----------------------------------------------------------
LABELS = ["Strabismus", "Strabismus-Free"] 
INPUT_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60 

# Path to your V12 Model
MODEL_PATH = os.path.join(
    settings.BASE_DIR, "admin_panel", "ai_model", "smartsight_resnet50_v12.tflite"
)

# Load TFLite Interpreter
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ V12 Model Loaded Successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")

# Load Haar Cascade for Eye Detection (Built-in to OpenCV)
HAAR_EYE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

# -----------------------------------------------------------
# 🕵️ STRICT VALIDATION LOGIC
# -----------------------------------------------------------
def validate_eye_strict(image_bytes):
    """
    Returns (True, "") if valid.
    Returns (False, "Reason") if invalid.
    Strictly checks for human eyes and distance.
    """
    try:
        # 1. Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_image)
        # Convert RGB (PIL) to BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        img_h, img_w = gray.shape

        # 2. Detect Eyes
        eye_cascade = cv2.CascadeClassifier(HAAR_EYE_PATH)
        # scaleFactor=1.1, minNeighbors=5 (High strictness to avoid false positives like furniture)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # CHECK 1: Are there any eyes?
        if len(eyes) == 0:
            return False, "⚠️ No human eyes detected. Please ensure the photo is clear and contains eyes."

        # CHECK 2: Distance Check (Is it close up?)
        # We calculate the width of the largest eye relative to the image width.
        max_eye_width = 0
        for (x, y, w, h) in eyes:
            if w > max_eye_width:
                max_eye_width = w
        
        # Calculate ratio: Eye Width / Image Width
        # If the eye is smaller than 10% of the image width, the person is too far away.
        eye_ratio = max_eye_width / img_w
        
        if eye_ratio < 0.10: 
            return False, "⚠️ Too far away! Please move the camera CLOSER to your eyes."

        return True, ""

    except Exception as e:
        logger.error(f"Validation Error: {e}")
        # If OpenCV fails unexpectedly, we fail safe (or you can choose to pass)
        return False, "Could not validate image content."

def preprocess_for_v12(image_bytes):
    """
    Preprocessing specifically for ResNet50 (Caffe Mode):
    1. Resize to 224x224
    2. Convert RGB -> BGR
    3. Subtract Mean [103.939, 116.779, 123.68]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SIZE, Image.BILINEAR)
    
    img_array = np.array(img).astype(np.float32)

    # Convert RGB to BGR
    img_array = img_array[..., ::-1]

    # Mean Subtraction (ImageNet)
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]

    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# -----------------------------------------------------------
# 🚀 API ENDPOINT
# -----------------------------------------------------------
@csrf_exempt
def classify_eye_image(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Only POST allowed"}, status=405)

    if "image" not in request.FILES:
        return JsonResponse({"status": "error", "message": "No image uploaded"}, status=400)

    try:
        image_file = request.FILES["image"]
        image_bytes = image_file.read()

        # 1. STRICT VALIDATION
        # This will block: Dogs, Chairs, Full Body shots, Far away shots
        is_valid, error_message = validate_eye_strict(image_bytes)
        
        if not is_valid:
            return JsonResponse({
                "status": "error",
                "message": error_message
            }, status=400)

        # 2. PREPROCESSING (V12 Specific)
        input_data = preprocess_for_v12(image_bytes)

        # 3. INFERENCE
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])

        # 4. INTERPRETATION
        probs = softmax(raw_output[0])
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        diagnosis = LABELS[predicted_idx]

        # 5. RESULT FORMATTING
        probs_percent = {
            "Strabismus": round(float(probs[0]) * 100, 2),
            "Normal": round(float(probs[1]) * 100, 2)
        }

        # Save to DB logic (Simplified for brevity, keep your existing save logic)
        user_id = request.POST.get("user_id")
        user = User.objects.filter(id=user_id).first() if user_id else None
        
        screening = EyeScreening.objects.create(
            user=user,
            image=image_file,
            result=diagnosis,
            confidence=confidence * 100,
            remarks=f"V12 Diagnosis: {diagnosis}"
        )
        
        if user:
             Appointment.objects.create(
                user=user,
                reason="AI Screening Follow-up",
                preliminary_result=diagnosis,
                is_ai_screening=True,
                archive=False
            )

        message = "You are Strabismus-Free!" if diagnosis == "Strabismus-Free" else "Potential Strabismus Detected."

        return JsonResponse({
            "status": "success",
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs_percent,
            "message": message,
            "image_id": screening.id
        })

    except Exception as e:
        logger.exception("Error processing image")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)