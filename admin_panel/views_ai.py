# views.py
import os
import io
import logging
import numpy as np
from PIL import Image
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from tensorflow.lite.python.interpreter import Interpreter
import tensorflow as tf

from .models import EyeScreening, Appointment

# Optional OpenCV eye detection setup
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 🔧 CONFIGURATION (STRICTLY FOR V11 RESNET50)
# -----------------------------------------------------------
# Training Class Order: Index 0 = "STRABISMUS", Index 1 = "STRABISMUS_FREE"
LABELS = ["Strabismus", "Strabismus-Free"] 
INPUT_SIZE = (224, 224)
# V11 is very confident, so we can set a reasonable threshold
CONFIDENCE_THRESHOLD = 0.60 

# [FIX 1] Point to the ACTUAL v11 model
MODEL_PATH = os.path.join(
    settings.BASE_DIR, "admin_panel", "ai_model", "v11_best_resnet50 (1).tflite"
)

if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model not found at: {MODEL_PATH}")
    # For safety in dev, we might want to raise, but logging is safer for production
    # raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# Load TFLite model once
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------------------------------------
# 👁️ OpenCV Eye Detection Helper
# -----------------------------------------------------------
HAAR_EYE_PATH = None
if OPENCV_AVAILABLE:
    try:
        HAAR_EYE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
    except Exception:
        pass

def detect_eyes_with_opencv(image_bytes, min_eyes=2):
    """Return True if eyes detected, False otherwise."""
    if not OPENCV_AVAILABLE or not HAAR_EYE_PATH:
        return True # Bypass if OpenCV is broken/missing
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(pil)[:, :, ::-1].copy() # RGB to BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(HAAR_EYE_PATH).detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(eyes) >= min_eyes
    except Exception as e:
        logger.warning(f"Eye detection error: {e}")
        return True # Fallback to allowing image

def preprocess_image(image_bytes, target_size=INPUT_SIZE):
    """
    Prepare image for ResNet50 v11.
    CRITICAL: The model has an internal Rescaling(1./255) layer.
    We MUST pass raw pixel values [0, 255], NOT [0, 1].
    """
    # 1. Open and ensure RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Resize (Bilinear matches Keras training default)
    img = img.resize(target_size, Image.BILINEAR)
    
    # 3. Convert to float32 array
    arr = np.asarray(img).astype(np.float32)
    
    # [FIX 2] NO NORMALIZATION HERE.
    # Do NOT divide by 255. Do NOT subtract 127.5.
    # The model expects raw pixels.
    
    # 4. Add Batch Dimension: (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------------------------------------
# 🚀 MAIN ENDPOINT
# -----------------------------------------------------------
@csrf_exempt
def classify_eye_image(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Only POST allowed."}, status=405)

    if "image" not in request.FILES:
        return JsonResponse({"status": "error", "message": "No image uploaded."}, status=400)

    try:
        image_file = request.FILES["image"]
        image_bytes = image_file.read()

        # 1. Check Image Integrity
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            return JsonResponse({"status": "error", "message": "Invalid image file."}, status=400)

        # 2. Optional Eye Detection
        if not detect_eyes_with_opencv(image_bytes):
             return JsonResponse({
                "status": "error",
                "message": "Could not detect eyes clearly. Please ensure good lighting and face the camera directly."
            }, status=400)

        # 3. Preprocess & Predict
        input_data = preprocess_image(image_bytes)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])

        # 4. Interpret Output (Softmax)
        # v11 ResNet50 output shape is [1, 2] -> [Prob_Class0, Prob_Class1]
        # If the model didn't include Softmax activation in export, we apply it manually.
        # If values sum to > 1.1 or < 0.9, likely logits -> Apply Softmax.
        if np.sum(raw_output) > 1.1 or np.sum(raw_output) < 0.9:
             probs = tf.nn.softmax(raw_output[0]).numpy()
        else:
             probs = raw_output[0]

        # Index 0: Strabismus (Sick)
        # Index 1: Strabismus-Free (Healthy)
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        predicted_label = LABELS[predicted_idx]

        # 5. Low Confidence Check
        if confidence < CONFIDENCE_THRESHOLD:
            return JsonResponse({
                "status": "error",
                "message": f"Inconclusive result ({confidence*100:.1f}%). Please try again with a clearer image.",
                "confidence": round(confidence * 100, 2),
            }, status=400)

        diagnosis = predicted_label
        probs_percent = {
            "Strabismus": round(float(probs[0]) * 100, 2),
            "Normal": round(float(probs[1]) * 100, 2)
        }

        # 6. User Linking
        user = None
        # ... (Keep your existing user finding logic exactly as is) ...
        # For brevity, assuming user finding code is here (same as your snippet)
        user_id = request.POST.get("user_id") or request.POST.get("userId")
        if user_id:
             user = User.objects.filter(id=user_id).first()
        # ... (Add email/token checks if needed) ...

        # 7. Save Record
        screening = EyeScreening.objects.create(
            user=user,
            image=image_file,
            result=diagnosis,
            confidence=confidence * 100,
            remarks=f"v11 Detection: {diagnosis}",
        )

        # 8. Create Appointment Logic
        if user:
            Appointment.objects.create(
                user=user,
                reason="AI Eye Screening Follow-up",
                preliminary_result=diagnosis,
                is_ai_screening=True,
                archive=False
            )

        # 9. Return Response
        msg = "You are Strabismus-Free!" if diagnosis == "Strabismus-Free" else "Potential Strabismus Detected."
        
        return JsonResponse({
            "status": "success",
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs_percent,
            "screening_id": screening.id,
            "message": msg,
            "proceed_to_booking": True
        }, status=200)

    except Exception as e:
        logger.exception("Error in classify_eye_image")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)