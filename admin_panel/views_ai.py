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

# Try loading OpenCV (optional)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 🔧 SMARTSIGHT V12 CONFIG
# -----------------------------------------------------------
LABELS = ["Strabismus", "Strabismus-Free"]  # Index 0 sick, 1 healthy
INPUT_SIZE = (224, 224)

# SmartSight V12 is VERY accurate (94% accuracy)
CONFIDENCE_THRESHOLD = 0.70  # slightly higher than V11

MODEL_PATH = os.path.join(
    settings.BASE_DIR, "admin_panel", "ai_model", "smartsight_resnet50_v12.tflite"
)

if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ V12 Model NOT found at {MODEL_PATH}")

# Load TFLite model once at server start
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------------------------------------
# 👁️ Optional OpenCV Eye Detection
# -----------------------------------------------------------
HAAR_EYE_PATH = None
if OPENCV_AVAILABLE:
    try:
        HAAR_EYE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
    except Exception:
        pass

def detect_eyes_with_opencv(image_bytes, min_eyes=2):
    if not OPENCV_AVAILABLE or not HAAR_EYE_PATH:
        return True
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(pil)[:, :, ::-1].copy()  # RGB → BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(HAAR_EYE_PATH).detectMultiScale(
            gray, 1.1, 5, minSize=(30, 30)
        )
        return len(eyes) >= min_eyes
    except Exception as e:
        logger.warning(f"Eye detection error: {e}")
        return True

# -----------------------------------------------------------
# 🧠 SMARTSIGHT V12 PREPROCESSING
# -----------------------------------------------------------
def preprocess_image_v12(image_bytes):
    """
    SmartSight V12 uses ResNet50 (caffe-style):
    - Convert RGB → BGR
    - Subtract ImageNet mean: [103.939, 116.779, 123.68]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)

    # Apply EXACT preprocessing used in training
    arr = tf.keras.applications.resnet50.preprocess_input(arr)

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------------------------------------------
# 🚀 MAIN ENDPOINT — SMARTSIGHT V12 CLASSIFIER
# -----------------------------------------------------------
@csrf_exempt
def classify_eye_image(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Only POST allowed."}, status=405)

    if "image" not in request.FILES:
        return JsonResponse({"status": "error", "message": "No image provided."}, status=400)

    try:
        image_file = request.FILES["image"]
        image_bytes = image_file.read()

        # Validate image
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            return JsonResponse({"status": "error", "message": "Invalid or corrupted image."}, status=400)

        # Optional Eye Detection
        if not detect_eyes_with_opencv(image_bytes):
            return JsonResponse({
                "status": "error",
                "message": "Eyes not clearly detected. Ensure bright lighting and face the camera directly."
            }, status=400)

        # Preprocess with V12 rules
        input_data = preprocess_image_v12(image_bytes)

        # Run prediction
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])[0]

        # Ensure Softmax
        probs = tf.nn.softmax(raw_output).numpy()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        predicted_label = LABELS[predicted_idx]

        # Confidence check
        if confidence < CONFIDENCE_THRESHOLD:
            return JsonResponse({
                "status": "error",
                "message": f"Inconclusive result ({confidence * 100:.1f}%). Please upload a clearer image.",
                "confidence": round(confidence * 100, 2),
            }, status=400)

        probs_percent = {
            "Strabismus": round(float(probs[0]) * 100, 2),
            "Normal": round(float(probs[1]) * 100, 2),
        }

        # User linkage
        user = None
        user_id = request.POST.get("user_id") or request.POST.get("userId")
        if user_id:
            user = User.objects.filter(id=user_id).first()

        # Save screening result
        screening = EyeScreening.objects.create(
            user=user,
            image=image_file,
            result=predicted_label,
            confidence=confidence * 100,
            remarks=f"SmartSight V12: {predicted_label}"
        )

        # If user exists, automatically create appointment
        if user:
            Appointment.objects.create(
                user=user,
                reason="AI Eye Screening Follow-up",
                preliminary_result=predicted_label,
                is_ai_screening=True,
                archive=False
            )

        message = (
            "You are Strabismus-Free!"
            if predicted_label == "Strabismus-Free"
            else "Potential Strabismus Detected (Crossed Eyes)."
        )

        return JsonResponse({
            "status": "success",
            "diagnosis": predicted_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs_percent,
            "screening_id": screening.id,
            "message": message,
            "proceed_to_booking": True
        }, status=200)

    except Exception as e:
        logger.exception("V12 Screening Error")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
