from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan domain tertentu misalnya ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model 
model_path = "model/best.pt"
model = YOLO(model_path)

CLASS_COLORS = {
    "Glioma": (0, 255, 0),       # Hijau
    "Meningioma": (0, 0, 255),   # Merah
    "Pituitary": (255, 0, 0),    # Biru
}

@app.post("/predict/")
async def predict_mri_tumor(file: UploadFile = File(...)):
    try:

        start_time = time.time()
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        results = model.predict(source=img, conf=0.5, iou=0.5, imgsz=640)
        result = results[0]
        
        output_image_path = f"static/output_{uuid.uuid4().hex}.jpg"
        
        if result.boxes:
            annotated_img = img.copy()
            
            for box, mask in zip(result.boxes, result.masks):
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                
                if class_name == "No Tumor":
                    continue
                
                # Warna berdasarkan kelas
                color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default putih
                
                # Gambar bounding box
                cv2.rectangle(annotated_img, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             color, 2)
                
                # Gambar mask
                mask_data = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_data, 
                                         (annotated_img.shape[1], annotated_img.shape[0]))
                annotated_img[mask_resized > 0.5] = color
                
                # Label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_img, label, 
                           (int(bbox[0]), int(bbox[1]) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cv2.imwrite(output_image_path, annotated_img)
            
            # Ambil hasil deteksi pertama
            first_box = result.boxes[0]
            confidence = float(first_box.conf)
            class_id = int(first_box.cls)
            class_name = model.names[class_id]
            
            # Hitung waktu pemrosesan
            processing_time = time.time() - start_time
            
            return {
                "message": "sukses melakukan prediksi",
                "result": {
                    "confidence": confidence,
                    "terdeteksi": class_name,
                    "url_image_output": f"/{output_image_path}",
                    "processing_time": f"{processing_time:.2f}s"  # Waktu pemrosesan dalam detik
                }
            }
        else:
            processing_time = time.time() - start_time
            
            return {
                "message": "sukses melakukan prediksi",
                "result": {
                    "confidence": None,
                    "terdeteksi": "Tidak terdeteksi tumor / Bukan MRI otak",
                    "url_image_output": None,
                    "processing_time": f"{processing_time:.2f}s"  # Waktu pemrosesan dalam detik
                }
            }
            
    except Exception as e:
        return {"message": f"Error: {str(e)}"}