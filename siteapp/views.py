from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import cv2
import os

def runVideo(request):
    update_list=[]
    if request.method == 'POST':
        print('Uploading video*********')
        print(update_list)
        # Handle form submission
        uploaded_file = request.FILES['customFile']
        # Save the file 
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'video'))
        filename = fs.save(uploaded_file.name, uploaded_file)
        print(filename)
    
        video = filename
        frames_dir = 'video_info/frames'
        cap = cv2.VideoCapture(video)
        f = cap.get(cv2.CAP_PROP_FPS)                 
        seconds = 60
        frames = int(f * seconds)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frames == 0:
                if frame_count > 0 and frame_count < (cap.get(cv2.CAP_PROP_FRAME_COUNT) - frames):
                    frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_count +=1
            frame_count += 1

        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.transforms import functional as F
        from PIL import Image
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        coco_class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        folder_path = r"video_info/frames"

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path).convert("RGB")
                image_tensor = F.to_tensor(image).unsqueeze(0)
                with torch.no_grad():
                    predictions = model(image_tensor)
                confidence_threshold = 0.7
                filtered_predictions = [pred for pred, score in zip(predictions[0]['labels'].int().tolist(), predictions[0]['scores'].tolist()) if score >= confidence_threshold]
                class_names = [coco_class_names[idx] for idx in filtered_predictions]
                print(f"For image {filename}, Class Names:", class_names)
                items=(f"For image {filename}, Class Names:", class_names)
                update_list.append(items)


        return render(request, 'siteapp/home.html',{'items':update_list})
    update_list.clear()
    return render(request, 'siteapp/home.html')
