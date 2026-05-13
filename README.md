# AI Trash Sorting Robot (Trashboat)

This project implements an AI-powered robot that automatically sorts waste into trash, recycling, and compost using computer vision and a convolutional neural network (CNN) classifier.

## Dataset 

We used the [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) dataset from Kaggle, which originally contained images across 12 waste categories. We downsampled this to 3 classes:

- ♻️ Recycle: batteries, cardboard, glass, metal, paper, plastic 
- 🌱 Compost: food waste, organic matter
- 🗑️ Trash: clothes, shoes, general non-recyclable waste

The images were resized to 32x32 pixels and normalized. We applied random horizontal flips and rotations for data augmentation. The augmented dataset was split into training (70%), validation (15%) and test (15%) sets.

## Model Architectures

### CNN Classifier

```
Input: 3×32×32 (RGB image)
→ Conv(3→32) → ReLU → MaxPool
→ Conv(32→64) → ReLU → MaxPool  
→ Flatten (4096)
→ FC(4096→128) → ReLU
→ FC(128→3) → Softmax
```

Our CNN architecture consists of:
- 2 convolutional layers with 3x3 filters and increasing channel depth 
- ReLU activations and 2x2 max pooling after each conv layer
- 2 fully connected layers for classification
- Softmax output layer

### Object Detection Model

For real-time detection we use a pre-trained object detection model from OpenCV's DNN module:

```python
net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))  
net.setInputSwapRB(True)
```

This performs multi-class object detection. We filter for relevant waste objects and pass the cropped object images to our CNN for final classification.

## Installation

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Unzip datasets and config files:
   ```
   unzip -q originalImages.zip
   unzip -q config_files.zip
   ```

## Usage

### Training
Run the provided Jupyter notebooks to train the CNN classifier:

- `cnntraininggpu.ipynb` if you have a CUDA-enabled GPU 
- `cnntrainingcpu.ipynb` if you only have a CPU

Trained models are saved in the `savedModels/` directory.

### Real-time Inference
1. Edit `source/webcam.py` and set `model_path` to your trained CNN weights
2. Run the script: `python source/webcam.py`

This opens a webcam feed, detects waste objects, classifies them, and prints the predicted label.

### Single Image Inference 
1. Edit `source/main_image.py` and set:
   - `model_path` to your CNN weights path
   - `img_path` to your input image path
2. Run the script: `python source/main_image.py`

Predicted class label will be printed for the input image.

## Hardware

- Raspberry Pi 4 Model B (4GB RAM)
- Raspberry Pi Camera Module V2
- Arduino Uno Microcontroller 
- 2 x SG90 9G Micro Servo Motors
- 3D printed chassis and gear mechanisms

Refer to the report for detailed wiring diagrams and 3D models.

## Results

| Model    | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|----------|---------------|----------------|-------------|---------------|
| Custom CNN | 95.2%       | 94.8%          | 94.3%       | 94.5%         |
| ResNet18 | 97.5%         | 96.0%          | 95.8%       | 95.9%         |
| YOLOv8   | 98.3%         | 97.7%          | 97.7%       | 97.7%         |

YOLOv8 performed best overall, achieving 98% test accuracy. However, real-time inference performance was lower at around 80% due to motion blur and lighting variations. This can potentially be improved with more data augmentation and domain adaptation techniques.

Confusion Matrix for YOLOv8 on test set:
```
[[285   3   2]
 [  4 312   1]  
 [  2   1 274]]
```

## Demo

Here are some examples of Trashboat correctly sorting waste objects:

&lt;img src="demo1.jpg" width=40%&gt;
Prediction: Recycle ✅

&lt;img src="demo2.jpg" width=40%&gt;  
Prediction: Compost ✅

&lt;img src="demo3.jpg" width=40%&gt;
Prediction: Trash ✅

## Challenges &amp; Future Work

- The gripper mechanism struggled with very small objects. Using a soft compliant gripper design could improve grasping.
- Servo jitter caused some misalignment in the sorting bins. PID servo control could help.
- Lighting variations affected classification accuracy. Using exposure control and color correction algorithms can make the model more robust.
- Integrating a dedicated object detection model like YOLOv5/v8 may improve accuracy over the generic OpenCV DNN detector.
- Collecting a larger real-world dataset covering more waste object types and environments.

## Team

- Owen Barnes
- Felistus Karanja  
- Ryan Domathoti
- Daniel Anoruo
- Frankie Horter
- Myles Burrows

## References
1. [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) 
2. [OpenCV DNN Module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
3. [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
4. [YOLO Waste Detection](https://github.com/kimbring2/Waste_Detection_and_Classification)
