import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the pre-trained model
model = torch.hub.load('facebookresearch/detectron2', 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')

# Load the input image
img_path = 'path/to/your/image.jpg'
img = plt.imread(img_path)

# Transform the image to the required format
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = transform(img)

# Run the model on the input image
model.eval()
predictions = model([img_tensor])[0]

# Visualize the predictions
plt.imshow(img)
for i in range(len(predictions['keypoints'])):
    x, y, score = predictions['keypoints'][i]
    if score > 0.5:
        plt.scatter(x, y, color='r')
plt.show()
