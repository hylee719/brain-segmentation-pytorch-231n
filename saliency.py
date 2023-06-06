# create saliency maps as a visual
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from captum.attr import Saliency

# Load the pre-trained model
model = resnet50(pretrained=True)
model.eval()

# Load and preprocess the input image
image = Image.open('input_image.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Generate saliency map
saliency = Saliency(model)
attributions = saliency.attribute(input_batch, target=0)

# Convert the saliency map to a NumPy array
saliency_map = attributions.squeeze(0).detach().numpy()

# Save the saliency map as an image
plt.imshow(saliency_map, cmap='hot')
plt.axis('off')
plt.savefig('saliency_map.jpg', bbox_inches='tight', pad_inches=0)

