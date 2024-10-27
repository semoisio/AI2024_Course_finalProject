import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class LiverDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the images directory.
            masks_dir (str): Path to the labels (masks) directory.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # List of all image filenames
        self.image_filenames = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get the image path
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Create corresponding mask filename
        mask_filename = img_filename.replace(".png", "_mask.png")  # Adjust based on your image format
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # Load the mask
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale
        """
        if mask_filename == "000_mask.png":
            print("Mask 000")
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Image')
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title('Mask')
            plt.show()
        """
        
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # Apply the same transformations to the mask

        return image, mask


# Paths to dataset directories
images_dir = r".\Liver_Medical_Image _Datasets\Images"
masks_dir = r".\Liver_Medical_Image _Datasets\Labels"

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])

# Create the dataset
liver_dataset = LiverDataset(images_dir, masks_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = 360
val_size = len(liver_dataset) - train_size

# Check if the dataset has enough images
if len(liver_dataset) < 400:
    raise ValueError("Not enough images in the dataset. Please ensure there are 400 images.")

# Split the dataset
train_dataset, val_dataset = random_split(liver_dataset, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Determine the used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Contracting path
        self.enc_conv1 = self.double_conv(in_channels, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Expanding path
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(1024, 512)
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(512, 256)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            # default stride 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Contracting path
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(F.max_pool2d(enc1, 2))
        enc3 = self.enc_conv3(F.max_pool2d(enc2, 2))
        enc4 = self.enc_conv4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expanding path
        dec4 = self.up_conv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec_conv4(dec4)

        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec_conv1(dec1)

        # Output
        return self.out_conv(dec1)
    

# Initialize the U-Net model
model = UNet(in_channels=3, out_channels=1).to(device)

model.load_state_dict(torch.load(r'.\unet_liver_segmentation.pth'))

# Define function for IoU
def iou(pred, target, n_classes=1):
    ious = []
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    
    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.mean(ious)

# Define function for Pixel Accuracy
def pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()  # Thresholding at 0.5
    target = (target > 0.5).float()  # Ensure target is also binary

    # Print dimensions
    #print("Prediction shape:", pred.shape)
    #sprint("Target shape:", target.shape)

    correct = torch.eq(pred, target).int()
    accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

# Evaluate on test set
model.eval()
miou_total = []
mpa_total = []

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        #print(outputs)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid since using BCEWithLogitsLoss
        #print(outputs)
        # Calculate mIoU and mPA
        miou_total.append(iou(outputs, masks))
        mpa_total.append(pixel_accuracy(outputs, masks))

mean_miou = np.mean(miou_total)
mean_mpa = np.mean(mpa_total)

print(f"Mean IoU: {mean_miou}")
print(f"Mean Pixel Accuracy: {mean_mpa}")




def segment_single_image(model, image_path, transform, device):
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)  # Apply sigmoid to get the probabilities
    
    # Convert the output to a binary mask (threshold at 0.5)
    output_mask = output.squeeze().cpu().numpy() > 0.5
    
    return output_mask

# Example usage
image_path = r".\Liver_Medical_Image _Datasets\Images\241.png"
output_mask = segment_single_image(model, image_path, transform, device)

# Plot the output mask
plt.imshow(output_mask, cmap="gray")
plt.show()

# Save the output mask
Image.fromarray((output_mask * 255).astype(np.uint8)).save("segmented_liver_image.jpg")
