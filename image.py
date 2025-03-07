import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("D:\data.csv")
print(df.head())

print("Shape of the dataset: ",df.shape)

labels = df.iloc[:, 0].values 
print("Labels: ",labels.shape) 
images = df.iloc[:, 1:].values  
print("Images: ",images.shape)
images = images.reshape(-1, 28, 28)

#Displaying images
for i in range(10):
    matching_images = []
    for j in range(len(labels)):
        if labels[j] == i:
            matching_images.append(images[j])
    img = matching_images[0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {i}")
    plt.axis("off")

plt.show()

#Verifying Grayscale
min_val = images.min()
max_val = images.max()
print(f"Pixel value range: {min_val} to {max_val}")


pixel_stats = df.iloc[:, 1:].describe().T 
print(pixel_stats[["mean", "std", "min", "max"]])



