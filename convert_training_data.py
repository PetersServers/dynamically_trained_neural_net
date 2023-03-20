# Path to the directory containing the JPEG images
import cv2
import os
import numpy as np
import time

def convert_training_data(directory):

    image_size = (256, 256)

    # Initialize an empty list to store the images
    images = []
    labels = []

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image

        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):

            labels.append(int(filename.rsplit('.', 2)[0].rsplit('_', 1)[-1]))  # split the filename at the last '.' and take the first part
            # Load the image as grayscale
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, image_size)
            '''
            plt.imshow(img, cmap='gray')
            plt.title(filename)
            plt.show()
            print(labels[-1])
            time.sleep(3)
            '''
            # Convert the image to a NumPy array and append it to the list
            img_array = np.array(img_resized)
            images.append(img_array)


    # Convert the list of images to a 4D tensor
    images = np.array(images)
    images = np.reshape(images, (images.shape[0], image_size[0], image_size[1], 1))
    images = images.astype('float32') / 255.0

    # Convert the list of labels to a NumPy array
    labels = np.array(labels)

    return images, labels


