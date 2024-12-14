import cv2
import matplotlib.pyplot as plt

def show_img(img_path):
    # Show example from Date images

    image = cv2.imread(img_path)
    print (image.shape)
    plt.imshow(image)
    plt.show()



