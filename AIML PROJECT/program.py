import cv2
import numpy as np
import argparse
import os


# Paths to the pre-trained model files and input image
DIR=r"C:/Users/DEEPIKA/OneDrive/Documents/vspython/AIML"
prototxt = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
model = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")
points = os.path.join(DIR, r"model/pts_in_hull.npy")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args=vars(ap.parse_args())
print("Load model")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts=np.load(points)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
image=cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab=cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized=cv2.resize(lab, (224, 224))
L=cv2.split(resized)[0]
L -= 50
print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab=net.forward()[0, :, :, :].transpose((1, 2, 0))
ab=cv2.resize(ab, (image.shape[1], image.shape[0]))
L=cv2.split(lab)[0]
colorized=np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized=cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized=(255*np.clip(colorized, 0, 1)).astype("uint8")
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)