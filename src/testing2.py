import pickle
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from mtcnn import MTCNN
from PIL import Image
import os
import cv2

# Initialize model and detector
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()

# Paths
data_path = 'data'
save_path = 'artifacts/extracted_features/embedding.pkl'

# List to store features
feature_list = []

# Extract features for all images in the dataset
actors = os.listdir(data_path)
for actor in actors:
    actor_path = os.path.join(data_path, actor)
    for file in os.listdir(actor_path):
        img_path = os.path.join(actor_path, file)
        img = cv2.imread(img_path)
        results = detector.detect_faces(img)

        if results:
            x, y, width, height = results[0]['box']
            face = img[y:y+height, x:x+width]

            image = Image.fromarray(face)
            image = image.resize((224, 224))
            face_array = np.asarray(image)
            face_array = face_array.astype('float32')

            expanded_img = np.expand_dims(face_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img)
            result = model.predict(preprocessed_img).flatten()

            feature_list.append(result)

# Save the features to a file
with open(save_path, 'wb') as f:
    pickle.dump(np.array(feature_list), f)
