import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Provide the path to the local dataset directory
base_dir = r'C:\Users\kanda\OneDrive\Desktop\Deepfake Detection\archive\Dataset'  # Adjust to your dataset's location

# Image parameters
img_size = 224
batch_size = 36  # Adjust based on your system's capacity

for subdir in os.listdir(base_dir):
    print(f"Contents of {subdir}:")
    print(os.listdir(os.path.join(base_dir, subdir)))

train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')

print(f"Fake in Train: {len(os.listdir(os.path.join(train_dir, 'Fake')))}")
print(f"Real in Train: {len(os.listdir(os.path.join(train_dir, 'Real')))}")



# Set the path for an example image
image_path = os.path.join(validation_dir, 'Fake', 'fake_0.jpg')  # Adjust to match your structure


# Reading and displaying the image
img = mpimg.imread(image_path)
print(img.shape)

plt.imshow(img)
plt.axis('off')
plt.show()

# Image Data Generators
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% of testing
)

# Train Generator
train_gen = data_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation Generator
valid_gen = data_gen.flow_from_directory(
    validation_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model definition
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_gen.num_classes, activation='softmax'))

# Model summary
model.summary()

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=5,
    validation_data=valid_gen,
    validation_steps=valid_gen.samples // batch_size
)
# Save the model to a file
model.save('deepfake_detection_model.keras')
print("Model saved successfully!")

print("Evaluating model........")
val_loss, val_accuracy = model.evaluate(valid_gen, steps=valid_gen.samples // batch_size)
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

def load_and_preprocess(image_path, target_size=(224, 224)):
    img = Image.open(image_path)  # Load the image
    img = img.resize(target_size)  # Resize the image
    img_array = np.array(img)  # Convert image into numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Scale the image value to [0,1]
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get class with highest probability
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

# Create a mapping from class indices to class name
class_indices = {v: k for k, v in train_gen.class_indices.items()}
class_indices

# Save the class names as a JSON file
json.dump(class_indices, open('class_indices.json', 'w'))

# Example predictions
image_path = r'C:\Users\kanda\OneDrive\Desktop\Deepfake Detection\archive\Dataset\Validation\Real/real_1000.jpg'  # Adjust as necessary
predicted_class_name = predict_image_class(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)

image_path = r'C:\Users\kanda\OneDrive\Desktop\Deepfake Detection\archive\Dataset\Validation\Fake/fake_10.jpg'  # Adjust as necessary
predicted_class_name = predict_image_class(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)

image_path = r'C:\Users\kanda\OneDrive\Desktop\Deepfake Detection\archive\Dataset\Validation\Real/real_0.jpg'  # Adjust as necessary
predicted_class_name = predict_image_class(model, image_path, class_indices)
print("Predicted Class Name:", predicted_class_name)

# Save the model to a file
model.save('deepfake_detection_model.keras')
print("Model saved successfully!")
