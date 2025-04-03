import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import json

# Define image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATASET_DIR = "/Users/vahid/Documents/Pumpkin_Split_Dataset/train"

# Check if dataset directory exists
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' not found. Please check the path.")

# Data augmentation and loading
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False)  # Ensure order is maintained for evaluation

# Print number of classes and images per class
class_labels = list(train_generator.class_indices.keys())
print(f"Total Classes: {len(class_labels)}")
print("Images per Class:")
for class_name, class_index in train_generator.class_indices.items():
    print(f"{class_name}: {np.sum(train_generator.classes == class_index)} images")

# Save class labels to a JSON file
class_labels_dict = {index: class_name for class_name, index in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(class_labels_dict, f)

print("Class labels saved successfully!")

# Define an improved MP Neuron model
input_layer = Input(shape=(128, 128, 3))
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Improve generalization
out_layer = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=out_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate for better convergence
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15)  # Increased epochs for better training

# Save the trained model
model.save('final_model.h5')

# Print MP model accuracy in percentage
final_acc = history.history['accuracy'][-1] * 100
print(f'MP Model Accuracy: {final_acc:.2f}%')

# Plot accuracy graph
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title(f'Training and Validation Accuracy\nFinal Accuracy: {final_acc:.2f}%')
plt.legend()
plt.grid(True)
plt.show()

# Create feature extractor model
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
feature_extractor.save('feature_extractor.h5')

print("Feature extractor model saved successfully!")

# Extract features for standardization
train_features = feature_extractor.predict(train_generator)
validation_features = feature_extractor.predict(validation_generator)

# Standardize features
scaler = StandardScaler()
scaler.fit(train_features)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully!")

# Apply standardization
train_features = scaler.transform(train_features)
validation_features = scaler.transform(validation_features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)  # You can adjust the number of components
train_features_pca = pca.fit_transform(train_features)
validation_features_pca = pca.transform(validation_features)

# Save PCA model
joblib.dump(pca, 'pca.pkl')
print("PCA model saved successfully!")

# Evaluate model performance
y_true = validation_generator.classes  # True labels
y_pred = model.predict(validation_generator)  # Predictions
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.show()
