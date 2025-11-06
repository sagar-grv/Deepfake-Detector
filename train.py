import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time

# --- 1. DEFINE CONSTANTS ---

# --- SPEED TWEAKS APPLIED ---
# We are using 128x128 images (4x smaller)
# This lets us use a 2x bigger batch size. This = SPEED.
BATCH_SIZE = 32
IMG_SIZE = (128, 128) # Was (256, 256)
EPOCHS = 10 

# Define our dataset paths
base_dir = os.path.join("dataset", "Celeb_V2")
train_dir = os.path.join(base_dir, "Train")
val_dir = os.path.join(base_dir, "Val")
test_dir = os.path.join(base_dir, "Test")

print(f"--- Deepfake Detector Initializing (FAST_MODE) ---")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Found GPU: {tf.config.list_physical_devices('GPU')}")
print(f"Batch Size: {BATCH_SIZE} (Upped for speed)")
print(f"Image Size: {IMG_SIZE} (Downsized for speed)")
print(f"Epochs to run: {EPOCHS}")
print("-" * 40)


# --- 2. LOAD DATA ---
print("Loading datasets...")

# Use Keras's magic utility to load images from directories.
# It automatically infers the labels (REAL/FAKE) from the folder names.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary' # REAL vs FAKE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False # Keep test data in order for evaluation
)

# --- 3. VERIFY DATA & OPTIMIZE ---
class_names = train_dataset.class_names
print(f"Found classes: {class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

print("Data loaded and prefetch-optimized successfully.")
print("-" * 40)


# --- 4. BUILD THE MODEL ---
print("Building model...")

# 1. Load the pre-trained XceptionNet model
base_model = tf.keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet
    include_top=False,   # Ditch the final 1000-class layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3) # Now uses (128, 128, 3)
)

# 2. Freeze the base model
base_model.trainable = False

# 3. Create our new "head" (the part we will train)
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# We need to scale pixel values from [0, 255] to [-1, 1]
x = tf.keras.applications.xception.preprocess_input(inputs)

# Pass the preprocessed images to our frozen base model
x = base_model(x, training=False) 

# Add our own layers on top
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Flattens the output
x = tf.keras.layers.Dense(256, activation='relu')(x) # A "thinking" layer
x = tf.keras.layers.Dropout(0.5)(x) # Helps prevent overfitting
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Our final 0-or-1 answer

# 4. Put it all together
model = tf.keras.Model(inputs, outputs)

# 5. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Let's see what we built!
print("Model built and compiled successfully.")
model.summary()
print("-" * 40)


# --- 5. TRAIN THE MODEL ---
print(f"STARTING TRAINING for {EPOCHS} epochs...")
print("Your GPU is now at work. This will be much faster.")

start_time = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)
end_time = time.time()

print("TRAINING COMPLETE!")
print(f"Total training time: {((end_time - start_time) / 60):.2f} minutes")
print("-" * 40)


# --- 6. SAVE OUR WORK ---
print("Saving model...")
model.save(os.path.join("models", "deepfake_detector_v1.h5"))
print("Model saved to models/deepfake_detector_v1.h5")
print("-" * 40)


# --- 7. EVALUATE THE MODEL ---
print("Evaluating model on the test dataset...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")
print("-" * 40)


# --- 8. PLOT TRAINING HISTORY ---
# ---!!!--- BUG FIX APPLIED HERE ---!!!---
print("Generating training history plot...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy'] # Fixed: was 'validation_accuracy'
loss = history.history['loss']
val_loss = history.history['val_loss']     # Fixed: was 'validation_loss'

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.suptitle("Model Training History")
plt.savefig("training_history.png")
print("Saved training_history.png to project folder.")
print("-" * 40)


# --- 9. SHOW DETAILED PREDICTIONS ---
# Get the true labels and our model's predictions
print("Generating detailed classification report...")
y_true = []
for images, labels in test_dataset:
    y_true.extend(labels.numpy())

y_pred_probs = model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Print the full report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_true, y_pred))
print("-" * 40)
print("All steps complete. Project finished!")