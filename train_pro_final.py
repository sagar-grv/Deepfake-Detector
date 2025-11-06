import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
# ---!!!--- SAFETY NET 1: IMPORT ModelCheckpoint ---!!!---
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 1. DEFINE CONSTANTS ---
BATCH_SIZE = 16 
IMG_SIZE = (260, 260) 
EPOCHS = 20 

# Define our dataset paths
base_dir = os.path.join("dataset", "Celeb_V2")
train_dir = os.path.join(base_dir, "Train")
val_dir = os.path.join(base_dir, "Val")
test_dir = os.path.join(base_dir, "Test")

# ---!!!--- NEW FILENAME FOR THE FINAL MODEL ---!!!---
# We will save ONLY the weights to this file. This will not crash.
FINAL_WEIGHTS_FILE = "deepfake_detector_FINAL.weights.h5"
CHECKPOINT_FILEPATH = os.path.join("models", FINAL_WEIGHTS_FILE)


print(f"--- Deepfake Detector Initializing (BULLETPROOF_MODE) ---")
print(f"Will save best model weights to: {CHECKPOINT_FILEPATH}")
print("-" * 40)


# --- 2. LOAD DATA ---
print("Loading datasets...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False
)

# --- 3. VERIFY DATA & OPTIMIZE ---
class_names = train_dataset.class_names
print(f"Found classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
print("Data loaded and prefetch-optimized successfully.")
print("-" * 40)


# --- 4. BUILD THE MODEL (This is our "Blueprint") ---
print("Building model with EfficientNetB2 base...")

def build_model():
    base_model = tf.keras.applications.EfficientNetB2(
        weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False) 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
print("Model built and compiled successfully.")
model.summary() 
print("-" * 40)


# --- 5. TRAIN THE MODEL ---
print(f"STARTING TRAINING for a MAX of {EPOCHS} epochs...")

# ---!!!--- SAFETY NET 2: CONFIGURE CALLBACKS ---!!!---

# Callback 1: Stop if it's not learning
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3, # Wait 3 epochs for improvement
    verbose=1,
    restore_best_weights=True # Give us the best model, not the last one
)

# Callback 2: Save the best model to a file *during* training
# This is our iron-clad insurance policy.
model_checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH, # Save to our file
    monitor='val_accuracy',       # Watch validation accuracy
    mode='max',                 # We want the *max* accuracy
    save_best_only=True,        # Only save if it's a new "high score"
    save_weights_only=True,     # THIS IS THE FIX. Only save weights.
    verbose=1                   # Tell us when it saves!
)

print("Callbacks Active: EarlyStopping (patience=3) and ModelCheckpoint (saving best weights).")

start_time = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint] # Pass in BOTH safety nets
)
end_time = time.time()

print("TRAINING COMPLETE!")
print(f"Total training time: {((end_time - start_time) / 60):.2f} minutes")
print("-" * 40)


# --- 6. SAVE OUR WORK ---
# The ModelCheckpoint has *already saved* the best weights.
# We no longer need the broken model.save() command.
# We'll just load the weights our checkpoint saved to be 100% sure.
print(f"Loading best weights from {CHECKPOINT_FILEPATH}...")
model.load_weights(CHECKPOINT_FILEPATH)
print("Best model weights loaded for evaluation.")
print("-" * 40)


# --- 7. EVALUATE THE MODEL ---
print("Evaluating model on the test dataset...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")
print("-" * 40)


# --- 8. PLOT TRAINING HISTORY ---
print("Generating training history plot...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy'] 
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

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
plt.suptitle("Model Training History (EfficientNetB2, Pro)")
plt.savefig("training_history_efficientnet_pro.png")
print("Saved training_history_efficientnet_pro.png to project folder.")
print("-" * 40)


# --- 9. SHOW DETAILED PREDICTIONS ---
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