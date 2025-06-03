import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

split = np.load('./mobileNetData/chest_xray_split_mobileNet.npz') # loading the splitted dataset for MobileNet
x_train, y_train = split['data_train'], split['labels_train']
x_val, y_val = split['data_val'], split['labels_val']
x_test, y_test = split['data_test'], split['labels_test']

img_size = 224
num_classes = 2

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet') # I use MobileNet v2
base_model.trainable = False  # Freeze base to prevent weights from being updated during training

inputs = Input(shape=(img_size, img_size, 3)) # input layer should match the image size and RGB channels
x = base_model(inputs, training=False) # first going through base model
x = GlobalAveragePooling2D()(x) # applying pooling to reduce dimensions and prepare for classification
outputs = Dense(num_classes, activation='softmax')(x) #  I chose a dense softmax for 2-class classification as an output layer
model = Model(inputs, outputs) # building the model

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) # Using accuracy as a metric

checkpoint = ModelCheckpoint('./models/mobilenet_best.h5', # I use the best model as a checkpoint
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose=1)

history = model.fit(x_train, y_train_cat,
                    validation_data=(x_val, y_val_cat),
                    epochs=10,
                    batch_size=32,
                    callbacks=[checkpoint]) # using checkpoint for callbacks

test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred_classes, target_names=['NORMAL', 'PNEUMONIA']))

# Plots:

plt.figure(figsize=(12, 5)) # accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss') # train loss plot
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred_classes)  # confusion matrix
labels = ['NORMAL', 'PNEUMONIA']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
