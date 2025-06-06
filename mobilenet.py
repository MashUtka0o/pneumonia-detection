import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

split = np.load('./mobileNetData/chest_xray_split_mobileNet.npz') # loading the splitted dataset for MobileNet

def train_model(split): # training the model
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
                        epochs=20,
                        batch_size=32,
                        callbacks=[checkpoint]) # using checkpoint for callbacks
    test_loss, test_acc = model.evaluate(x_test, y_test_cat)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred_classes, target_names=['NORMAL', 'PNEUMONIA']))
    return history

def accuracy_plot(history): # if you want to plot accuracy
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

def plot_confusion_matrix(model, split):
    x_test, y_test = split['data_test'], split['labels_test']
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)  # confusion matrix
    labels = ['NORMAL', 'PNEUMONIA']

    test_loss, test_acc = model.evaluate(x_test, to_categorical(y_test, num_classes=2), verbose=0)
    print(f"Loaded model test accuracy: {test_acc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title='Binary Confusion Matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

history = train_model(split) # train the model
accuracy_plot(history) # plot the training process

plot_confusion_matrix(load_model('./models/mobilenet_best.h5'), split)
