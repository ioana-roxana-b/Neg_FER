import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the path to the dataset
dataset_path = 'CK+48'

# Define the target image size
img_size = (224,224)

# Define the data transformations with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=keras.applications.vgg16.preprocess_input
)

# Load the dataset and preprocess the images
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=40,
    class_mode='categorical',
    subset='training'
)
test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=40,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Load the VGG16 model
base_model = keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(img_size[0], img_size[1], 3)
)

# Add custom top layers with regularization
x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# Fine-tuning the base model layers
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Learning rate scheduling
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model with callbacks
num_epochs = 50
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy: %.2f%%' % (test_acc * 100))

# Make predictions on the test set
y_pred = np.argmax(model.predict(test_generator), axis=-1)
y_true = test_generator.classes

# Compute confusion matrix
conf_mat = tf.math.confusion_matrix(y_true, y_pred)
print(conf_mat)
"""""
# Print confusion matrix
plt.imshow(conf_mat, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(train_generator.num_classes)
plt.xticks(tick_marks, train_generator.class_indices, rotation=45)
plt.yticks(tick_marks, train_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
"""
