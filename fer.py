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

# Define the data transformations
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=keras.applications.vgg16.preprocess_input
)

# Load the dataset and preprocess the images
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
test_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load the VGG16 model
base_model = keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(img_size[0], img_size[1], 3)
)

# Add custom top layers
x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
num_epochs = 50
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=test_generator
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy: %.2f%%' % (test_acc * 100))

# Make predictions on the test set
y_pred = np.argmax(model.predict(test_generator), axis=-1)
y_true = test_generator.classes

# Compute confusion matrix
conf_mat = tf.math.confusion_matrix(y_true, y_pred)

# Print confusion matrix

plt.imshow(conf_mat, cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(train_generator.num_classes)
plt.xticks(tick_marks, train_generator.class_indices, rotation=45)
plt.yticks(tick_marks, train_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
