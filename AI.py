import os
import math
import tensorflow as tf
from tensorflow.keras.applications import VGG16, EfficientNetB5
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Параметры
img_size = (224, 224)
batch_size = 32
epochs = 10

# Пути к данным
train_dir = r'C:\VSC\AI\AI\train'
validation_dir = r'C:\VSC\AI\AI\val'

# Генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Загрузка данных
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Проверка загрузки данных
print(f"Классы: {train_generator.class_indices}")
print(f"Тренировочные данные: {train_generator.samples} изображений")
print(f"Валидационные данные: {validation_generator.samples} изображений")
print(f"Всего фото: {validation_generator.samples + train_generator.samples}" )

# Функция для создания модели
def create_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Функция для обучения и сохранения модели
def train_model(model, model_name):
    steps_per_epoch = math.ceil(train_generator.samples / batch_size)
    validation_steps = math.ceil(validation_generator.samples / batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )

    # Сохранение модели
    model.save(f"{model_name}_model.h5")
    print(f"Модель {model_name} сохранена как {model_name}_model.h5")

    # Построение графиков
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Создание или загрузка моделей
num_classes = 1

# VGG16
vgg16_model_path = 'VGG16_model.h5'
if os.path.exists(vgg16_model_path):
    print("Загрузка модели VGG16 из файла...")
    vgg16_model = load_model(vgg16_model_path)
else:
    print("Создание и обучение модели VGG16...")
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    vgg16_model = create_model(vgg16_base, num_classes)
    vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    train_model(vgg16_model, 'VGG16')

# EfficientNetB5
effnet_model_path = 'EfficientNetB5_model.h5'
if os.path.exists(effnet_model_path):
    print("Загрузка модели EfficientNetB5 из файла...")
    effnet_model = load_model(effnet_model_path)
else:
    print("Создание и обучение модели EfficientNetB5...")
    effnet_base = EfficientNetB5(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    effnet_model = create_model(effnet_base, num_classes)
    effnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    train_model(effnet_model, 'EfficientNetB5')
