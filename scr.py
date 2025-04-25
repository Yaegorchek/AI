import os
from PIL import Image
import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype='>i4')[0]
        if magic == 2051:  # Изображения
            num_images, rows, cols = np.frombuffer(f.read(12), dtype='>i4')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            return images
        elif magic == 2049:  # Метки
            num_labels = np.frombuffer(f.read(4), dtype='>i4')[0]
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        else:
            raise ValueError("Неверный формат MNIST файла")

# Пути к файлам
images_path = r"C:/nums/train-images-idx3-ubyte"
labels_path = r"C:/nums/train-labels.idx1-ubyte"

# Загрузка данных
images = read_idx(images_path)
labels = read_idx(labels_path)

# Папка для сохранения
output_dir = 'mnist_renamed'
os.makedirs(output_dir, exist_ok=True)

# Счетчики для каждой цифры
counters = {digit: 0 for digit in range(10)}

# Переименование и сохранение
for i in range(len(images)):
    digit = labels[i]
    img = Image.fromarray(images[i])
    new_name = f"{digit}_{counters[digit]}.jpg"
    img.save(os.path.join(output_dir, new_name))
    counters[digit] += 1

print(f"Все изображения переименованы и сохранены в {output_dir}")
print("Количество изображений для каждой цифры:")
for digit, count in counters.items():
    print(f"{digit}: {count} изображений")