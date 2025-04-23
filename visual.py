from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

class NeuralNetworkApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Обнаружение цифры 7")
        self.window.geometry('950x720')
        self.window.configure(bg='#e8ecf0')

        self.selected_network = IntVar(value=1)
        self.image_path = None

        # Загрузка предобученных моделей
        self.model1 = self.load_model_1()
        self.model2 = self.load_model_2()

        self.create_widgets()

    def create_widgets(self):
        button_style = {
            'bg': '#1E88E5', 'fg': 'white',
            'font': ('Segoe UI', 10, 'bold'),
            'padx': 15, 'pady': 8, 'bd': 0,
            'relief': FLAT
        }

        style = {'font': ('Segoe UI', 11), 'bg': '#e8ecf0'}

        control_frame = Frame(self.window, bg='#e8ecf0')
        control_frame.pack(pady=20)

        self.btn_load = Button(control_frame, text="Загрузить изображение", 
                               command=self.load_image, **button_style)
        self.btn_load.grid(column=0, row=0, padx=10)

        radio_frame = Frame(control_frame, bg='#e8ecf0')
        radio_frame.grid(column=1, row=0, padx=20)

        self.radio1 = Radiobutton(radio_frame, text="Нейросеть 1 (VGG16)", 
                                  variable=self.selected_network, value=1, **style)
        self.radio1.pack(anchor=W)

        self.radio2 = Radiobutton(radio_frame, text="Нейросеть 2 (EfficientNet)", 
                                  variable=self.selected_network, value=2, **style)
        self.radio2.pack(anchor=W)

        self.btn_process = Button(control_frame, text="Обработать изображение", 
                                  command=self.process_image, **button_style)
        self.btn_process.grid(column=2, row=0, padx=10)

        image_frame = Frame(self.window, bg='#e8ecf0')
        image_frame.pack(pady=10)

        self.original_canvas = self.create_image_panel(image_frame, "Оригинал", 0)
        self.processed_canvas = self.create_image_panel(image_frame, "Результат", 1)

        self.result_label = Label(self.window, text="", font=('Segoe UI', 13, 'bold'), 
                                  bg='#e8ecf0')
        self.result_label.pack(pady=15)

    def create_image_panel(self, parent, label_text, column):
        frame = Frame(parent, bd=2, relief=GROOVE, bg='white')
        frame.grid(row=0, column=column, padx=20)

        Label(frame, text=label_text, font=('Segoe UI', 11, 'bold'), bg='white').pack(pady=6)
        canvas = Canvas(frame, width=400, height=400, bg='white', bd=0, highlightthickness=0)
        canvas.pack()
        return canvas

    def load_model_1(self):
        print("Загрузка модели VGG16...")
        return load_model("vgg16_model.h5")

    def load_model_2(self):
        print("Загрузка модели EfficientNetB5...")
        return load_model("efficientnet_model.h5")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.show_image(file_path, self.original_canvas)
            self.result_label.config(text=f"Изображение загружено: {file_path.split('/')[-1]}", fg='black')

    def show_image(self, path, canvas):
        canvas.delete("all")
        img = Image.open(path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        canvas.image = photo
        canvas.create_image(200, 200, image=photo, anchor='center')

    def predict_with_model(self, model, image_path, preprocess_func):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))  # Под размер модели
        img_array = np.array(img)
        img_array = preprocess_func(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Предположим: 7 = класс 7
        return predicted_class == 7

    def process_image(self):
        if not self.image_path:
            self.result_label.config(text="Сначала загрузите изображение!", fg='red')
            return

        try:
            if self.selected_network.get() == 1:
                result = self.predict_with_model(self.model1, self.image_path, vgg_preprocess)
            else:
                result = self.predict_with_model(self.model2, self.image_path, eff_preprocess)

            self.show_image(self.image_path, self.processed_canvas)

            if result:
                self.result_label.config(text="✅ Цифра 7 обнаружена на изображении", fg='green')
            else:
                self.result_label.config(text="❌ Цифра 7 не обнаружена", fg='red')

        except Exception as e:
            self.result_label.config(text=f"Ошибка: {str(e)}", fg='red')

# Запуск
window = Tk()
app = NeuralNetworkApp(window)
window.mainloop()
