import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tensorflow as tf

count = [0] * 10
class GUIApplication:

    def __init__(self, master=None, model=None):
        self.model = model
        self.master = master
        self.canvas_width = 280  # Change this to change the size of the drawing area
        self.canvas_height = 280  # Change this to change the size of the drawing area
        self.pen_width = 5  # Change this to change the thickness of the pen
        self.color = "black"
        self.points = []
        self.setup()

    def setup(self):
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.button = tk.Button(self.master, text="Predict", command=self.predict)
        self.button.pack()

        self.label_input = tk.Entry(self.master)
        self.label_input.pack()

        self.label = tk.Label(self.master, text="")
        self.label.pack()

        self.train_button = tk.Button(self.master,text="Train",command=self.train)
        self.train_button.pack()

        self.training_data = []

    def paint(self, event):
        x1, y1 = (event.x - self.pen_width), (event.y - self.pen_width)
        x2, y2 = (event.x + self.pen_width), (event.y + self.pen_width)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color)
        self.points.append((event.x, event.y))

    def predict(self):
        image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        draw = ImageDraw.Draw(image)
        for point in self.points:
            draw.ellipse([point[0]-self.pen_width, point[1]-self.pen_width, point[0]+self.pen_width, point[1]+self.pen_width], fill="black")

        image = image.resize((28, 28),Image.NEAREST)
        image = np.array(image)
        image = np.invert(image)
        _,image = cv2.threshold(image,175,255,cv2.THRESH_BINARY)
        image = image.flatten().reshape(1,28, 28)
        if(self.label_input.get().strip()):
            if (int(self.label_input.get()) == 0): count[0] += 1
            elif (int(self.label_input.get()) == 1): count[1] += 1
            elif (int(self.label_input.get()) == 2): count[2] += 1
            elif (int(self.label_input.get()) == 3): count[3] += 1
            elif (int(self.label_input.get()) == 4): count[4] += 1
            elif (int(self.label_input.get()) == 5): count[5] += 1
            elif (int(self.label_input.get()) == 6): count[6] += 1
            elif (int(self.label_input.get()) == 7): count[7] += 1
            elif (int(self.label_input.get()) == 8): count[8] += 1
            elif (int(self.label_input.get()) == 9): count[9] += 1
            label = int(self.label_input.get())
            self.training_data.append((image, label))
        img = image
        img = img.reshape(28,28)
        _, img = cv2.threshold(img,175,255,cv2.THRESH_BINARY)
        
        cv2.imwrite("numero5.jpg",img)

        image = tf.keras.utils.normalize(image,axis=1)
        prediction = self.model.predict([image])
        self.label.config(text=f"The number is {np.argmax(prediction)}")

        self.points.clear()
        self.canvas.delete("all")

    def normalize(self, image):
        return image / 255.0
    def train(self):
        minCount = min(count)

        #Try creating arrays
        try: 
            images, labels = zip(*self.training_data)
        except: 
            print("Training data cannot empty")
            return
        
        #Check for input
        if minCount < 1:
            print("Not enough data, each digit should have at least 1 input")
            return

        #Generating arrays with minCount inputs for each digit
        label_images = {i: [] for i in range(10)}

        for img, lbl in zip(images, labels):
            label_images[lbl].append(img)

        final_images = []
        final_labels = []

        for lbl, imgs in label_images.items():
            final_images.extend(imgs[:minCount])
            final_labels.extend([lbl]*minCount)

        final_images = np.array(final_images)
        final_labels = np.array(final_labels)

        final_images = final_images.reshape(-1, 28, 28)  
        final_images = tf.keras.utils.normalize(final_images, axis=1)

        self.model.fit(final_images, final_labels, epochs=7)
        self.model.save("best_model.h7")

        self.training_data.clear()
