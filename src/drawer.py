import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import network

class DigitDrawer:

    def __init__(self, network):
        self.network = network

        #give it a window
        self.window = tk.Tk()
        self.window.title("Draw A Digit")

        #set the canvas specs
        self.canvas = tk.Canvas(self.window, width = 280, height = 280, background='black')
        self.canvas.pack()

        #create the PIL image to feed to the ai
        self.image = Image.new('L', (28, 28), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.pixels = [[0 for _ in range(28)] for _ in range(28)]

        #set up mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        #set up the buttons and their functions
        button_frame = tk.Frame(self.window)
        button_frame.pack()
        tk.Button(button_frame, text="Save & Predict", 
                 command=self.save_and_predict).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Clear", 
                 command=self.clear_canvas).pack(side=tk.LEFT)
        
        #the text to show off the prediction that the ai made
        self.prediction_label = tk.Label(self.window, text="Draw a digit and click Save", font=('Arial', 16))
        self.prediction_label.pack()

        #points to be used in other functions
        self.last_x = None
        self.last_y = None

    def paint(self, event):
        x, y = event.x, event.y

        #scale down for the ai to use
        img_x = x // 10
        img_y = y // 10

        #do the actual draw
        if 0 <=img_x < 28 and 0 <= img_y < 28:
            # Mark this pixel as drawn
            self.pixels[img_y][img_x] = 1

            #update the PIL image
            self.draw.point((img_x, img_y), fill='black')  # Add this line to update the PIL image
            
            # Draw a white square on the canvas
            self.canvas.create_rectangle(img_x*10, img_y*10, img_x*10+10, img_y*10+10, 
                                        fill='white', outline='')

    def save_and_predict(self):
        #save the image and prepare it for calc
        inverted = ImageOps.invert(self.image)

        save_path = "my_digit.png"
        inverted.save(save_path)
        print(f"image saved to {save_path}")

        #convert to a numpy array so the ai can lowkey read that 
        img_array = np.array(inverted).reshape(784, 1) / 255.0

        predication = np.argmax(self.predict_with_network(img_array))
        self.prediction_label.config(text=f"The AI predicts: {predication}")

    def clear_canvas(self):
        #clear canvas and reset the label
        self.canvas.delete('all')
        self.image = Image.new('L', (28, 28), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit and click Save")

    def predict_with_network(self, image_array):
        #run the image data through the network
        return self.network.feedforward(image_array)

    def run(self):
        self.window.mainloop()


