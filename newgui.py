from tkinter import *
from tkinter import ttk, font
from PIL import Image, ImageTk, ImageDraw
import torch
from torch import zeros, nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from cnn import ConvNeuralNet, normalize, norm, binarize, binary
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

plt.ion()

model = ConvNeuralNet()
model.load_state_dict(torch.load("model.pth"))

model.eval()

def combine(x, y):
    return x + (1-x)*y

def centre(tensor):
    shape = tensor.shape
    m_00 = torch.sum(tensor)
    m_10 = sum([tensor[y][x]*x for x in range(shape[0]) for y in range(shape[1])])
    m_01 = sum([tensor[y][x]*y for x in range(shape[0]) for y in range(shape[1])])


    return (round(float(m_10/m_00)), round(float(m_01/m_00)))

def translate(tensor, vector):
    shape = tensor.shape
    x, y = vector
    new_tensor = torch.zeros(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            if row-y > -1 and col-x > -1:
                try:
                    new_tensor[row][col] = tensor[row-y][col-x]
                except IndexError:
                    new_tensor[row][col] = 0.
            else:
                new_tensor[row][col] = 0.
    return new_tensor


class SketchPad:
    def __init__(self, canvas, mpl_frame):
        self.image_tensor = zeros(28, 28, dtype=float)
        self.position_history = [None]
        self.canvas = canvas
        self.px = 1/plt.rcParams['figure.dpi']
        self.figure = Figure((280*self.px, 280*self.px))
        self.frame = mpl_frame
        self.prediction = ttk.Label(root, text=f"")
        self.prediction.grid(row = 3, pady=10)
        self.viewport = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.viewport.draw()
        self.viewport.get_tk_widget().grid(row=1, column=2, padx=20, pady=20)
        self.old_x = None
        self.old_y = None
        self.image = Image.new("L", (280,280), 0)
        self.font = font.Font(root=canvas.master, font="family", exists=False, size=70, weight="bold")
        

    def draw_dot(self, e):
        pass
        # image_tensor = self.image_tensor
        # canvas = self.canvas
        # x = ( e.x//10 )*10
        # y = ( e.y//10 )*10
        # if (x, y) != self.position_history[-1]:
        #     image_tensor[y//10][x//10] = combine(image_tensor[y//10][x//10], 1.)
        #     image_tensor[y//10 + 1][x//10] = combine(image_tensor[y//10 + 1][x//10], 1.)
        #     image_tensor[y//10][x//10 + 1] = combine(image_tensor[y//10][x//10 + 1], 1.)
        #     image_tensor[y//10 + 1][x//10 + 1] = combine(image_tensor[y//10 + 1][x//10 + 1], 1.)
            
        #     canvas.create_circle(e.x, e.y, 10)
            
    def draw_stroke(self, e):        
        if self.old_x and self.old_y:
            draw = ImageDraw.Draw(self.image)
            draw.line([self.old_x, self.old_y, e.x, e.y], fill="white", width=20, joint="curve")
            draw.circle([e.x, e.y], 10, fill="white")
            self.canvas.create_line(self.old_x, self.old_y, e.x, e.y, capstyle="round", width=20, smooth=True, fill="black")
            small_image = self.image.resize((28,28), Image.BICUBIC)
            self.image_tensor = torch.from_numpy(np.asarray(small_image))
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_tensor = zeros(28, 28, dtype=float)
        self.image = Image.new("L", (280,280), 0)


    def show_plot(self, e=None):
        self.image_tensor = self.image_tensor/255
        self.figure = Figure((280*self.px, 280*self.px))
        ax = self.figure.add_subplot()
        centroid = centre(self.image_tensor)
        self.image_tensor = translate(self.image_tensor, (14-centroid[0],14-centroid[1]))
        im = ax.imshow(self.image_tensor, cmap="viridis", vmin=0, vmax = 1)
        self.figure.colorbar(im)
        self.viewport = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.viewport.draw()
        self.viewport.get_tk_widget().grid(row=1, column=2, padx=20, pady=20)

    def predict(self, e=None):
        self.show_plot()
        tensor = self.image_tensor.reshape(1, 1, 28, 28).type(torch.float)
        tensor = binarize(tensor)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(1)
            sm = nn.Softmax()
            confidence = sm(logits)[0][pred].item()*100
            torch.set_printoptions(sci_mode=False)
            if confidence < 90:
                print(f"{sm(logits)}")
            self.prediction.grid_forget()
            self.prediction = ttk.Label(master=root, text=f"Prediction = {pred.item()} | Confidence = {confidence:.1f}%", font=self.font)
            self.prediction.grid(row = 3, pady=10)
        self.clear_canvas()

        
root = Tk()
b_frame = Frame(root)
b_frame.grid(row=2)
i_frame = Frame(root)
i_frame.grid(row=1)
canvas = Canvas(i_frame, width=280, height=280, background="white", cursor="spraycan")
sketchpad = SketchPad(canvas, i_frame)
clear = ttk.Button(b_frame, command=sketchpad.clear_canvas, text="CLEAR")
show = ttk.Button(b_frame, command=sketchpad.show_plot, text="SHOW")
predict_b = ttk.Button(b_frame, command=sketchpad.predict, text="PREDICT")
canvas.grid(row=1, column=1, padx=20, pady=20)
clear.grid(row = 2, column=1, padx=20)
show.grid(row=2, column=2, padx=20)
predict_b.grid(row=2, column=3, padx=20)

canvas.bind("<B1-Motion>", sketchpad.draw_stroke)
canvas.bind("<Button-1>", sketchpad.draw_dot)
canvas.bind("<ButtonRelease-1>", sketchpad.reset)
root.bind("<Return>", sketchpad.predict)
root.mainloop()

