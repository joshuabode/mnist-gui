from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from torch import zeros, nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from cnn import ConvNeuralNet, normalize, norm, binarize, binary

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
    print(shape)
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
        

    def draw_dot(self, e):
        image_tensor = self.image_tensor
        canvas = self.canvas
        x = ( e.x//10 )*10
        y = ( e.y//10 )*10
        if (x, y) != self.position_history[-1]:
            image_tensor[y//10][x//10] = combine(image_tensor[y//10][x//10], 1.)
            image_tensor[y//10 + 1][x//10] = combine(image_tensor[y//10 + 1][x//10], 1.)
            image_tensor[y//10][x//10 + 1] = combine(image_tensor[y//10][x//10 + 1], 1.)
            image_tensor[y//10 + 1][x//10 + 1] = combine(image_tensor[y//10 + 1][x//10 + 1], 1.)
            
            canvas.create_rectangle(x, y, x+10, y+10, fill="black", width=0)
            canvas.create_rectangle(x+10, y, x+20, y+10, fill="black", width=0)
            canvas.create_rectangle(x, y+10, x+10, y+20, fill="black", width=0)
            canvas.create_rectangle(x+10, y+10, x+20, y+20, fill="black", width=0)
            
    def draw_stroke(self, e):
        image_tensor = self.image_tensor
        canvas = self.canvas
        x = ( e.x//10 )*10
        y = ( e.y//10 )*10
        if (x, y) != self.position_history[-1]:
            image_tensor[y//10][x//10] = combine(image_tensor[y//10][x//10], 1.)
            image_tensor[y//10 + 1][x//10] = combine(image_tensor[y//10 + 1][x//10], 1.)
            image_tensor[y//10][x//10 + 1] = combine(image_tensor[y//10][x//10 + 1], 1.)
            image_tensor[y//10 + 1][x//10 + 1] = combine(image_tensor[y//10 + 1][x//10 + 1], 1.)
            
            canvas.create_rectangle(x, y, x+10, y+10, fill="black", width=0)
            canvas.create_rectangle(x+10, y, x+20, y+10, fill="black", width=0)
            canvas.create_rectangle(x, y+10, x+10, y+20, fill="black", width=0)
            canvas.create_rectangle(x+10, y+10, x+20, y+20, fill="black", width=0)

            self.position_history.append((x, y))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_tensor = zeros(28, 28, dtype=float)


    def show_plot(self, e=None):
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
            print(f"Confidence = {confidence:.1f}%") 
            torch.set_printoptions(sci_mode=False)
            print(f"{sm(logits)}")
            print(pred.item())
            self.prediction.grid_forget()
            self.prediction = ttk.Label(master=root, text=f"Prediction = {pred.item()} | Confidence = {confidence:.1f}%")
            self.prediction.grid(row = 3, pady=10)
        #self.clear_canvas()

        
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
root.bind("<Return>", sketchpad.predict)
root.mainloop()

