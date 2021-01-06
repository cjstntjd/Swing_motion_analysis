import sys 
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import os
import random 
   
# main window 
# which inherits QDialog 
class Window(QDialog): 
       
    # constructor 
    def __init__(self, parent=None): 
        super(Window, self).__init__(parent) 
   
        # a figure instance to plot on 
        self.figure = plt.figure() 
   
        # this is the Canvas Widget that  
        # displays the 'figure'it takes the 
        # 'figure' instance as a parameter to __init__ 
        self.canvas = FigureCanvas(self.figure) 
   
        # this is the Navigation widget 
        # it takes the Canvas widget and a parent 
        self.toolbar = NavigationToolbar(self.canvas, self) 
   
        # Just some button connected to 'plot' method 
        self.button = QPushButton('Plot') 
           
        # adding action to the button 
        self.button.clicked.connect(self.plot) 
   
        # creating a Vertical Box layout 
        layout = QVBoxLayout() 
           
        # adding tool bar to the layout 
        layout.addWidget(self.toolbar) 
           
        # adding canvas to the layout 
        layout.addWidget(self.canvas) 
           
        # adding push button to the layout 
        layout.addWidget(self.button) 
           
        # setting layout to the main window 
        self.setLayout(layout) 
   
    # action called by thte push button 
    def plot(self): 
        fname = QFileDialog.getOpenFileName(self, 'Open file', "",
                                            "All Files(*);; Python Files(*.py);; CSV Files(*.csv);; Excel Files(*.xlsx)", '/home')   
        f = open(fname[0],'r')
        d = f.read()
        data = d.split('\n')
        data.pop(0)
        index = data.pop(0)
        real_data = []
        for y in range(len(data)):
            if data[y]=='':
                continue
            real_data.append(data[y].split(','))
        df = pd.DataFrame(real_data)
        index_li = index.split(',')
        df.columns = index_li
        for y in index_li:
            df[y] = pd.to_numeric(df[y],downcast='float')
        
        X = np.array(df['AX'])
        Y = np.array(df['AY'])
        Z = np.array(df['AZ'])
   
        # clearing old figure 
        self.figure.clear() 
   
        # create an axis 
        ax = self.figure.add_subplot(111,projection='3d') 
        ax.scatter(X,Y,Z)
        ax.plot(X,Y,Z)
   
        # refresh canvas 
        self.canvas.draw() 
   
# driver code 
if __name__ == '__main__': 
       
    # creating apyqt5 application 
    app = QApplication(sys.argv) 
   
    # creating a window object 
    main = Window() 
       
    # showing the window 
    main.show() 
   
    # loop 
    sys.exit(app.exec_()) 