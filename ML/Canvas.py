from os import PathLike
from re import L
import matplotlib.pyplot as plt
import numpy as np

class Canvas:
    def __init__(self,name) -> None:
        self.name = name


class Canvas_Lightning:
    def __init__(self) -> None:
        pass
    
    def pie(self, name, data,labels,save_path=None):
        plt.figure()
        plt.title(name)
        explode = [0 for i in labels]
        explode_index = np.argmax(data)
        explode[explode_index] =0.5
        plt.pie(x=data,labels=labels,explode=explode,autopct='%1.1f%%')
        plt.axis('equal') 
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
    
    def single_bar(self, name, data, label, ylabel, save_path=None):
        plt.figure()
        plt.title(name)
        plt.ylabel(ylabel)
        width = 0.4
        plt.bar(range(len(data)),data,width=width,tick_label=label)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)

    def campare_bar(self, name,data,data_name,label,save_path=None):
        plt.figure()
        plt.title(name)
        width = 0.4
        data_num = len(data)
        for i in range(data_num):
            arange = np.arange(len(data[i]))+i*width
            plt.bar(arange, data[i],width=width,tick_label =label,label=data_name[i])
        plt.tight_layout()
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
    
    def plot(self, name, x,y, ylabel, save_path=None):
        plt.figure()
        plt.title(name)
        plt.ylabel(ylabel)
        plt.plot(x,y)
        plt.xticks(range(len(x)),labels=x,rotation=90)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)

    def show(self):
        plt.show()
    