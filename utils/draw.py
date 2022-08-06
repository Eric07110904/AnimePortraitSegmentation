import plotly.graph_objects as go 
import plotly.io as pio
import numpy as np 
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt 

 
def draw_loss(train_loss_record, valid_loss_record, filename):
    x = [i for i in range(len(valid_loss_record))]
    minposs = valid_loss_record.index(min(valid_loss_record))
    plt.figure()
    plt.axvline(minposs, linestyle="--", color="r", label = "Early Stopping Checkpoint")
    plt.plot(x, train_loss_record, label="Training Loss")
    plt.plot(x, valid_loss_record, label="Validation Loss")
    plt.title("Train/Valid Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    #plt.show()

