import numpy as np
from os import name

if "posix" in name:
    from matplotlib import use
    use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.mlab import dist_point_to_segment


class SegmentationGUI(object):

    _modes = ["plot",
              "connect"] # plot: place landmarks, connect: connect landmarks
  
    _alpha = 0.30
    _ind = None #active vertex index
  
    #artist objects
    line = None
    plot = None
    poly = None
    verts = None    #plotted landmarks

    img = None      #original image slice
    mask = None     #resulting 2d bin mask
    
    def __init__(self):
        return

    def return_mask(self):
        return self.mask

def manual_segmentation(img, verts=[], title=""):
    
    if img is None:
        raise AttributeError("User must provide <numpy.ndarray> image")
    
    gui = None
    fig, ax = None, None

    def initialize_GUI():
        xlen = img.shape[1]
        ylen = img.shape[0]
        
        #define axis and corresponding figure img falls under
        fig, ax = plt.subplots()
        #load image onto the axis
        ax.imshow(img, cmap='gray')
        ax.set_xlim([0., xlen])
        ax.set_ylim([ylen, 0.])
        ax.autoscale = False
        # preventing plot from clearing image
        ax.hold(True)
        
        gui = SegmentationGUI()

        return

    initialize_GUI()
    plt.show()
    
    return gui.return_mask()


def main():

    test_img = np.zeros((256, 256))
    binmask = manual_segmentation(test_img)

    return

if __name__ == '__main__':
    main()
