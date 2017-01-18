import numpy as np
from os import name

if "posix" in name:
    from matplotlib import use
    use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.mlab import dist_point_to_segment
from copy import deepcopy

class SegmentationGUI(object):

    _modes = "plot"  #plot: place landmarks, connect: connect landmarks

    _usage_title = {True: "LEFT: add landmark, RIGHT: delete landmark\n"
                          "Press 'm' to switch modes",
                    False: "'i': insert, 'RIGHT': delete\n"
                           "Press 'Enter' to crop, 'm' to switch modes"}
 
    _alpha = 0.30
    _ind = None #active vertex index
    _showverts = True
    _epsilon = 5
    
    #artist objects
    _line = None
    _plot = None
    _poly = None
    _verts = None    #plotted landmarks
    _background = None
    
    _img = None      #original image slice
    _mask = None     #resulting 2d bin mask
    
    fig, ax, canvas = None, None, None

    def __init__(self, img, mask, verts, title):
        self._img = img    
        self._mask = mask
        self._verts = verts
        
        """initializing GUI"""
        #define axis and corresponding figure img falls under
        self.fig, self.ax = plt.subplots()
        self.canvas = self.fig.canvas
        
        #load image onto the axis
        self.ax.imshow(img, cmap='gray')
        self.ax.set_xlim([0., img.shape[1]])
        self.ax.set_ylim([img.shape[0], 0.])
        self.ax.autoscale = False
        #preventing plot from clearing image
        self.ax.hold(True)
        
        self.initialize_verts()
        self.initialize_plot()
        self.connect_activity()

        return


    def initialize_verts(self):
        return 


    def initialize_plot(self):
        self._plot = self.ax.plot([], [], marker='o', markerfacecolor='b', 
                                        linestyle='none', markersize=5)[0]
        self.replot()
        self.canvas.draw()


    def connect_activity(self):
        self.canvas.mpl_connect('button_press_event',  
                                 self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', 
                                 self.button_release_callback)
        self.canvas.mpl_connect('scroll_event', 
                                 self.scroll_callback)
        self.canvas.mpl_connect('motion_notify_event', 
                                 self.motion_notify_callback)
        self.canvas.mpl_connect('draw_event', 
                                 self.draw_callback)
        self.canvas.mpl_connect('key_press_event', 
                                 self.key_press_callback)

    
    def button_press_callback(self, event):
        if not self._showverts: 
            return
        if not event.inaxes: 
            return

        self._ind = self.get_nearest_vertex_idx(event)
        # Do whichever action corresponds to the mouse button clicked
        if event.button == 1:
            self.add_vertex(event)
        elif event.button == 3:
            self.remove_vertex(event)
        # Re-plot the landmarks on canvas
        self.replot()
        self.canvas.draw()


    def button_release_callback(self, event):
        if not self._showverts:
            return

        self._ind = None


    def scroll_callback(self, event):
        if not self._showverts: 
            return
        if self._modes == "plot": 
            return

        if event.button == "up":
            if self._alpha < 1.00:
                self._alpha += 0.05
        elif event.button == "down":
            self._alpha -= 0.05
            if self._alpha <= 0.00:
                self._alpha = 0.00

        # self.ax.set_ylabel("Alpha: %.2f" % self.alpha)
        self._poly.set_alpha(self._alpha)
        self.canvas.draw()


    def motion_notify_callback(self, event):
        #on mouse movement
        if not self._showverts: return
        if not event.inaxes: return
        if event.button != 1: return
        if self._ind is None: return

        self.move_vertex_to(event)
        self.canvas.restore_region(self._background)
        self.redraw()


    def draw_callback(self, event):
        if self._modes == "connect":
            self._background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.redraw()


    def key_press_callback(self, event):
        if not event.inaxes:
            return

        if event.key == 't':
            self.switch_vis()
        elif event.key == 'm':
            self.switch_modes()
        elif event.key == 'i':
            self.insert_vertex(event)
        elif event.key == 'enter':
            self.poly2mask()
        self.canvas.draw()

 
    def switch_vis(self):
        if self._modes == "plot":
            return

        self._showverts = not self._showverts
        if not self._showverts:
            self._line.set_marker(None)
            self._ind = None
        else:
            self._line.set_marker('o')

 
    def add_vertex(self, event):
        # Adds a point at cursor
        if self._modes == "connect":
            return
        if self._modes == "plot":
            self._verts.append((event.xdata, event.ydata))


    def insert_vertex(self, event):
        if self._modes == "plot": 
            return
        if not self._showverts: 
            return

        p = event.xdata, event.ydata  # display coords
        mod = len(self._verts)
        for i in range(len(self._verts)):
            s0 = self._verts[i%mod]
            s1 = self._verts[(i + 1)%mod]
            d = dist_point_to_segment(p, s0, s1)
            if d <= 5:
                self._poly.xy = np.array(
                    list(self._poly.xy[:i+1]) +
                    [(event.xdata, event.ydata)] +
                    list(self._poly.xy[i+1:]))
                self._line.set_data(zip(*self._poly.xy))
                self._verts = [tup for i, tup in enumerate(self._poly.xy) if i != len(self._poly.xy)-1]
                break


    def remove_vertex(self, event):
        # Removes the point closest to the cursor
        index = self._ind
        if not index is None:
            del self._verts[index]
            if self._modes == "connect":
                if len(self._verts) <= 1:
                    self.switch_modes()
                else:
                    self._poly.xy = [x for x in self._verts]
                    self._line.set_data(zip(*self._poly.xy))


    def get_nearest_vertex_idx(self, event):
        if len(self._verts) > 0:
            distance = [(v[0] - event.xdata) ** 2 +
                        (v[1] - event.ydata) ** 2 for v in self._verts]
            if np.sqrt(min(distance)) <= self._epsilon:
                return distance.index(min(distance))
        return None


    def move_vertex_to(self, event):
        x, y = event.xdata, event.ydata
        self._poly.xy[self._ind] = x, y
        self._verts[self._ind] = x, y
        if self._ind == 0:
            self._poly.xy[-1] = self._poly.xy[self._ind]
        self._line.set_data(zip(*self._poly.xy))


    def switch_modes(self):
        if not self._showverts:
            return

        if self._modes == "plot":
            self.switch2poly()
        elif self._modes == "connect":
            self.switch2plot()


    def switch2plot(self):
        self._modes = "plot"
        #self.ax.set_title(self.title[True])
        #self.ax.set_ylabel("")

        self.replot()
        if self._poly:
            self._poly.xy = [(0, 0)]


    def switch2poly(self):
        if len(self._verts) <= 1:
                raise AttributeError("Requires 2 or more vertices to draw region")
        self._modes = "connect"
        #self.ax.set_title(self.title[False])
        #self.ax.set_ylabel("Alpha: %.2f" % self.alpha)

        #self.verts_sort()
        if self._poly is None:
            self.create_polygon()
        else:
            self._poly.xy = np.array(self._verts[:])
            self._line.set_data(zip(*self._poly.xy))
        self._plot.set_data([],[])


    def poly2mask(self):
        if self._modes == "plot": 
            return
        # if not self.verts: return

        #self.covered_pixels = []
        for x in range(self._img.shape[1]):
            for y in range(self._img.shape[0]):
                if self._poly.get_path().contains_point((x,y)):
                    #self.covered_pixels.append((x,y))
                    self._mask[y][x] = 1
                else:
                    self._mask[y][x] = 0
        #plt.close()
        self._showverts = False


    def replot(self):
        # Apply the changes to the vertices / canvas
        if len(self._verts) > 0:
            x, y = zip(*self._verts)
        else:
            x, y = [], []

        if self._modes == "plot":
            self._plot.set_xdata(x)
            self._plot.set_ydata(y)


    def redraw(self):
        self.ax.draw_artist(self._poly)
        self.ax.draw_artist(self._line)
        self.canvas.blit(self.ax.bbox)


    def create_polygon(self):
        self._poly = Polygon(self._verts, 
                             animated=True, alpha=self._alpha)
        self.ax.add_patch(self._poly)

        x, y = zip(*self._poly.xy)
        self._line = Line2D(x, y, marker='o',
                           markerfacecolor='r', animated=True, markersize=5)
        self.ax.add_line(self._line)


    def return_mask(self):
        return self._mask


    def return_verts(self):
        return self._verts

    
    def return_img(self):
        return self._img


def manual_segmentation(img, **kwargs):
    
    if img is None:
        raise AttributeError("User must provide <numpy.ndarray> image")
    
    xlen = img.shape[1]
    ylen = img.shape[0]
   
    mask = deepcopy(kwargs.pop("mask", np.zeros((ylen, xlen))))
    verts = deepcopy(kwargs.pop("verts", []))
    title = deepcopy(kwargs.pop("title", None))
    
    gui = SegmentationGUI(img, mask, verts, title)
    plt.show()

    mask = gui.return_mask()
    verts = gui.return_verts()

    return mask


def main():

    test_img = np.zeros((256, 256))
    binmask = manual_segmentation(test_img)
    
    fig, ax = plt.subplots()
    #load image onto the axis
    ax.imshow(binmask, cmap='gray')
    ax.set_xlim([0., binmask.shape[1]])
    ax.set_ylim([binmask.shape[0], 0.])
    ax.autoscale = False
    #preventing plot from clearing image
    ax.hold(True)

    plt.show()
    return

if __name__ == '__main__':
    main()
