from tkinter import *
import cv2
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage import morphology
from skimage.morphology import (dilation)
from scipy.ndimage import gaussian_filter




global gray_image, ix, iy, label_, label, path1, path2, path, coords

print(np.shape(gray_image))
gray_image=gray_image[1]
plt.imshow(gray_image, cmap='gray')
plt.show()

print('sigma is     ', sigma)
print('threshold is     ', threshold)


class Node:
    def __init__(self, location):
        self.location = location
        self.neighbors = []
        self.parent = None
        self.g = float('inf')
        self.f = 0

    def clear(self):
        self.parent = None
        self.g = float('inf')
        self.f = 0

    def addneighbor(self, cost, other):
        # add edge in both directions
        self.neighbors.append((cost, other))
        other.neighbors.append((cost, self))

    def __gt__(self, other):  # make nodes comparable
        return self.f > other.f

    def __repr__(self):
        return str(self.location)
        
class Graph:
    def __init__(self, grid, thresholdfactor):
        # get value that corresponds with thresholdfactor (which should be between 0 and 1)
        values = sorted([value for row in grid for value in row])
        splitvalue = values[int(len(values) * thresholdfactor)]
        # splitvalue = 0.3

        print("split at ", splitvalue)
        if splitvalue == 0:
            print("No Valid Connection")
        # simplify grid values to booleans and add extra row/col of dummy cells all around
        width = len(grid[0]) + 1
        height = len(grid) + 1
        colors = ([[False] * (width + 1)] +
            [[False] + [value < splitvalue for value in row] + [False] for row in grid] +
            [[False] * (width + 1)])

        nodes = []
        for i in range(height):
            noderow = []
            nodes.append(noderow)
            for j in range(width):
                node = Node((i, j))
                noderow.append(node)
                cells = [colors[i+1][j]] + colors[i][j:j+2]  # 3 cells around location: SW, NW, NE
                for di, dj in ((1, 0), (0, 0), (0, 1), (0, 2)):  # 4 directions: W, NW, N, NE
                    cost = 0
                    if (di + dj) % 2:  # straight
                        # if both cells are hostile, then not allowed
                        if cells[0] or cells[1]:  # at least one friendly cell
                            # if one is hostile, higher cost
                            cost = 13 if cells[0] != cells[1] else 10
                        cells.pop(0)
                    elif cells[0]:  # diagonal: cell must be friendly
                        cost = 15
                    if cost:
                        node.addneighbor(cost, nodes[i-1+di][j-1+dj])
        self.nodes = nodes

    @staticmethod
    def reconstructpath(node):
        # global path1
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
            if node==None:
                break
            node1=node.location
            # print(node1)
            path1.append(node1[0])
            path2.append(node1[1])
            # print(type(node1))

        path.reverse()
        # path1.append(path)
        return path

    @staticmethod
    def heuristic(a, b):
        # optimistic score, assuming all cells are friendly
        dy = abs(a[0] - b[0])
        dx = abs(a[1] - b[1])
        return min(dx, dy) * 15 + abs(dx - dy) * 10

    def clear(self):
        # remove search data from graph 
        for row in self.nodes:
            for node in row:
                node.clear()

    def a_star(self, start, end):
        self.clear()
        startnode = self.nodes[start[0]][start[1]]
        endnode = self.nodes[end[0]][end[1]]
        startnode.g = 0
        openlist = [startnode] 
        closed = set()
        while openlist:
            node = heappop(openlist)
            if node in closed:
                continue
            closed.add(node)
            if node == endnode:
                return self.reconstructpath(endnode)
            for weight, neighbor in node.neighbors:
                g = node.g + weight
                if g < neighbor.g:
                    neighbor.g = g
                    neighbor.f = g + self.heuristic(neighbor.location, endnode.location)
                    neighbor.parent = node
                    heappush(openlist, neighbor)
                

def onclick(event):
    # global ix, iy, label_, label, path1, path2, path, coords
    thresh_map=settings.thresh_map
    print('thresh_map is     ', settings.thresh_map)

    coords=coords
    connect=connect

    # plt.axis('off')
    print("Selected action Line or curve",connect)
    print('clicking action')
    # plt.imshow(grid, cmap='gray')
    ix, iy = event.xdata, event.ydata
    print (f'x = {ix}, y = {iy}')
    coords.append([round(iy), round(ix)])

    
    if len(coords) == 2:
        print('points selected')
        # print(coords)
        if connect == 0:
            print('Moving or Zooming')

        elif connect == 1:
            path2=path2
            path1=path1
            path2=coords[1] 
            path2.reverse()
            path1=coords[0]
            path1.reverse()
            point1=[path1[0], path2[0]]
            point2=[path1[1], path2[1]]
            ln, =plt.plot(point1, point2, linestyle="-",  color="red", linewidth=5)
            plt.show()
            coords=[]
            path2=[]
            path1=[]
            

        elif connect == 2:
            path2=path2
            path1=path1
            graph = Graph(grid, thresh_map) # provide the threshold at initialisation
            path = graph.a_star(coords[0], coords[1])
            print('in onclick')
            ln, =plt.plot(path2[:], path1[:], linestyle="-",  color="red", linewidth=5)
            plt.margins(x=0, y=0)
            plt.show()
            path2=[]
            path1=[]
        coords=[]
        print( 'label is ' + label_)        

def final_plotting ():
    global label, path1, path2, grid,  path
    # fig= plt.figure('sd')
    print(connect)
    print('plotting in action')
    # def on_scroll(event):
    #     # Get the current limits of the x and y axes
    #     cur_xlim = ax.get_xlim()
    #     cur_ylim = ax.get_ylim()
    
    #     # Define the zoom factor (adjust as needed)
    #     zoom_factor = 1.5
    
    #     # Perform zoom in or out based on the scroll direction
    #     if event.button == 'up':
    #         # Zoom in
    #         new_xlim = (cur_xlim[1] - cur_xlim[0]) / zoom_factor
    #         new_ylim = (cur_ylim[1] - cur_ylim[0]) / zoom_factor
    #     elif event.button == 'down':
    #         # Zoom out
    #         new_xlim = (cur_xlim[1] - cur_xlim[0]) * zoom_factor
    #         new_ylim = (cur_ylim[1] - cur_ylim[0]) * zoom_factor
    #     else:
    #         # Ignore other scroll directions
    #         return
    
    #     # Update the limits of the x and y axes
    #     ax.set_xlim([event.xdata - new_xlim / 2, event.xdata + new_xlim / 2])
    #     ax.set_ylim([event.ydata - new_ylim / 2, event.ydata + new_ylim / 2])
    
    #     # Redraw the figure
    #     fig.canvas.draw()
    
    fig = plt.figure()
    ax = fig.subplots()
    plt.subplots_adjust(left = 0.3, bottom = 0.25)
    # fig, ax = plt.subplots()
    ax.set_title('click on points')
    plt.imshow(gray_image, cmap='gray')
    plt.imshow(smoothed, cmap=plt.cm.viridis, alpha=.2)
    plt.axis('off')
    plt.connect('button_press_event', onclick)
    plt.show()
    coords=[]
    gray_image=gray_image

    path1=[]
    path2=[]
    path=[]
    flag=flag+1
    flag=flag 

    
   
k = frangi(255-(gray_image)) 
blurred =(gaussian_filter(k, sigma=sigma)/gray_image.max())
blurred=((blurred/blurred.max())*255).astype(int)
image_b = (blurred > threshold).astype(int)
skeleton = morphology.skeletonize(image_b).astype(np.uint8)
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(skeleton,kernel,iterations = 5)
smoothed = gaussian_filter(dilation*255, sigma=sigma)
grid = 255-smoothed
grid= grid>100

fig=plt.figure('thresholded image')
plt.imshow(grid, cmap='gray')
plt.show()
final_plotting()
