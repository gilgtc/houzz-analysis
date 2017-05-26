import pandas as pd
import numpy as np
import scipy
import matplotlib.colors as colors
import pylab as plt
from PIL import Image
import os 
from natsort import natsorted
import glob
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

def plot(cat,filenumber,df_pixels,df_top_colors):
    
    folder = u'C:\\Users\\Tiger\\Documents\\Python\\houzz\\dataset'
    DIR = os.path.join(folder, cat, 'sample')
    all_imgs = natsorted(glob.glob(os.path.join(DIR,'*.jpg')))
    im_orig = Image.open(all_imgs[filenumber])
    
    fac = 255.
    ar = df_pixels.loc[str(filenumber)]
    fs = 10
    fig = plt.figure(figsize=(15,4))
    ax1 = fig.add_subplot(131)

    ax1.imshow(im_orig)  
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(ar.iloc[:, 0], ar.iloc[:, 1],ar.iloc[:, 2], c = np.abs(ar/fac), s = 4)
    ax2.set_title('Pixel Color', fontsize=fs)
    ax2.set_xlabel('RED', fontsize=fs, color = 'red')
    ax2.set_ylabel('GREEN', fontsize=fs, color = 'green')
    ax2.set_zlabel('BLUE', fontsize=fs, color = 'blue')
    
    xl = -.1
    dist = 0.3
    
    colors_rgb = df_top_colors.iloc[filenumber,0:3]
    colors_name = df_top_colors.iloc[filenumber,6:]
    
    for color_rgb, color_name in zip(colors_rgb, colors_name):
        xl += dist
        ax3 = fig.add_subplot(133)
        color_rgba = np.array(colors.to_rgba(np.array(color_rgb)/fac))
        color_rgba_c = 1 - color_rgba
        color_rgba_c[3] = 1
        color_rgba_c = tuple(color_rgba_c)
        color_rgba = tuple(color_rgba)
        color_rgb_c = 1 - np.array(color_rgb)/fac
        
        rectangle = patches.Rectangle((xl-dist+0.1, 0.),0.3, 1., color = color_rgb_c)
        
        circle = plt.Circle((xl-0.05, 0.5), .1, color = color_rgba)
         
        ax3.add_artist(circle)
        ax3.add_patch(rectangle)
        ax3.set_aspect(.8)
        
        ax3.text(xl-dist+0.12, 0.92, color_name, color = color_rgba)
        
    ax3.axis('off')
    ax3.set_title('Top 3 colors (left to right)',y=1.1, fontsize=fs+2)

if __name__ == '__main__':  
    cats = ['kitchen', 'bedroom', 'bathroom', 'living']
    #cats = ['bedroom']

    plt.close('all')
    for cat in cats:

        df_pixels = pd.read_pickle('%s_pixels.pkl' % cat)
        df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat)

        plot(cat,10,df_pixels,df_top_colors)