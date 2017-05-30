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

def place_logo(fig):
    folder = u'C:\\Users\\Tiger\\Documents\\Python\\houzz\\dataset'    
    logo_path = os.path.join(folder, 'houzz.jpg')
    logo = Image.open(logo_path)
    logo = logo.resize((80, 80))
    #image.imread(logo_path)
    height = logo.size[1]
    logo = np.array(logo).astype(np.float) / 255
    fig.figimage(logo, 0, fig.bbox.ymax-height, zorder=0)
    
def format_axes(ax, Np, arrang):
    alpha = 0.9
    # Remove grid lines (dotted lines inside plot)
    ax.grid(False)
    # Remove plot frame
    ax.set_frame_on(False)

    # Customize title, set position, allow space on top of plot for title
    ax.set_title(ax.get_title(), fontsize = 26, alpha = alpha, ha = 'left')
    plt.subplots_adjust(top=0.9)
    ax.title.set_position((0,1.2))

    # Set x axis label on top of plot, set label text
    ax.xaxis.set_label_position('top')
    xlab = 'Top 10 %s frequent colors' % arrang
    ax.set_xlabel(xlab, fontsize = 16, alpha = alpha, ha = 'left')
    ax.xaxis.set_label_coords(0, 1.15)
    
    # Position x tick labels on top
    ax.xaxis.tick_top()
    # Remove tick lines in x and y axes
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    xticks = ax.get_xticks().tolist()
    ax.set_xticks(xticks)
    xticks_formatted = ['{:3.2f}%'.format(x*100/Np) for x in xticks]
    ax.set_xticklabels(xticks_formatted, fontsize = 12, alpha = alpha)
    
def plot(cat,filenumber,df_pixels,df_top_colors):
    
    folder = u'C:\\Users\\Tiger\\Documents\\Python\\houzz\\dataset'
    DIR = os.path.join(folder, cat, 'sample')
    all_imgs = natsorted(glob.glob(os.path.join(DIR,'*.jpg')))
    im_orig = Image.open(all_imgs[filenumber])
    
    fac = 255.
    ar = df_pixels.loc[str(filenumber)]
    fs = 10
    fig = plt.figure(figsize=(10,4),facecolor='#9BC643')
    ax1 = fig.add_subplot(131)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(im_orig)  
    ax2 = fig.add_subplot(132, projection='3d', axisbg='#9BC643')
    ax2.scatter(ar.iloc[:, 0], ar.iloc[:, 1],ar.iloc[:, 2], c = np.abs(ar/fac), s = 4)
    ax2.set_title('Pixel Color Distribution',y=1.1, fontsize=fs+2)
    ax2.set_xlabel('RED', fontsize=fs, color = 'red')
    ax2.set_ylabel('GREEN', fontsize=fs, color = 'green')
    ax2.set_zlabel('BLUE', fontsize=fs, color = 'blue')
    ax2.grid(False)
    ax2.axis('off')
    xl = -.1
    dist = 0.3
    
    colors_rgb = df_top_colors.iloc[filenumber,0:3]
    colors_name = df_top_colors.loc[filenumber,'color0':'color2']
    print colors_name
    
    for color_rgb, color_name in zip(colors_rgb, colors_name):
        xl += dist
        ax3 = fig.add_subplot(133)
        color_rgba = np.array(colors.to_rgba(np.array(color_rgb)/fac))
        color_rgba_c = 1 - color_rgba
        color_rgba_c[3] = 1
        color_rgba_c = tuple(color_rgba_c)
        color_rgba = tuple(color_rgba)
        color_rgb_c = 1 - np.array(color_rgb)/fac
        
        rectangle = patches.Rectangle((xl-dist+0.1, 0.),0.3, 1., color = '#9BC643')
        
        circle = plt.Circle((xl-0.05, 0.5), .1, color = color_rgba)
         
        ax3.add_artist(circle)
        ax3.add_patch(rectangle)
        ax3.set_aspect(.8)
        
        #ax3.text(xl-dist+0.12, 0.92, color_name, color = color_rgba)
        
    ax3.axis('off')
    ax3.set_title('Top 3 colors (left to right)',y=1.1, fontsize=fs+2)
    place_logo(fig)
if __name__ == '__main__':  
    cats = ['kitchen', 'bedroom', 'bathroom', 'living']
    #cats = ['bedroom']

    plt.close('all')
    for cat in cats:

        df_pixels = pd.read_pickle('%s_pixels.pkl' % cat)
        df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat)

        plot(cat,43,df_pixels,df_top_colors)