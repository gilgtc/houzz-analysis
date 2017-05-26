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
import webcolors
import re
import matplotlib.image as image
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

def place_logo(fig):
    folder = u'C:\\Users\\Tiger\\Documents\\Python\\houzz\\dataset'    
    logo_path = os.path.join(folder, 'houzz.jpg')
    logo = Image.open(logo_path)
    logo = logo.resize((80, 80))
    #image.imread(logo_path)
    height = logo.size[1]
    logo = np.array(logo).astype(np.float) / 255
    fig.figimage(logo, 0, fig.bbox.ymax-height, zorder=0)

def format_axes(axes):
    # Remove grid lines (dotted lines inside plot)
    axes[num].grid(False)
    # Remove plot frame
    axes[num].set_frame_on(False)

    # Customize title, set position, allow space on top of plot for title
    axes[num].set_title(axes[num].get_title(), fontsize = 26, alpha = alpha, ha = 'left')
    plt.subplots_adjust(top=0.9)
    axes[num].title.set_position((0,1.2))

    # Set x axis label on top of plot, set label text
    axes[num].xaxis.set_label_position('top')
    xlab = 'Top 10 least frequent colors'
    axes[num].set_xlabel(xlab, fontsize = 16, alpha = alpha, ha = 'left')
    axes[num].xaxis.set_label_coords(0, 1.15)
    
    # Position x tick labels on top
    axes[num].xaxis.tick_top()
    # Remove tick lines in x and y axes
    axes[num].yaxis.set_ticks_position('none')
    axes[num].xaxis.set_ticks_position('none')
    xticks = axes[num].get_xticks().tolist()
    axes[num].set_xticks(xticks)
    xticks_formatted = ['{:3.2f}%'.format(x*100/Np) for x in xticks]
    axes[num].set_xticklabels(xticks_formatted, fontsize = 12, alpha = alpha)
    
    folder = u'C:\\Users\\Tiger\\Documents\\Python\\houzz\\dataset'    
    logo_path = os.path.join(folder, 'houzz.jpg')
    
cats = ['Kitchen', 'Bedroom', 'Bathroom', 'Living Room']
#cats = ['Bathroom']

plt.close('all')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,9),facecolor='#9BC643')
axes = axes.flatten()
dict_colors = dict(webcolors.css3_hex_to_names.items())
alpha = .8
for num, cat in enumerate(cats):
    
    cat_key = re.search(r'([^\s]+)', cat).group().lower()
    df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
    Np = len(df_top_colors)
    df_top_colors = df_top_colors['color1'].value_counts().iloc[0:10].iloc[::-1]
    df_top_colors_list = list(df_top_colors.index)
    df_top_colors.plot(kind = 'barh', ax=axes[num], color = df_top_colors_list,
                         title = cat)
    
    format_axes(axes)

place_logo(fig)
plt.tight_layout()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,9),facecolor='#9BC643')
axes = axes.flatten()
for num, cat in enumerate(cats):
    
    cat_key = re.search(r'([^\s]+)', cat).group().lower()
    df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
    Np = len(df_top_colors)
    df_top_colors = (df_top_colors['color1'].value_counts().iloc[-10:].iloc[::-1])
    df_top_colors_list = np.array(list(df_top_colors.index))
    df_top_colors.plot(kind = 'barh', ax=axes[num], color = df_top_colors_list,
                         title = cat)
    
    format_axes(axes)

place_logo(fig)
plt.tight_layout()

fig, axes = plt.subplots(nrows=4, ncols=1, figsize = (15,9),facecolor='#9BC643')
axes = axes.flatten()
for num, cat in enumerate(cats):
    cat_key = re.search(r'([^\s]+)', cat).group().lower()
    df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
    Np = len(df_top_colors)
    for n in range(Np):
        axes[num].plot([n,n], [0,1], color = df_top_colors['color1'].iloc[n], linewidth = 1.)
    
    axes[num].axis('off')
    axes[num].set_facecolor('black')

place_logo(fig)
plt.tight_layout()