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
from mpl_toolkits.mplot3d import Axes3D
import hilbert

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

def top_10(exclude_colors = 0):    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,9),facecolor='#9BC643')
    axes = axes.flatten()

    for num, cat in enumerate(cats):
        
        cat_key = re.search(r'([^\s]+)', cat).group().lower()
        df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
        if exclude_colors:            
            df_top_colors = df_top_colors.query('color0 not in @exclude')
    
        Np = len(df_top_colors)
        df_top_colors = df_top_colors['color0'].value_counts().iloc[0:10].iloc[::-1]
        df_top_colors_list = list(df_top_colors.index)
        df_top_colors.plot(kind = 'barh', ax=axes[num], color = df_top_colors_list,
                             title = cat)
        
        format_axes(axes[num], Np, 'most')

    place_logo(fig)
    plt.tight_layout()
    
def bottom_10(exclude_colors = 0):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,9),facecolor='#9BC643')
    axes = axes.flatten()
    for num, cat in enumerate(cats):
        
        cat_key = re.search(r'([^\s]+)', cat).group().lower()
        df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
        
        Np = len(df_top_colors)
        df_top_colors = (df_top_colors['color0'].value_counts().iloc[-10:].iloc[::-1])
        df_top_colors_list = np.array(list(df_top_colors.index))
        df_top_colors.plot(kind = 'barh', ax=axes[num], color = df_top_colors_list,
                             title = cat)
        
        format_axes(axes[num], Np, 'most')
    
    place_logo(fig)
    plt.tight_layout()
    
def serialize_colors(col_num, exclude_colors = 0, rgb_sort = 1, real_colors = 0):
    fac = 255.
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize = (15,9),facecolor='#9BC643')
    axes = axes.flatten()
    for num, cat in enumerate(cats):
        cat_key = re.search(r'([^\s]+)', cat).group().lower()
        df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
        
        if exclude_colors:
            df_top_colors = df_top_colors.query('color0 not in @exclude')
            
        # Create new column with sum of squares of RGB values
        df_top_colors.loc[:,'rgb%d' % col_num] = df_top_colors.loc[:,'color%d_rgb' % col_num].apply(lambda x: np.sqrt(x[0]**2+x[1]**2+x[2]**2))        
        
        # Create new column with RGB transformed into Hilbert space
        df_top_colors.loc[:,'hilbert%d' % col_num] = df_top_colors.loc[:,'color%d_rgb' % col_num].apply(lambda x: hilbert.Hilbert_to_int([int(x[0]), int(x[1]), int(x[2])]))
        df_top_colors.loc[:,'hilbert_2d_%d' % col_num] = df_top_colors.loc[:,'hilbert%d' % col_num].apply(lambda x: hilbert.int_to_Hilbert( x, nD=2 ))
        print df_top_colors.loc[:,'hilbert_2d_%d' % col_num]        
        
        # Count the frequency of a particular color in 'color#' and
        # make new column with that information.
        df_top_colors.loc[:,'freq%d' % col_num] = df_top_colors.groupby('color%d' % col_num)['color%d' % col_num].transform('count')
        #df_top_colors.loc[:,'freq%d' % col_num] = df_top_colors.groupby('color%d_hex' % col_num)['color%d_hex' % col_num].transform('count')
        
        #df_top_colors = df_top_colors.sort_values('freq%d' % col_num, ascending = 0)
        #df_top_colors = df_top_colors.groupby('freq%d' % col_num).apply(pd.DataFrame.sort_values,'hilbert%d' % col_num)#.iloc[::-1]
        #df_top_colors = df_top_colors.groupby('freq%d' % col_num).apply(pd.DataFrame.sort_values,'color%d' % col_num)
        #df_top_colors = df_top_colors.groupby('hilbert%d' % col_num).apply(pd.DataFrame.sort_values,'color%d' % col_num)#.iloc[::-1]
        #df_top_colors = df_top_colors.groupby('hilbert%d' % col_num).apply(pd.DataFrame.sort_values,'color%d_hex' % col_num)#.iloc[::-1]
        #df_top_colors = df_top_colors.sort_values('hilbert%d' % col_num)
        #df_top_colors = df_top_colors.sort_values('color%d' % col_num)
        #df_top_colors = df_top_colors.sort_values('hilbert%d' % col_num)
        
        if rgb_sort:
            df_top_colors = df_top_colors.groupby('freq%d' % col_num).apply(pd.DataFrame.sort_values,'rgb%d' % col_num).iloc[::-1]        
        else:
            df_top_colors = df_top_colors.sort_values('hilbert%d' % col_num)
        Np = len(df_top_colors)
        bbox = axes[num].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width = bbox.width
        lw = width/Np*fig.dpi
        
        aa = df_top_colors.iloc[:,df_top_colors.columns.get_loc('hilbert_2d_%d' % col_num)]
        cc = df_top_colors.iloc[:,df_top_colors.columns.get_loc('color%d_rgb' % col_num)]
#        for n in range(Np):
#            x = aa[n][0]
#            y = aa[n][1]
#            plt.plot(x,y,'.', markersize = 10, color = tuple(cc[n]/255.))
        for n in range(Np):
            if real_colors:
                axes[num].plot([n,n], [0,1], color = (df_top_colors.loc[:,'color%d_rgb' % col_num].iloc[n])/fac, 
                           linewidth = lw, alpha = 1)
            else:
                axes[num].plot([n,n], [0,1], color = (df_top_colors.loc[:,'color%d' % col_num].iloc[n]), 
                           linewidth = lw, alpha = 1)
            
#            axes[num].plot(df_top_colors.loc[:,'hilbert_2d_%d' % col_num], 
#                           color = (df_top_colors.loc[:,'color%d' % col_num].iloc[n])
#            
        
        axes[num].axis('off')
        #axes[num].set_facecolor('white')
        # Customize title, set position, allow space on top of plot for title
        axes[num].set_title(cat)
        axes[num].set_title(axes[num].get_title(), fontsize = 26, alpha = alpha, ha = 'left')
        plt.subplots_adjust(top=0.9)
        axes[num].title.set_position((0.043,1.))
    
    place_logo(fig)
    plt.tight_layout()
    
def hilbert_curve_plot(col_num, exclude_colors = 0, rgb_sort = 1, real_colors = 0, is_3d = 0):
    
    fac = 255.
   
    #fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (9,9),facecolor='#9BC643')
    fig = plt.figure(figsize = (9,9),facecolor='#9BC643')
    axes = [None]*4
    for num, cat in enumerate(cats):
        axes[num] = fig.add_subplot(2, 2, num+1, projection='3d')
        cat_key = re.search(r'([^\s]+)', cat).group().lower()
        df_top_colors = pd.read_pickle('%s_top_colors.pkl' % cat_key)
        
        if exclude_colors:
            df_top_colors = df_top_colors.query('color0 not in @exclude')
            
        # Create new column with sum of squares of RGB values
        df_top_colors.loc[:,'rgb%d' % col_num] = df_top_colors.loc[:,'color%d_rgb' % col_num].apply(lambda x: np.sqrt(x[0]**2+x[1]**2+x[2]**2))        
        
        # Create new column with RGB transformed into Hilbert space
        df_top_colors.loc[:,'hilbert%d' % col_num] = df_top_colors.loc[:,'color%d_rgb' % col_num].apply(lambda x: hilbert.Hilbert_to_int([int(x[0]), int(x[1]), int(x[2])]))
        df_top_colors.loc[:,'hilbert_2d_%d' % col_num] = df_top_colors.loc[:,'hilbert%d' % col_num].apply(lambda x: hilbert.int_to_Hilbert( x, nD=2 ))
        df_top_colors.loc[:,'hilbert_3d_%d' % col_num] = df_top_colors.loc[:,'hilbert%d' % col_num].apply(lambda x: hilbert.int_to_Hilbert( x, nD=3 ))
        
        # Count the frequency of a particular color in 'color#' and
        # make new column with that information.
        df_top_colors.loc[:,'freq%d' % col_num] = df_top_colors.groupby('color%d' % col_num)['color%d' % col_num].transform('count')
        #df_top_colors.loc[:,'freq%d' % col_num] = df_top_colors.groupby('color%d_hex' % col_num)['color%d_hex' % col_num].transform('count')
        
        #df_top_colors = df_top_colors.sort_values('freq%d' % col_num, ascending = 0)
        #df_top_colors = df_top_colors.groupby('freq%d' % col_num).apply(pd.DataFrame.sort_values,'hilbert%d' % col_num)#.iloc[::-1]
        #df_top_colors = df_top_colors.groupby('freq%d' % col_num).apply(pd.DataFrame.sort_values,'color%d' % col_num)
        #df_top_colors = df_top_colors.groupby('hilbert%d' % col_num).apply(pd.DataFrame.sort_values,'color%d' % col_num)#.iloc[::-1]
        #df_top_colors = df_top_colors.groupby('hilbert%d' % col_num).apply(pd.DataFrame.sort_values,'color%d_hex' % col_num)#.iloc[::-1]
        #df_top_colors = df_top_colors.sort_values('hilbert%d' % col_num)
        #df_top_colors = df_top_colors.sort_values('color%d' % col_num)
        #df_top_colors = df_top_colors.sort_values('hilbert%d' % col_num)
        
        if rgb_sort:
            df_top_colors = df_top_colors.groupby('freq%d' % col_num).apply(pd.DataFrame.sort_values,'rgb%d' % col_num).iloc[::-1]        
        else:
            df_top_colors = df_top_colors.sort_values('hilbert%d' % col_num)
        Np = len(df_top_colors)
        bbox = axes[num].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width = bbox.width
        lw = width/Np*fig.dpi
        
        if is_3d:            
            aa = df_top_colors.iloc[:,df_top_colors.columns.get_loc('hilbert_3d_%d' % col_num)]
            cc = df_top_colors.iloc[:,df_top_colors.columns.get_loc('color%d_rgb' % col_num)]
            x, y, z = [], [], []
            print aa.head()
            print cc.head()
            for n in range(Np):
                x.append(cc[n][0])
                y.append(cc[n][1])
                z.append(cc[n][2])
            #axes[num] = fig.add_subplot(132, projection='3d')
            axes[num].scatter(x,y,z, c = cc/255., s = 4)
            axes[num].set_title('Pixel Color', fontsize = 12)
            #axes[num].set_xlabel('RED', fontsize=fs)
            #axes[num].set_ylabel('GREEN', fontsize=fs)
            #axes[num].set_zlabel('BLUE', fontsize=fs)
            #axes[num].plot(x,y,'.', markersize = 10, color = tuple(cc[n]/255.))
            
            
            axes[num].axis('off')
            #axes[num].set_facecolor('black')
    #        # Customize title, set position, allow space on top of plot for title
            axes[num].set_title(cat)
            axes[num].set_title(axes[num].get_title(), fontsize = 26, alpha = alpha, ha = 'left')
            #plt.subplots_adjust(top=0.9)
            axes[num].title.set_position((0.043,1.))
        else:
            aa = df_top_colors.iloc[:,df_top_colors.columns.get_loc('hilbert_2d_%d' % col_num)]
            cc = df_top_colors.iloc[:,df_top_colors.columns.get_loc('color%d_rgb' % col_num)]
            for n in range(Np):
                x = aa[n][0]
                y = aa[n][1]
                axes[num].plot(x,y,'.', markersize = 10, color = tuple(cc[n]/255.))
    #        
            
            axes[num].axis('off')
            #axes[num].set_facecolor('black')
    #        # Customize title, set position, allow space on top of plot for title
            axes[num].set_title(cat)
            axes[num].set_title(axes[num].get_title(), fontsize = 26, alpha = alpha, ha = 'left')
            plt.subplots_adjust(top=0.9)
            axes[num].title.set_position((0.043,1.))
    place_logo(fig)
    plt.tight_layout()

exclude = ['ghostwhite','white','snow','silver', 'darkgray', 'gainsboro', 'grey', 'lightgray', 'black', 
                     'whitesmoke', 'dimgrey', 'lightslategrey',
                     'sienna', 'floralwhite', 'linen', 'lavender','antiquewhite','beige']    
cats = ['Kitchen', 'Bedroom', 'Bathroom', 'Living Room']
#cats = ['Bathroom']
dict_colors = dict(webcolors.css3_hex_to_names.items())
alpha = .8

plt.close('all')
#top_10(1)
#bottom_10()
serialize_colors(1,0,1,0)
#hilbert_curve_plot(1,0,0,1,1)
#serialize_colors(2,1)
#serialize_colors(3,1)

