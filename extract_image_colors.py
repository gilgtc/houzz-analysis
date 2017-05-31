import os
import glob
from PIL import Image
import scipy
import pandas as pd
import numpy as np
import re
from natsort import natsorted
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import scipy.cluster
import matplotlib.colors as colors
import webcolors
import sys

def create_single_image_dataframe(img_path):
    im_orig = Image.open(img_path)
    im = im_orig.resize((100, 100))      # optional, to reduce time
    ar = scipy.misc.fromimage(im)
    shape = ar.shape

    # reshape image pixel rgb information to Np x Nc, where 
    # Np = number of pixels and Nc = number of colors
    ar = ar.reshape(scipy.prod(shape[:2]), shape[2]).astype(float)
    df_pixels = pd.DataFrame(ar,columns = ['red', 'green', 'blue'])
    
    df_pixels['filenumber'] = re.search(r'\d+',img_path).group()
    df_pixels['pixel'] = df_pixels.index.values
    df_pixels.set_index(['filenumber','pixel'], inplace = True)    
    
    return df_pixels

def create_single_image_list(img_path):
    im_orig = Image.open(img_path)
    im = im_orig.resize((100, 100))      # optional, to reduce time
    ar = scipy.misc.fromimage(im)
    shape = ar.shape

    # reshape image pixel rgb information to Np x Nc, where 
    # Np = number of pixels and Nc = number of colors
    ar = ar.reshape(scipy.prod(shape[:2]), shape[2]).astype(float)
    ar[4] = re.search(r'\d+',img_path).group()
    
    df_pixels = pd.DataFrame(ar,columns = ['red', 'green', 'blue'])
    
    df_pixels['filenumber'] = re.search(r'\d+',img_path).group()
    df_pixels['pixel'] = df_pixels.index.values
    df_pixels.set_index(['filenumber','pixel'], inplace = True)    
    
    return df_pixels.values.tolist()

def create_dataframes(category, Np):
    from progress_timer import progress_timer
    folder = u'C:\\Users\\Tiger\\Documents\\Python\\houzz\\dataset'
    DIR = os.path.join(folder, category, 'sample')
    all_imgs = natsorted(glob.glob(os.path.join(DIR,'*.jpg')))
    
    df_pixels = pd.DataFrame()
    top_colors_rgb = []
    top_colors_hex = []
    top_colors_name = []
    df_pixels_list = []
    #initialize progress_timer
    #pt = progress_timer(description= 'For loop example', n_iter = Np)
    print('Creating dataframe for %s using %d images.' % (category, Np))
    for num, img_path in enumerate(all_imgs[0:Np]):
        print('On image %d of %d.' % (num + 1, Np))
        if num == 0:
            df_pixels = df_pixels.append(create_single_image_dataframe(img_path))
        
        df_pixel = create_single_image_dataframe(img_path)
        df_pixels_list.append(df_pixel)
        
        colors_hex, colors_rgb = find_top_three_colors(df_pixel.loc[:,'red':'blue'])        
        top_colors_rgb.append(colors_rgb)
        top_colors_hex.append(colors_hex)
        
        top_color_name = []
        for color_rgb in colors_rgb:
            cn = list(get_color_name(color_rgb))[1]
            top_color_name.append(cn)
        top_colors_name.append(top_color_name)
        # update progress_timer
        #pt.update()
    
    # finish progress_timer
    #pt.finish() 
    
    df_pixels = pd.concat(df_pixels_list)
    print top_colors_name
    df_top_colors_hex = pd.DataFrame(top_colors_rgb, columns = ['color%d_rgb' % x for x in range(NUM_COLORS)])
    df_top_colors_rgb = pd.DataFrame(top_colors_hex, columns = ['color%d_HEX' % x for x in range(NUM_COLORS)])
    df_top_colors_name = pd.DataFrame(top_colors_name, columns = ['color%d' % x for x in range(NUM_COLORS)])
    
    df_top_colors = df_top_colors_hex.join(df_top_colors_rgb)
    df_top_colors = df_top_colors.join(df_top_colors_name)
    df_top_colors.index.name = 'filenumber'
        
    return df_pixels, df_top_colors

def closest_color(requested_color):
    
    min_colors = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
        
    return min_colors[min(min_colors.keys())]

def get_color_name(requested_color):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
        actual_name = None
    return actual_name, closest_name

def find_top_three_colors(ar):
    NUM_CLUSTERS = NUM_COLORS
    scaler = StandardScaler()
    ar = scaler.fit_transform(ar)
    
    print 'finding clusters'
    if SCIPY:
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    else:
        k_means = cluster.KMeans(n_clusters=NUM_CLUSTERS).fit(ar)
        codes = k_means.cluster_centers_.squeeze()
    #print 'cluster centres:\n', codes
    
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
     
    top = scaler.inverse_transform(codes[np.argsort(counts)[::-1]][0:NUM_COLORS])
    ar = scaler.inverse_transform(ar)
    
    colors_hex = []
    colors_rgb = []
    
    for tt in top:    
        colors_hex.append(''.join(chr(int(c)) for c in tt).encode('hex'))
    for color in colors_hex:
        colors_rgb.append(list(colors.hex2color('#%s' % color)))
    
    return colors_hex, list(np.array(colors_rgb)*255.)

if __name__ == '__main__':        
    
    LOAD = 0
    SAVE = 1
    NUM_COLORS = 20
    if LOAD:
        df_kitchen_pixels = pd.read_pickle('kitchen_pixels.pkl')
        df_bedroom_pixels = pd.read_pickle('bedroom_pixels.pkl')
        df_bathroom_pixels = pd.read_pickle('bathroom_pixels.pkl')
        df_living_pixels = pd.read_pickle('living_pixels.pkl')
        df_kitchen_top_colors = pd.read_pickle('kitchen_top_colors.pkl')
        df_bedroom_top_colors = pd.read_pickle('bedroom_top_colors.pkl')
        df_bathroom_top_colors = pd.read_pickle('bathroom_top_colors.pkl')
        df_living_top_colors = pd.read_pickle('living_top_colors.pkl')
    else:
        cats = sys.argv[1:]
        if cats == []:
            cats = ['kitchen', 'bedroom', 'bathroom', 'living']
        Np = 500
        SCIPY = 0        
        
        for cat in cats:
            df_pixels, df_top_colors = create_dataframes(cat, Np)
            if SAVE:
                df_pixels.to_pickle('%s_pixels.pkl' % cat)
                df_top_colors.to_pickle('%s_top_colors.pkl' % cat)