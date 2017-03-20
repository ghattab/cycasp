#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series # for convenience
import pims
import trackpy as tp
import warnings
import cv2
import json
import networkx as nx

from operator import itemgetter
from itertools import combinations, product, groupby
from scipy.spatial import Delaunay, ConvexHull
from networkx.algorithms.components.connected import connected_components
from networkx.readwrite import json_graph



def preprocess(idir):
    ''' Loads images then runs the preprocessing part of CYCASP
        idir: path containing the 3 channels (RGB)
        exports the resulting image into the default seevis_output folder
    '''
    # load imgs
    f_red, f_green, f_blue = get_imlist(idir)
    print "\n%d files found\t" %(len(f_red)*3)
    print "Loading data..."
    red, size_var = load_img(f_red)
    # 1st frame properties (rows, cols)/(height, width)
    rows, cols, channels = size_var.shape
    print "Image size ", size_var.shape
    green, sv = load_img(f_green)
    blue, sv = load_img(f_blue)
    # enhancing the image
    rgb, ug, uclahe, ctrast, \
    dblur, tmask, res = approach(red, blue, green)
    # default export
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outdir = timestr+"cycasp_output"
    out = [res]; dirs = [outdir]
    export(f_red, dirs, out)
    return outdir, red, green, blue


def get_data(outdir, red, green, blue, diam=11):
    ''' Loads the output of the preprocessing steps for particle extraction
        Returns the formatted data
    '''
    frames = pims.ImageSequence("../"+outdir+"/*tif")
    print frames

    # particle diameter
    features = tp.batch(frames[:frames._count], diameter=diam, \
                        minmass=1, invert=True)

    # Link features in time
    search_range = diam-2 # sigma_(max)

    lframes = int(np.floor(frames._count/3)) # r, g, b images are loaded
    imax = int(np.floor(15*lframes/100)) # default max 15% frame count
    t = tp.link_df(features, search_range, memory=imax)
    # default neighbour strategy: KDTree

    # Filter spurious trajectories
    imin = int(np.floor(10*lframes/100)) # default min 10% frame count
    t1 = tp.filter_stubs(t, imin) # if seen in imin
    
    # Compare the number of particles in the unfiltered and filtered data
    print 'Unique number of particles (Before filtering):', t['particle'].nunique()
    print '(After):', t1['particle'].nunique()

    # export pandas data frame with filename being current date and time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    data = pd.DataFrame({ 'x': t1.x, 'y': t1.y,'z':t1.frame,\
                        'mass':t1.mass, 'size':t1.size, 'ecc':t1.ecc,\
                        'signal':t1.signal, 'ep':t1.ep, 'particle':t1.particle\
                        })
    
    # format the dataframe / original indexing
    data["n"] = np.arange(len(data))
    
    print("Sorting dataframe by time...")
    data = data.sort(columns='z', ascending=True)

    print("Extracting pixel values of particles...")
    r, g, b = get_val(red, 2, data), get_val(green, 1, data), get_val(blue, 0, data)

    print("Normalising rgb values to relative quantities...")
    r1, g1, b1 = np.array(r), np.array(g), np.array(b)
    r = (r1-np.min(r1))*(65535/np.max(r1))
    g = (g1-np.min(g1))*(65535/np.max(g1))
    b = (b1-np.min(b1))*(65535/np.max(b1))

    print("Adding (r,g,b) values as columns to dataframe...")
    strname, px_val = ["r", "g", "b"], [r, g, b]
    add_arrays_df(strname, px_val, data)

    # sort back to original state
    data = data.sort(columns='n', ascending=True)
    
    # remove the previously created column
    data.drop('n', axis=1, inplace=True)
    
    # format df with rgb values to uint8
    data = format_df(data)
    
    print "Dataframe summary:\n", data.describe()
    file_name = "../particles_"+timestr+".csv"; print "Exporting %s" %(file_name)
    data.to_csv(file_name, sep='\t', encoding='utf-8')
    
    return data


def format_df(d):
    ''' Formats and cleans the dataframe d
        Args    d dataframe
        Returns modified d with 4 new columns (r8, g8, b8 and n)
    '''
    # Rename several DataFrame columns for uint16
    d = d.rename(columns = {
        'r':'r16',
        'g':'g16',
        'b':'b16',
    })
    # Create new columns for uint8 for easier manipulation (from uint16)
    #uint8  0 -- 255
    #uint16 0 -- 65535
    d["r8"] = (d.r16.values/256)
    d["g8"] = (d.g16.values/256)
    d["b8"] = (d.b16.values/256)
    # Remove particles that have no colour information for all r, g and b
    selection = d[(d.r8.values == 0) & (d.g8.values==0) & (d.b8.values==0)].particle.unique()
    print "Colour filtering: %s particle trajectories removed" %len(selection)
    # returned particle IDs are used to delete corresponding trajectories
    for p in selection: d = d[d.particle != int(p)]
    d["n"] = np.zeros(len(d)) # new column n in d
    return d


def get_px(array, c, ind):
    ''' Extract at a time point t all coordinates from a specific c channel
        Args    an array of the format t, x, y
                c, list of images pertaining to one channel
                ind, the index for image slicing for that particular channel
                example for red channel (format BGR) ind = 2
        Returns list l of the values per pixel coord
    '''
    l = []
    for t, x, y in array:
        l.append(c[t][...,ind][x, y])
    return l


def get_val(c, ind, data):
    ''' Map get_px procedure to return n lists per channel
    '''
    l = []; a = np.column_stack([data.z, np.floor(data.x), np.floor(data.y)])
    for i in xrange(len(c)):
        l.extend(get_px(a[np.where(a[...,0] == i)].astype(int), c, ind))
    return l


def add_arrays_df(strname, list, data):
    ''' Returns the updated dataframe with 3 new columns (header strname+i)
        Converts individual lists to arrays then adds to pandas df
        Returns dataframe
    '''
    for i in xrange(3):
        data[strname[i]] = np.array(list[i])
    return data


def elapsed_time(start_time):
    print("\t--- %4s seconds ---\n" %(time.time()-start_time))


def create_dir(dir):
    ''' Directory creation
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
    print "Directory '%s' created" %(str(dir))


def get_imlist(path):
    ''' Returns a list of filenames for all compatible extensions in a directory
	    Args : path of the directory and the supplied user choice
        Handles a directory containing i_ext as extension and 
        c2, c3, c4 as red, green, blue channels respectively
		Returns the list of files to be treated
    '''
    i_ext = tuple([".tif",".jpg",".jpeg",".png"])
    flist = [os.path.join(path, f) for f in os.listdir(path) \
         if f.lower().endswith(i_ext)]
    if len(flist) != None:
        ext = str.split(flist[0],".")[-1]
        f_red = [ r for r in flist if r.endswith("c2"+"."+ext) ]
        f_green = [ g for g in flist if g.endswith("c3"+"."+ext) ]
        f_blue = [ b for b in flist if b.endswith("c4"+"."+ext) ]
        if len(f_red) == None or len(f_red) == 0:
            print "error: image filenames do not comply. Image filenames must be formatted as follows: red, c1.tif; )"
            sys.exit(1)
        return f_red, f_green, f_blue
    else:
        warnings.filterwarnings("ignore")
        print 'Directory contains unsupported files. Please refer to the README file)'
        sys.exit(1)


def load_img(flist):
    ''' Loads images in a list of arrays
	    Args : list of files
	    Returns list of all the ndimage arrays
    '''
    imgs = []
    for i in flist:
        imgs.append(cv2.imread(i, -1)) # return img as is
    size_var = cv2.imread(i)
    return imgs, size_var


def rgb_to_gray(img):
    ''' Converts an RGB image to greyscale, where each pixel
        now represents the intensity of the original image.
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def invert(img):
    ''' Inverts an image
    '''
    cimg = np.copy(img)
    return cv2.bitwise_not(cimg, cimg)


def approach(red, blue, green):
    ''' The preprocessing steps
        Args    red, blue, green channels
        Returns 8 different variables highlighting the most important steps
    '''
    rgb, fgray, ug, uclahe, ctrast, dblur, mblur, tmask, res, res2 = [], [], [], [], [], [], [], [], [], []
    
    print "[Init.] Preprocessing images..."
    
    maxIntensity, phi, theta = 255.0, 1, 1 # parameters for global contrast
    x = np.arange(maxIntensity) # max Intensity depends on dtype of image data

    for i in range(len(red)):
        # Add up 3C into RGB (default mode is in BGR)
        red[i][...,[0,1]], green[i][...,[0,2]], blue[i][...,[1,2]] = 0, 0, 0
        tmp_rgb = red[i] + green[i] + blue[i]
        rgb.append(tmp_rgb)
        
        # 1. signal enhancement
        ###########################################
        # rgb to gray
        tmp_gray = rgb_to_gray(tmp_rgb)
        fgray.append(tmp_gray)
        # uint16 to inverted ubyte
        tmp_ug = invert(ubyte(tmp_gray))
        ug.append(tmp_ug)
        # contrast limited histogram equalisation
        gclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        tmp_uclahe = gclahe.apply(tmp_ug)
        tmp_uclahe = np.array(tmp_uclahe, dtype="uint8")
        uclahe.append(tmp_uclahe)
        # global contrast enhancement
        tmp_ctrast = (maxIntensity/phi)*(tmp_uclahe/(maxIntensity/theta))**2
        tmp_ctrast = np.array(tmp_ctrast, dtype="uint8")
        ctrast.append(tmp_ctrast)
        
        # 2. subtract single bacterial signal
        ###########################################
        # bilateral filtering
        sblur = cv2.bilateralFilter(tmp_ctrast, 5, 75, 75)
        dblur.append(sblur)
        # Adaptive thresholding
        tmp_ssignal = cv2.adaptiveThreshold(sblur, 255, \
                                    cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 15, 2)
        tmask.append(tmp_ssignal)
        
        # 3. adaptively mask the signal region
        ###########################################
        # median blurring (rblur for FG signal region)
        rblur = cv2.medianBlur(tmp_ctrast, 15)
        mblur.append(rblur)
        # binary thresholding
        ret, thresh2 = cv2.threshold(rblur, 225, 255, cv2.THRESH_BINARY)
        # background/foreground masking
        mask = np.ones(red[0].shape[:2], dtype="uint8")*255
        tmp_res = cv2.bitwise_or(thresh2, tmp_ssignal, mask=mask)
        res.append(tmp_res)

    return rgb, ug, uclahe, ctrast, dblur, tmask, res


def export(flist, dirs, out):
    ''' Enumerates elements in dirs and formats the filename depending on flist
    '''
    for j in enumerate(dirs):
        create_dir("../"+j[1]) # create all dirs
        for i in range(len(flist)):
            f = "../"+j[1]+ "/" +j[1].split(" - ")[-1]+ "_" + flist[i].split("/")[-1]
            tmp = out[j[0]][i]
            cv2.imwrite(f, tmp)


def get_cmap(N, map):
    ''' Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color. Uses a colour palette and Mappable scalar
    '''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=map)
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def ubyte(img):
    return cv2.convertScaleAbs(img, alpha=(255.0/65535.0))


def find_missing(integers_list, start=None, limit=None):
    ''' Given a list of integers and optionally a start and an end, finds all
        the integers from start to end that are not in the list
    '''
    start = start if start is not None else integers_list[0]
    limit = limit if limit is not None else integers_list[-1]
    return [i for i in range(start,limit + 1) if i not in integers_list]


def load_data(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True, sep='\t')
    data = pd.DataFrame({ 'x': data.x, 'y': data.y,'z':data.z,\
                        'mass':data.mass, 'size':data.size, 'data':data.ecc,\
                        'signal':data.signal, 'ep':data.ep, 'particle':data.particle\
                        })
    return data


def modalgo(d, euclid=10, ru=50, gu=50, bu=50, t=10):
    ''' fun. wrap the initialisation, splitting, merging of patches
        Args    d particle diameter
                euclid geometrical distance default 10 px
                ru, gu, bu rgb specific channel diff. default 50(absolute value)
                t merge time window. default 10 frames.
        Returns d dataframe and G graph
    '''
    # Proximity/colour based filtering
    tm = np.max(d.z.values) #tm or time point max
    G = init(d, tm, distu, ru, gu, bu)

    # Incremental assignment of particle tracks that have no patch ids
    G = increment(G, d, tm, distu, ru, gu, bu)

    # read from d to get patches return lists of patch ids and particle ids
    update_graph(d, G)

    m=nx.get_node_attributes(G, 'p').keys()
    print "Patches span over %s time points with info. from %s nodes" %(tm+1, len(m))

    ct = [i for i in G.nodes() if (G.node[i]=={'id':0})==False]
    for i in ct:
        if len(G.node[i]['id'])!=len(G.node[i]['pb']):
            print i, len(G.node[i]['id'])!=len(G.node[i]['pb'])

    print "[Init.] Computing merges between patches"
    candidates = find_merges(G, d, tm, distu, ru, gu, bu)
    if not(empty(candidates))==True: print "[Done] Merge candidates list created"
    # iff candidate merge stay merged at least j consecutive frames do:

    ppairs = true_merge(candidates, t) # list of patch ids pair
    # update graph and dataframe
    d, G = post_merge(d, G, ppairs)
    return d, G


def init(d, tm, distu, ru, gu, bu):
    ''' Filters all particles at last time based on user parameters
        Create patches, assign incrementally through time and particle tracks
        the patch information in column n then build up the graph data structure
        Args    d dataframe
                tm time point max
                distu euclidean distance
                ru, gu, bu absolute difference in each colourspace
        Returns G graph
    '''
    print"[Init.] Finding patches at t_max"
    # res returns all the possible particle combinations at tm
    res = combine(tm, d)
    # display first 5 elements
    res.head()
    print "[Init.] Graph with %s nodes" %(tm+1)
    G = nx.path_graph(tm+1) # create a graph with number of time points + 1
    # filter all interactions by user values
    l, rest = pfilter(res, tm, distu, ru, gu, bu)
    #l = [['a','b','c'],['b','d','e'],['k'],['o','p'],['e','f'],['p','a'],['d','g']]
    ls = graphcc(l) #returns a sorted connected components' list of particle ids
    pvl = patch_encode(d, ls, 0)
    verify_patch(d, ls, distu, ru, gu, bu)
    # get in the same list order the corresponding coordinates couples [(x, y)]
    coords, bounds, withins = get_all(d, tm, ls)
    # prints [['a', 'c', 'b', 'e', 'd', 'g', 'f', 'o', 'p'], ['k']]
    # if other singleton patches exist
    if rest:
        print "time tmax %s elements don't interact"%(len(rest))
        cr, br, wr = get_all(d, tm, rest)
        patchids = patch_encode(d, rest, len(ls))
        pvl.extend(patchids)
        coords.extend(cr); bounds.extend(br); withins.extend(wr);
    # A node includes:
    # t time point and patch id
    # ls particle ids per patch
    # coords (x,y) per patch of particle ids
    # bounds or concave hull per patch
    # withins or difference of two sets (coords and bounds) comprising core pts of hull
    # set id attribute for all nodes to recover maximum value while encoding
    nx.set_node_attributes(G, 'id', 0)
    # add node for patches at time point max
    pb = get_pids(d, tm, bounds); pw = get_pids(d, tm, withins)
    G.add_node(tm, id=pvl, p=ls, c=coords, b=bounds, pb=pb, w=withins, pw=pw)
    print "At t=%s : %s patches" %(tm, len(pvl)) # for i in xrange(len(pvl)): print len(ls[i])+","
    print "t\tn Patches\n%s,\t%s" %(tm, len(ls)+len(rest))
    return G


def increment(G, d, tm, distu, ru, gu, bu):
    ''' Incrementally runs the modular algorithm to create and verify patches
        Args    G graph data structure
                d dataframe
                tm time point max
                distu euclidean distance
                ru, gu, bu absolute diff. in each colourspace dimension
        Returns updated graph G
    '''
    print "[Init.] Assigning particles to patches incrementally (across time points)..."
    # time i
    # pids list of patch ids
    # p particles id
    # c coordinates of p
    # b concave hull of c
    # pb particle ids of b
    # w core points within b
    # pw particle ids of w
    for i in range(tm-1, -1, -1):
        # select all particles at time point i with no patch encoding
        tmp_d = d[ (d.z.values.ravel()==i) & (d.n.values.ravel()==0) ]
        if not(len(tmp_d)): # for no particles
            continue
        elif len(tmp_d) == 1: # one particle at time point t
            pid = [tmp_d.particle.values.tolist()]
            coords = zip( tmp_d.x.values, tmp_d.y.values )
            m = max(flatten([G.node[n]['id'] for n in G.nodes()]))
            val = m+1
            sele = d['particle'].isin(tmp_d.particle.values)
            d.set_value(sele, 'n', val)#; print d[sele]
            # add attributes to existing nodes
            G.add_node(i, id = [val], p = pid, c = coords, \
                       b = coords, pb = pid, w = coords, pw = pid )
            print "%s,\t%s" %(i, len(pid))
    
        else:
            # filter the distance sorted interactions by user values
            tmp_res = combine(i, tmp_d).sort(['dist', 'dr', 'dg', 'db'], \
                                             ascending=[1,1,1,1])
            if empty(tmp_res):
                continue
            else:
                tmp_res, rest = pfilter(tmp_res, i, distu, ru, gu, bu)
                plist = graphcc(tmp_res)
                # 4 possibilities plist (combinations) and rest (singleton patches)
                if not plist and not rest:
                    continue
                
                elif plist and not rest:
                    verify_patch(d, plist, distu, ru, gu, bu)
                    coords, bounds, withins = get_all(d, i , plist)
                    m = max(flatten([G.node[n]['id'] for n in G.nodes()]))
                    patchids = patch_encode_t(d, plist, m)
                    pb = get_pids(d, i, bounds); pw = get_pids(d, i, withins)
                    G.add_node(i, id=patchids, p=plist, c=coords, b=bounds,\
                               pb=pb, w=withins, pw=pw)

                elif not plist and rest:
                    coords1, bounds1, withins1 = get_all(d, i , rest)
                    m = max(flatten([G.node[n]['id'] for n in G.nodes()]))
                    patchids1 = patch_encode_t(d, rest, m)
                    G.add_node(i, id=patchids1, p=rest, c=coords1, b=bounds1,\
                               pb=rest, w=withins1, pw=rest)

                elif plist and rest:
                    verify_patch(d, plist, distu, ru, gu, bu)
                    coords, bounds, withins = get_all(d, i , plist)
                    coords1, bounds1, withins1 = get_all(d, i , rest)
                    # manage max patch ids value on the fly
                    m = max(flatten([G.node[n]['id'] for n in G.nodes()]))
                    patchids = patch_encode_t(d, plist, m)
                    m = max(patchids)
                    patchids1 = patch_encode_t(d, rest, m)
                    # recover particle ids given a list of coord
                    patchids.extend(patchids1); coords.extend(coords1);
                    bounds.extend(bounds1); withins.extend(withins1);
                    
                    p1 = get_pids(d, i, coords)
                    pb1 = get_pids(d, i, bounds)
                    pw1 = get_pids(d, i, withins)

                    G.add_node(i, id=patchids, p=p1, c=coords, b=bounds,\
                               pb=pb1, w=withins, pw=pw1)
    # if nodes in G exist print
    if G.nodes():
        print "[Done]\tIncremental assignment:\t\
        %s patches from %s particles\
        "%(max(flatten([G.node[n]['id'] for n in G.nodes()])), int(max(d.particle)))
    return G


def get_all(df, t, l):
    ''' Args:   df dataframe pandas to select pids from
        t time point at which selection of coord is desired
        l list of lists containing relevant particle ids
        Returns a list of lists with the corresponding  (x, y) coord as tuples
    '''
    coords, bounds, withins, pb, pw = [], [], [], [], []
    for i in range(len(l)):
        if len(l[i]) <= 6: # control for min amount of cords for Delaunay method
            coord = pids_coords(df, t, l[i])
            bound, within = coord, coord
            # secondary control if empty replace by coords
            for j in xrange(len(withins)):
                if empty(withins[j]):
                    withins[j]=coords[j]
        else:
            coord = pids_coords(df, t, l[i])
            bound, within = concave(coord)
        coords.append(coord); bounds.append(bound); withins.append(within)
    return coords, bounds, withins


def pids_coords(df, t, l):
    ''' Retrieves coordinates of relevant particles from dataframe
        Args    df dataframe pandas to select pids from
                t time point at which selection of coord is desired
                l list of lists containing relevant particle ids
        Returns a list of lists with the corresponding  (x, y) coord as tuples
    '''
    # selection of particles at time point t
    # consider the special case of 1 particle on its own
    if len(l) == 1:
        s = df[ (df.z == t) & (df['particle']==l[0]) ]
        return list(zip( s.x.values.ravel(), s.y.values.ravel() ))
    elif len(l) > 1:
        s = df[ (df.z == t) & (df['particle'].isin(l)) ]
        return list(zip( s.x.values.ravel(), s.y.values.ravel() ))


def flatten(l):
    ''' Flattens a list e.g. [ [], [] , [] ] -> [ ]
        Requirement: input list must be a list of lists of int
        Args    l list
        Returns a flattened list
    '''
    l2 = [x for x in l if x]
    return [x for y in l2 for x in y]


def combine(t, df):
    ''' Creates pairwise combinations for a time point
        Args    t time point
                df dataframe
        Returns res resulting selection of the dataframe with 6 new columns
    '''
    dm = df[df.z.values == t] # slice data at t
    # sometimes whole dataframe is one particle (one line in data frame)
    if len(dm) == 1:
        return dm
    else:
        # generate all possible combinations of the resulting rows
        index = np.array(list(combinations(range(dm.shape[0]), 2)))
        df1, df2 = [dm.iloc[idx].reset_index(drop=True) for idx in index.T]
        res = pd.concat([
                         np.hypot(df1.x - df2.x, df1.y - df2.y),
                         np.abs(df1[["r8", "g8", "b8"]] - df2[["r8", "g8", "b8"]]),
                         df1.particle, df2.particle
                         ], axis=1)
        res.columns = ["dist", "dr", "dg", "db", "pid1", "pid2"]
        return res


def scombine(sele):
    ''' Creates pairwise combinations for a given selection
        Args    s selection
        Returns res resulting selection of the dataframe with 6 new columns
    '''
    index = np.array(list(combinations(range(sele.shape[0]), 2)))
    df1, df2 = [sele.iloc[idx].reset_index(drop=True) for idx in index.T]
    res = pd.concat([
                     np.hypot(df1.x - df2.x, df1.y - df2.y),
                     np.abs(df1[["r8", "g8", "b8"]] - df2[["r8", "g8", "b8"]]),
                     df1.particle, df2.particle
    #                 df1.n.astype(str) + "-" + df2.n.astype(str)
                     ], axis=1)
    res.columns = ["dist", "dr", "dg", "db", "pid1", "pid2"]
    return res


def pfilter(res, t, distu, ru, gu, bu):
    ''' Filters dataframe res - output of combine() - given user parameters
        Args    res dataframe
                distu, ru, gu and bu are user set, a default preset is given
                t time point
        Returns res unique positive interactions present in initial arg res
                r3 all non-interacting particles or singletons
    '''
    # filter by n conditions: rows that have da & db & dc < 51 (255/5) & dist < 100
    r1 = res[ (res['dist']<distu) & (res['dr']<ru) & (res['dg']<gu) & (res['db']<bu) ]
    # recover the two columns of particle ids
    r2 = np.array(r1[['pid1', 'pid2']])
    # catch all the other elements that get filtered out and recover them externally
    r3 = np.array(res[['pid1', 'pid2']])
    r3 = list(set(np.unique(np.ravel(r3))) - set(np.unique(np.ravel(r2))))
    r3 = np.array(r3).reshape(1, -1).T.tolist()
    # Split numpy array into n arrays based on unique values in the first column
    res = np.split(r2, np.where(np.diff(r2[:,0]))[0]+1)
    # remove duplicates within each array
    res = [list(np.unique(i)) for i in res] # arr1 contains values
    return res, r3


def list2graph(l):
    ''' Creates a graph from a list
        Args    l list
        Returns G graph networkx.classes.graph.Graph
    '''
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also implies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    ''' Returns the edges of the graph formed by l
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        Args    l list
        See list2graph()
    '''
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current


def graphcc(l):
    ''' Computes the connected component on a list
        Args    l list
        Returns a sorted list of the given connected components of a graph
    '''
    return sorted(connected_components(list2graph(l)))


def patch_encode(df, l, v):
    ''' Encodes patch ids to the dataframe
        Args    df dataframe
                l unflattened list of interacting particle ids
                v value of of the patch id
        Returns vlist unique patch ids used in column 'n' of main dataframe
    '''
    vlist = []
    for i in range(len(l)):
        # select particles in each patch
        sele = df['particle'].isin(l[i])
        # encode patch ids to 'n' column
        df.set_value(sele, 'n', v+i+1)
        # recover patch ids for later usage
        vlist.append(v+i+1)
    return vlist


def patch_encode_t(df, l, v, t):
    ''' Encodes patch ids to the dataframe excluding all time points after t
        Args    df dataframe
        l unflattened list of interacting particle ids
        v value of of the patch id
        Returns vlist unique patch ids used in column 'n' of main dataframe
        '''
    vlist = []
    for i in range(len(l)):
        # select particles in each patch
        sele = (df['particle'].isin(l[i])) & (df['z']<=t)
        # encode patch ids to 'n' column
        df.set_value(sele, 'n', v+i+1)
        # recover patch ids for later usage
        vlist.append(v+i+1)
    return vlist


def verify_patch(d, ls, distu, ru, gu, bu):
    ''' Verifies whether a patch can be divided given user parameters
        Args    d dataframe
                tm time point max
                distu euclidean distance
                ru, gu, bu absolute diff. in each colourspace dimension
        Returns void funtion
    '''
    for i in range(len(ls)): # loop over each patch
        # df at patch i
        df = d[d['particle'].isin(ls[i])]
        # sort the dataframe
        df = df.sort(columns=['z'])
        # get a list of z
        names = df['z'].unique().tolist()
        names.sort(reverse=True)
        names.pop(0) # do not run verification on t_max or t_present
        for j in names: # loop over each time point minus tmax in names
            # perform a lookup on a 'view' of the dataframe
            view = df.loc[(df.z==j)]
            # query all rows within a view
            res = combine(j, view)
            if len(res)==1: continue # if 1 particle do increment
            # filter by given parameters returning main list l and o other singletons
            l, o = pfilter(res, j, distu, ru, gu, bu)
            # connected components on main result
            r = graphcc(l)
            if empty(r) or len(r)==1:
                continue
            elif len(r)!=1:
                print "Splits required at t=%s"%j
                # sort by descending order the sublists
                r.sort(key=len, reverse=True)
                # separate largest element from rest of particles
                r0, r1 = r.pop(0), flatten(r)
                o = flatten(o); r1.extend(o)
                # decode the value of 'n' for selected rows to 0
                view = d.ix[((d['particle'].isin(ls[i])) & (d['z']==j) & d['particle'].isin(r1)),'n'] = 0
            else:
                continue


def empty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(empty, inList) )
    return False # Not a list


def concave(points, alpha_x=150, alpha_y=250):
    ''' Computes the concave hull using scipy and networkx
        Args    points list of coordinates [ [], [], [], ...]
                alpha_x
    '''
    if empty(points):
        print "[Error]\tempty list given to concave procedure. Exiting..."
        sys.exit(0)
    points = [(i[0],i[1]) if type(i) <> tuple else i for i in points]
    de = Delaunay(points)
    dec = []
    a = alpha_x
    b = alpha_y
    for i in de.simplices:
        tmp = []
        j = [points[c] for c in i]
        if abs(j[0][1] - j[1][1])>a or abs(j[1][1]-j[2][1])>a or \
           abs(j[0][1]-j[2][1])>a or abs(j[0][0]-j[1][0])>b or \
           abs(j[1][0]-j[2][0])>b or abs(j[0][0]-j[2][0])>b:
            continue
        for c in i:
            tmp.append(points[c])
        dec.append(tmp)
    G = nx.Graph()
    for i in dec:
            G.add_edge(i[0], i[1])
            G.add_edge(i[0], i[2])
            G.add_edge(i[1], i[2])
    ret = []
    for graph in nx.connected_component_subgraphs(G):
        ch = ConvexHull(graph.nodes())
        tmp = []
        for i in ch.simplices:
            tmp.append(graph.nodes()[i[0]])
            tmp.append(graph.nodes()[i[1]])
        ret.append(tmp)
    # concave hull (boundary) and all points inside the hull
    return ret[0], list(set(points) - set(ret[0]))
    

def get_pids(d, i, lst):
    ''' Given a list or list of lists containing tuples of coordinates
        Retrieve the corresponding list of particle ids
        Args   lst the list
                d the data frame
                i time point
        Returns r list of particle for the given coordinates
    '''
    di = d[d['z']==i]
    r = []
    for m in lst:
        temp = pd.DataFrame(m)[0]
        r.extend([di[di.x.isin(temp)].particle.values.tolist()])
    return r


def update_graph(d, G):
    ''' Finds all particles at a time point i that are not in a graph.node
        Args    d data frame pandas
                G graph networkx path_graph(i_{max})
        Returns void function
    '''
    print "Updating graph nodes with patch ids, coordinates, boundaries, etc"
    # get the Dictionary of attributes key-ed by node number (given key is p)
    part = nx.get_node_attributes(G, 'p')
    k = part.keys() # recover time points where p exists
    # slice data and extend node attributes
    for i in G.nodes():
        # slice d at i
        td = d[ (d.z.values.ravel()==i) ]
        if not len(td): continue
        # if time point i is in k (shared particle and patch ids)
        if i in k:
            print "At node", i
            # difference between [particle ids in d at i] and [ G.node[i]['p'] ]
            diff = list( set( np.unique(td.particle.values) ) \
                        -set( np.unique(flatten(G.node[i]['p'])) ) )
            # sub sliced d and sorted by patch ids encoding column n
            diffd = td[td['particle'].isin(diff)].sort(['n'], ascending=[1])
            # get patch ids for diff particles as integers
            patchids = map(int, np.unique(diffd.n.values).tolist())
            # slice diffd by values of n
            plist = []
            for id in patchids:
                plist.append(diffd[diffd.n==id].particle.tolist())
            # get coords
            coords, bounds, withins = get_all(d, i, plist)
            #            print "\n\nha:",bounds
            pbounds = get_pids(d, i, bounds); pwithins = get_pids(d, i, withins)
            # update the node attributes at i
            ni = G.node[i]
            # preserve brackets
            if len(ni['c'])==1: # if one element append usage
                patchids.extend(ni['id']); plist.extend(ni['p'])
                coords.append(ni['c']); bounds.append(ni['b'])
                pbounds.extend(ni['pb']); withins.append(ni['w'])
                pwithins.extend(ni['pw'])
            else: # multiple elements in list hence extend
                patchids.extend(ni['id']); plist.extend(ni['p'])
                coords.extend(ni['c']); bounds.extend(ni['b'])
                pbounds.extend(ni['pb']); withins.extend(ni['w'])
                pwithins.extend(ni['pw'])
                #
            G.remove_node(i)
            G.add_node(i, id=patchids, p=plist, c=coords, b=bounds, \
                       pb=pbounds, w=withins, pw=pwithins)
        if i not in k:
            print "At node", i
            # sort selection by patch ids (col: 'n')
            di = td.sort(['n'], ascending=[1])
            patchids = map(int, np.unique(di.n.values).tolist())
            plist = []
            for id in patchids:
                plist.append(di[di.n==id].particle.tolist())
            coords, bounds, withins = get_all(d, i, plist)
            pb = get_pids(d, i, bounds); pw = get_pids(d, i, withins)
            G.add_node(i, id=patchids, p=plist, c=coords, b=bounds,\
                       pb=pb, w=withins, pw=pw)
    #            print "\n\nreally",bounds
    print "[Done]\tGraph update:\t%s patches in %s nodes"\
        %(max(flatten([G.node[n]['id'] for n in G.nodes()])), max(G.nodes()))



def find_merges(G, d, tm, distu, ru, gu, bu):
    ''' Finds all possible merges between all pair of non singleton patches
        Args    G graph 
                d dataframe
                distu spatial distance
                ru, gu, bu colour distance
        Returns candidates merge list
    '''
    # control first where G.node[i]=={'id':0} exists
    ct = [i for i in G.nodes() if (G.node[i]=={'id':0})==False]
    candidates = []
    print "[Init.] Evaluating merges"
    for i in ct:
        print "At node", i
        # loop over each time point
        # box only patches with coordinates length > 1
        r = [G.node[i]['b'].index(j) for j in G.node[i]['b'] if len(j)>1]
        r = list(np.unique(r))
        print "Evaluating %s patches" %(len(r))
        # at least 2 coord. to make a rectangle
        if empty(r):
            continue
        else:
            print "At node ",i
            # inter-patch pair combinations
            c = list(combinations(r,2))
            # for each pair in [(,), (,), (,), etc] create bounding rectangles
            for k in c:
                r1, r2 = G.node[i]['b'][k[0]], G.node[i]['b'][k[1]] # coords
                # if necessary created rectangles are rotated
                rect1 = cv2.minAreaRect(np.array(r1, dtype="float32"))
                rect2 = cv2.minAreaRect(np.array(r2, dtype="float32"))
                p1, p2 = G.node[i]['p'][k[0]], G.node[i]['p'][k[1]] # particles
                # if no intersection then continue
                #INTERSECT_NONE=0 - No intersection
                #INTERSECT_PARTIAL=1 - There is a partial intersection
                #INTERSECT_FULL=2 - One of the rectangle is fully enclosed in the other
                if cv2.rotatedRectangleIntersection(rect1, rect2)[0]==0:
                    continue
                # then slice df at time point i using particles within rect1 + 2
                else:
                    pt = p1; pt.extend(p2) # particles from p1 and p2
                    sele = d.ix[ (d['particle'].isin(pt)) & (d['z']==i) ]
                    # then assess possibility of merge:
                    # combine particle pairs present in selection
                    res = scombine(sele)
                    # filter all interactions by user values
                    l, rest = pfilter(res, tm, distu, ru, gu, bu)
                    # connected components
                    ls = graphcc(l)
                    # if merge possible put particles in ordered candidates list
                    if len(ls)==1:
                        id1 = int(d[ (d.z==i) & (d.particle==G.node[i]['pb'][k[0]][0])].n)
                        id2 = int(d[ (d.z==i) & (d.particle==G.node[i]['pb'][k[1]][0])].n)
                        print id1, id2
                        candidates.append([i, id1, id2])
    return candidates


def true_merge(candidates, limit):
    ''' Finds merge candidates given a limit
        Args    candidates list
                limit int value of min consecutive frames
        Returns list of patch ids for true merge
    '''
    if limit == 1:
        print "[Error]\ta merge step of 1 cannot be supplied."
    # convert list of lists to pandas df
    headers = ['t', 'i1', 'i2']
    td = pd.DataFrame(candidates, columns=headers)
    # count by unique pairs in i1 and i2
    tdu = td.groupby(['i1', 'i2']).count()
    # create a dataframe with the occurences
    tda = pd.DataFrame({'occ' : td.reset_index().groupby(['i1', 'i2'])['index'].apply(lambda x:list(x))})
    # add count column to df
    tda['count'] = pd.Series(tdu['t'], index=tda.index)
    tda = tda.reset_index()
    # filter by Imax value referred to as trajectory linking appear/disappear limit
    l = [] # list of lists of consecutive items only
    for index, row in tda.iterrows():
        for k, g in groupby(enumerate(row['occ']), lambda ix : ix[0] - ix[1]):
            j = map(itemgetter(1), g)
            if len(j) >= limit:
                mn = min(td[(td.i1==row['i1']) & (td.i2==row['i2'])].t)
                mx = max(td[(td.i1==row['i1']) & (td.i2==row['i2'])].t)
                l.append([row['i1'], row['i2'], mn, mx])
    return l


def post_merge(d, G, l):
    ''' Calls updates on dataframe and graph data structures 
        Args    d pandas dataframe
                G networkx graph
                l list containing pairs of patch ids to be merged
        Returns d and G
    '''
    return update_df_postmerge(d, l), update_graph_postmerge(d, G, l)


def update_df_postmerge(d, ll):
    ''' patch encode into dataframe'''
    for i in range(len(ll)):
        # select all rows in which ll[i] patch ids exist
        sele = (d['n'].isin(l[i][:2]) & (df['z']>=l[i][2]) & (df['z']<=l[i][3]))
        # flip 3 to 1 (smaller patch ids wins since observed early on)
                newpid = min(ll[i][:2]])
        d.set_value(sele, 'n', newpid)
    return d


def update_graph_postmerge(d, G, l):
    ''' recover all nodes that contain patch ids 1 and 3'''
    part = nx.get_node_attributes(G, 'p')
    k = part.keys()
    for j in range(len(l)):
        for i in G.nodes():
            td = d[ (d.z.values.ravel()==i)]
            if not len(td): continue
            if i in k:
                ni = G.node[i]
                print "At node", i
                l[j].sort(); m = l[j] # to have m[0] always the min value or newpid
                m0, m1 = m[0], m[1]; lst0 = [m0]
                m0i = [ni['id'].index(x) for x in ni['id'] if x == m0]
                m1i = [ni['id'].index(x) for x in ni['id'] if x == m1]
                if empty(m0i) and empty(m1i):
                    continue
                elif m0i and m1i:
                    ni = G.node[i]
                    for ind in ni:
                        p1, p2 = ni[ind].pop(m0i[0]), ni[ind].pop(m1i[0])
                    pids = d[ (d.z==i) & (d.n==m0) ].particle.values.ravel()
                    pids = np.unique(pids)
                    # df selection to get coord. couples
                    s = d[ (d.z==i) & (d['particle'].isin(pids))]
                    coords = list(zip( s.x.values.ravel(), s.y.values.ravel() ))
                    bounds, withins = concave(coords)
                    ni['id'].append(m0); ni['c'].append(coords)
                    ni['b'].append(bounds); ni['w'].append(withins)
                    ni['p'].append(pids.tolist())
                    pbounds, pwithins = [], []
                    # get particle bounds and withins
                    di = d[d['z']==i]
                    for n in bounds:
                        x1 = n[0]
                        pbounds.extend(di[di.x==x1].particle.values.tolist())
                    ni['pb'].append(pbounds)
                    for o in withins:
                        x2 = o[0]
                        pwithins.extend(di[di.x==x2].particle.values.tolist())
                    ni['pw'].append(pwithins)
    print "[Done]\tGraph update:\t%s patches in %s nodes"\
    %( len(np.unique(d.n.values.ravel())), max(G.nodes()) )
    return G


def export_data(d, G, path):
    ''' Exports d pandas df and G networkx graph to
                .gpickle, .gexf, .graphml, .gml and .json files
        Args    d dataframe
                G graph
                path destination and filename (with same timestamp as csv file)
        Returns void function
    '''
    f = path.split('_'); f = f[len(f)-1].split('.')[0] #recover path timestamp
    o = '/'.join(path.split('/')[:len(path.split('/'))-1])+'/'+"graph_"+f
    # pass full output path o to write the different graph formats
    
    nx.write_gpickle(G, o+".gpickle")
    nx.write_gexf(G, o+".gexf")
    nx.write_graphml(G, o+".graphml")
    nx.write_gml(G, o+".gml")
    with open(o+".json", 'w') as out:
        out.write(json.dumps(json_graph.node_link_data(G)))
    
    print "Dataframe summary:\n", data.describe()

    file_name = "../particles_"+o+".csv"; print "Exporting %s" %(file_name)
    data.to_csv(file_name, sep='\t', encoding='utf-8')




