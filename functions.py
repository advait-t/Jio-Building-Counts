import io
import ast
import cv2
import json
import time
import heapq
import imutils
import warnings
import tempfile
import webcolors
import numpy as np
import urllib.parse
import pandas as pd
from PIL import Image
from PIL import ImageFile
# from rembg.bg import remove
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from scipy.spatial import KDTree
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
from math import log, exp, tan, atan, pi, ceil



def convert_rgb_to_names(rgb_tuple):
    css3_db = webcolors.CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(webcolors.hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]

def findGreatesContour(contours):
    #Finding the largest contour and its index
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    area = []
    while (i < total_contours):
        ar = cv2.contourArea(contours[i])
        if(ar > largest_area):
            largest_area = ar
            largest_contour_index = i+1
        i+=1
    return area, largest_contour_index, largest_area

def get_image_data_ready(image):
#   Extracting RGB Values of the image and storing it in a dataframe
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.array(image)
    # print('Original Image')
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    # print('*'*30)
    # print('*'*30)

    image = image.reshape(-1,3)

    img_df = pd.DataFrame(image, columns=['r','g','b'])

    img_df['rgb'] = img_df.values.tolist()
    img_df = img_df.drop(['r','g','b'], axis = 1)
    
    # Making columns for percent of pixels covered by a colour and count of pixels
    final_dataframe = pd.DataFrame((img_df['rgb'].value_counts(normalize = True))*100)
    final_dataframe.rename({'rgb':'percent_pixels_covered'}, axis = 1, inplace=True)
    final_dataframe['pixels_covered'] = pd.DataFrame(img_df['rgb'].value_counts())

    final_dataframe['rgb_values'] = final_dataframe.index
    final_dataframe.reset_index(inplace = True, drop = True)

    final_dataframe['colour'] = ""
    for i in range(len(final_dataframe)):
        # Getting colour names from rgb values
        final_dataframe['colour'][i] = convert_rgb_to_names(final_dataframe['rgb_values'][i])

    top_dataframe = final_dataframe.nlargest(10, columns='percent_pixels_covered')
    return top_dataframe, final_dataframe, img_df

def print_information(contours, colour_name, rgb_value, show_all_diagrams, area_covered, colour_list = None):
    area, largest_contour_index, largest_area = findGreatesContour(contours)
    count = []
    count.append(len(contours))
    count.append(colour_name)
    # if show_all_diagrams =='Yes':
    #     for i in colour_list:
    #         if colour_name == i:
    #             print(" ")
    #             print('Percent Area Covered by the Polygons:', area_covered,"%")
    #             print("Number of Polygons:", len(contours))
    #             # print('Largest Polygon by Area:', largest_area)
    #             print("RGB Value:", rgb_value)
    #             print('Colour Name:', colour_name)
    #             print("*"*30)
    #             print("*"*30)
    #             print(" ")
    #             print(" ")
    return count

def remove_noise(dilation):
    contours = cv2.findContours(image=dilation, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect_areas = []
    final_contours = []
    for i, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_area = w * h
        if contour_area <= 2:
            dilation[y:y + h, x:x + w] = 0
        if contour_area >= 10:
            final_contours.append(contours[i])
#     print('Done')
    return final_contours

def object_counting_and_area(image, colour_name ,rgb_value, show_all_diagrams, area_covered, colour_list = None):
#     image = cv2.imread(path)
#     im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     im = image
    im = image.copy()
    white = [255,255,255]
    black = [0,0,0]

    
    # Make mask of all perfect pixels
    Rmask = np.all(im == rgb_value,  axis=-1)

    
    # Make all rgb_values pixels white and at same time everything else black
    im = np.where(np.all(im == rgb_value, axis=-1, keepdims=True), white, black)
    im = im.tolist()
    im = np.float32(im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(gray)

    
    # apply binary thresholding
    ret, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations = 1)
    

    # detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(image=dilation, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            
    
    #removing the unecessary noises
    final_contours = remove_noise(dilation)
    
    
    #Printing the image information:
    #print('The Interested areas are marked white in the binary image showing the buildings')
    
    
    #visualising the binary Image
    if show_all_diagrams =='Yes':
        for i in colour_list:
            if colour_name == i:
                dilation = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)
                # plt.imshow(dilation)
                # plt.axis('off')
                # plt.show()
                #cv2.imwrite('/Users/advait_t/Desktop/binary_image.png', dilation)
    
    
    # draw contours on the original image
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=final_contours, contourIdx=-1, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA)
    
    
    #Printing the image information:
    #print('We draw borders around buildings in the original image')
    
    
    # see the results
    if show_all_diagrams =='Yes':
        for i in colour_list:
            if colour_name == i:
                image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                # plt.imshow(image_copy)
                # plt.axis('off')
                # plt.show()
                #cv2.imwrite('/Users/advait_t/Desktop/contour_drawn_image.png', image_copy)
    
    
    count = print_information(final_contours, colour_name ,rgb_value, show_all_diagrams, area_covered, colour_list)
    return final_contours, count

def run_code(image,show_all_diagrams, colour_list):
    a = []
    top_dataframe, final_dataframe, img_df = get_image_data_ready(image)
    rgb_interested = [[254,250,0],[255, 251, 0]]
    colour_interested = ['yellow']
    for i in range(len(final_dataframe)):
        if final_dataframe['rgb_values'][i] in rgb_interested:
            contour, count =  object_counting_and_area(image,final_dataframe['colour'][i],final_dataframe['rgb_values'][i], show_all_diagrams, final_dataframe['percent_pixels_covered'][i].round(2), colour_list)
            a.append(count)

    count_df = pd.DataFrame(a, columns=['Building_Count','Colour_Name'])
    
    a = count_df[count_df['Colour_Name'].isin(['yellow'])]
    #print('Total Buildings:',sum(a['Building_Count']))
    total_buildings = sum(a['Building_Count'])
    return top_dataframe, count_df, final_dataframe, img_df, total_buildings

def keep_interested_area(image):
    temp_dir = tempfile.TemporaryDirectory()
    cv2.imwrite((str(temp_dir.name)+'/test.png'), np.array(image))
    f = np.fromfile((str(temp_dir.name)+'/test.png'))
    # result = remove((f))
    img = Image.open(io.BytesIO(f)).convert("RGBA")
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    # image = crop(image)
    temp_dir.cleanup()
    #cv2.imwrite('/Users/advait_t/Desktop/out1.jpg', image)
#     print('Done')
    return image

def latlontopixels(lat, lon, zoom, ORIGIN_SHIFT, INITIAL_RESOLUTION):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom, ORIGIN_SHIFT, INITIAL_RESOLUTION):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon

def coordinates_processing(polygon_coordinates):
    if isinstance(polygon_coordinates, list):
        polygon_coordinates = tuple(polygon_coordinates)
        
    first_element = polygon_coordinates[0]
    final_coordinates = polygon_coordinates + (first_element,)
    joined_string = "|".join(str(i) for i in final_coordinates)
    joined_string = joined_string.replace("(","").replace(")","")
    joined_string = joined_string.replace(" ", "")
    return joined_string

def gmaps_polygon_and_extractor(polygon_coordinates = 0):
    
    upperleft, lowerright = get_topleft_bottomright(polygon_coordinates)
    
    upperleft = str(upperleft).strip('()')
    lowerright = str(lowerright).strip('()')
    
    EARTH_RADIUS = 6378137
    EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
    INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0 #256.0
    ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0
    apikey = 'AIzaSyCsiLA-UdaaKuXYVXHopKpL-YOV9kf7mU4'
    zoom = 17
    

    ullat, ullon = map(float, upperleft.split(','))
    lrlat, lrlon = map(float, lowerright.split(','))

    # Set some important parameters
    scale = 1
    maxsize = 640

    # convert all these coordinates to pixels
    ulx, uly = latlontopixels(ullat, ullon, zoom, ORIGIN_SHIFT, INITIAL_RESOLUTION)
    lrx, lry = latlontopixels(lrlat, lrlon, zoom, ORIGIN_SHIFT, INITIAL_RESOLUTION)

    # calculate total pixel dimensions of final image
    dx, dy = lrx - ulx, uly - lry

    # calculate rows and columns
    cols, rows = int(ceil(dx/maxsize)), int(ceil(dy/maxsize))

    # calculate pixel dimensions of each small image
    bottom = 120
    largura = int(ceil(dx/cols))
    altura = int(ceil(dy/rows))
    alturaplus = altura + bottom


    # setting the polygons to the required format
    if polygon_coordinates != 0:
        coordinates = coordinates_processing(polygon_coordinates)
        
    final = Image.new("RGB", (int(dx), int(dy)))
    for x in range(cols):
        for y in range(rows):
            dxn = largura * (0.5 + x)
            dyn = altura * (0.5 + y)
            latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom/2, zoom, ORIGIN_SHIFT, INITIAL_RESOLUTION)
            position = ','.join((str(latn), str(lonn)))
            urlparams = urllib.parse.urlencode({'center': position,
                                          'zoom': str(zoom),
                                          'size': '%dx%d' % (largura, alturaplus),
                                          'maptype': 'roadmap'})
            if polygon_coordinates !=0:
                url = 'https://maps.googleapis.com/maps/api/staticmap?key=' + apikey + '&'+ urlparams+ '&' + 'style=element:labels%7Cvisibility:off&style=feature:administrative%7Celement:geometry%7Cvisibility:off&style=feature:administrative.land_parcel%7Cvisibility:off&style=feature:administrative.neighborhood%7Cvisibility:off&style=feature:landscape.man_made%7Celement:geometry.fill%7Ccolor:0xfffb00&style=feature:landscape.natural%7Ccolor:0xb51a00&style=feature:landscape.natural.landcover%7Ccolor:0xb51a00&style=feature:landscape.natural.terrain%7Celement:geometry.fill%7Ccolor:0xb51a00&style=feature:poi%7Cvisibility:off&style=feature:road%7Cvisibility:off&style=feature:road%7Celement:labels.icon%7Cvisibility:off&style=feature:transit%7Cvisibility:off' + '&' + 'path=color:cornflowerblue' +'|'+ 'weight:20' +'|'+coordinates
            else:
                url = 'https://maps.googleapis.com/maps/api/staticmap?key=' + apikey + '&'+ urlparams+ '|'+ 'weight:1'
            f=urllib.request.urlopen(url)
            im=Image.open(BytesIO(f.read()))
            final.paste(im, (int(x*largura), int(y*altura)))
#     plt.imshow(final)
#     plt.show()
    final.save('/Users/advait_t/Desktop/raw_maps.png')
    final = keep_interested_area(final)
#     final = cv2.resize(final, (0,0), fx=0.5, fy=0.5) 
#     print(final.shape)
#     print('Done')
#     plt.imshow(final)
#     plt.show()
    return final

def get_topleft_bottomright(polygon):
    lat,long = map(list,zip(*polygon))
    max_lat = max(lat)
    min_lat = min(lat)
    left_long = min(long)
    right_long = max(long)

    top_left = ()
    bottom_right = ()
    top_left = top_left + (max_lat, left_long)
    bottom_right = bottom_right + (min_lat,right_long)
    return top_left, bottom_right

def input_polygons(path, colour_list):
    data = pd.read_csv(path)
    data['Building_Count'] = 0
    data['polygon_json'] = data['polygon_json'].apply(json.loads)
    for j in range(len(data['polygon_json'])):
        coordinates = data.loc[j,'polygon_json']['coordinates'][0]
        for i in range(len(coordinates)):
            coordinates[i].insert(0, coordinates[i].pop())
            coordinates[i] = tuple(coordinates[i])
        area_image = gmaps_polygon_and_extractor(coordinates)
        df, empty_df, final_df, img_df, total_buildings = run_code(area_image, 'Yes', colour_list)
        data['Building_Count'][j] = total_buildings
    data.to_csv('building_counts_frompy.csv')
    return data, empty_df, final_df, img_df