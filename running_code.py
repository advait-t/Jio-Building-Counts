import functions as f
import sys

path = (sys.argv[1])
# target = (sys.argv[2])

# path = 'ga_cluster_geohash_mapping_20211011_60perc.csv'
colour_list = 'yellow'
start = f.time.time()
data, empty_df, final_df, image_df = f.input_polygons(path, colour_list)
end = f.time.time()
print("Time elapsed: " +str(end - start)+"s")