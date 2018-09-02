# placepulse2_lynch
coding script for "Did Kevin Lynch see beyond what was seen?: Computer-vision-based analysis of Imageability"

## download Google Street View images 
using the Google Street View API, download the urban scenic images. 
```
python gsv_downloader.py
```

## preprocessing of the images
1. convert images into array
```
python generate_training_list.py
```

2. remove unavailable images
```
python remove_sorry.py
```

3. remove draw match
```
python remove_equal.py
```

4. convert the winner columns into binary
```
python convert_winner_binary.py
```

5. standarise the images
```
python image_rescaler.py
```

6. compute degree centrality
```
python degree_centrality.py
python cleaner_location_to_degree.py
```

## train the model
```
python lynch_rss_cnn.py
python rss_cnn.py
```

## predict 
```
python prediction.py
```


![alt text](https://raw.githubusercontent.com/soma11soma11/placepulse2_lynch/master/image.png)
