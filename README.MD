# rsc18
Publication of the code we used in the RecSys Challenge 2018.

## Requirements
* The code was tested with miniconda3 (python3.6.5)
* All neccessary libraries can be installed via requirements.txt
```console
pip install -r requirements.txt
```

## Usage
* Place the playlist JSON files in the data/original and the challenge set in the data/online directory
* Run the python scripts in the following order
  * prepare_data.py (Combines and converts the files into CSV with int ids for tracks, artists, and albums)
  * prepare_test.py (Converts the challenge set file into CSV while mapping the URIs to our int ids)
  * crawl_metadata.py (Optional, uses the libary spotipy to collect additional meta data for all tracks in the dataset)
    * USER, CLIENT\_ID, and CLIENT\_SECRET have to be adjusted in the file. 
  * create_sample.py (Creates a sample of 50k random playlists with a test set of 500 playlists)
  * caculate_parts.py (Contains the code to individualy compute the predictions for all employed methods for the 50k sample)
  * combine_parts.py (Combines all the individual solutions in our hybrid approach for the 50k sample)
    * The creation of our "creative" solution is commented out as the crawling is time consuming and marked as optional.
  * prepare_solution.py (Converts our solution format to the official submission format)
* Most of the scripts defines FOLDER_TRAIN and FOLDER_TEST along with other important parameters in the head of the file
  * To reproduce our final submissions change those folders to 'data/data_formatted/' and 'data/online/', respectively.
  * As mentioned above, the creative solutions is commented out and can be included in line 53 of combine_parts.py once the metadata is fully craweled. 
