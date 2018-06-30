# rsc18
Publication of the code we used in the RecSys Challenge 2018.

## Usage
* Place the playlist JSON files in the data/original and the challenge set in the data/online directory
* Run the python scripts in the following order
  * prepare_data.py (Combines and converts the files into CSV with int ids for tracks, artists, and albums)
  * prepare_test.py (Converts the challenge set file into CSV while mapping the URIs to our int ids)
  * create_sample.py (Creates a sample of 50k random playlists with a test set of 500 playlists)
  * main.py (Contains the code to compute the predictions in a single run for the 50k sample)
  * prepare_solution.py (Creates things in the format)
* Each script defines FOLDER_TRAIN and FOLDER_TEST along with other important parameters in the head of the file