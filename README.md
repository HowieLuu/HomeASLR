# HomeASLR
Training an ML model on American Sign Language Recognition (ASLR), then interfacing the model with Google Home to allow for sign language to speech 

## How to use
### Set-up
1. Install python dependencies (see requirements.txt)
2. The data has already been preprocessed (data/data.pickle)
    1. src/data_processing.py can be run on new ASL data to get hand landmark data if desired
3. The model has been pre-trained
    1. src/model.py can be run on data/data.pickle to retrain
4. Install nodeJS and dependencies (commands for windows)
    1. install node.js
    2. \[create directory for node red install\]
    3. cd \[new directory\]
    4. npm install node_red
    5. npm install node-red-contrib-castv2
    6. mklink red.js node_modules\node-red\red.js
### Running the application
1. Run the aslr.py script
2. start Node Red
    1. node node_red/red.js
