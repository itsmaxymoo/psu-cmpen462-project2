# psu-cmpen462-project2

## Setup

1. Create a virtual environment. The dependencies have weird version requirements here.
    * Note: Tensorflow requires python 3.12 which many users may not have. The easiest solution is to first download
        python 3.12 which will vary based on your OS. Then create a venv using python 3.12 by running the following command 
        on Linux `python3.12 -m venv venv` (this will make sure the virtual environment supports python 3.12). Now 
        Tensorflow will be happy
2. Install the dependencies: `pip install -r requirements.txt`

## To run the project:

From the project root: `python -m project <train.csv> <test.csv>`