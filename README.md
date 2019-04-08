# lego-ceptron
Lego bricks image classifier

## Installation
1. Create and activate virtualenv
    ```bash
    python3 -m virtualenv lego-ceptron
    source lego-ceptron/bin/activate
    ```
    alternatively, using virtualenvwrapper:
    ```bash
    mkvirtualenv --python=/usr/bin/python3 lego-ceptron
    ```
1. Install packages
    ```bash
    pip install -r requirements.txt
    ```
    
## Dataset
Download dalaset from [this Kaggle](https://www.kaggle.com/pacogarciam3/lego-brick-sorting-image-recognition)
(login reqired) and unzip it to `./data` directory.
Remember to unzip all subarchives as well, so that your file tree looks like:
```
./data/
   Base Images/
   Cropped Images/
   imageSetKey.csv
   other .jpg files
```