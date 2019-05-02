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
1. Add project's root directory to PYTHONPATH:
    ```bash
    add2virtualenv <project_root>
    ```
    
## Dataset
Download dalaset from [this Kaggle](https://www.kaggle.com/pacogarciam3/lego-brick-sorting-image-recognition)
(login reqired) and unzip it to `./data` directory.
Remember to unzip all subarchives as well, so that your file tree looks like this:
```
./data/
   Base Images/
   Cropped Images/
   imageSetKey.csv
   other .jpg files
```

After this go to `./data` folder and run:
```
python split_data.py
```
Additional files `train.csv`, `valid.csv` and `test.csv` will be created.

## Running
From the project's root directory run:
```
python src/run.py train|predict 1|2|3|4
```
Allowed arguments:

```
$ python src/run.py --help
usage: run.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-ts {96,128,160,192,224}]
              {train,predict} {1,2,3,4}

SNR Lego bricks classification 2019.

positional arguments:
  {train,predict}       Run mode.
  {1,2,3,4}             Task number: 1 - train FC layers only, 2 - train last
                        conv layer + FC, 3 - train whole model

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -ts {96,128,160,192,224}, --target_size {96,128,160,192,224}
                        Target input image size.
```
