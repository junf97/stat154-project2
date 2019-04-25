# Stat 154 Spring 2019 Project 2

This repository holds the Project 2: Cloud Classification of Stat154, Spring 2019, at UC Berkeley.

* `code`: This folder contains python codes that associate with each part of the write-up. You may find a
          requirements.txt file there which specifies all required modules to install.
* `data`: This folder contains the raw image data for our classification models to use.
* `documents`: This folder contains all relevant documents, including project spec, paper, etc.
* `report`: This folder contains the final write-up (in both pdf and Word) for the project.

## Usage

To reproduce the report for this project, you can find the write-up in both pdf format and Word
format (to re-generate the report if you want) in the `report` folder. The Word document is
created using [Microsoft Office 365](https://www.office365.com/). Previous versions of Word may not be supported.

To reproduce all images and data included in the write-up, you will need to
first put the raw image data from MISR sensors into the `data` folder according to the format specified
in Table 1 of `documents/project2.pdf`, and then run the Python code associated with each section
in the `code` folder. [Python 3.7.3](https://www.python.org/downloads/release/python-373/) (with [pip 19.1](https://pip.pypa.io/en/stable/))
is used for coding and all third-party libraries that were used are listed in `code/requirements.txt`.

The CVgeneric function is included separately in `code/sec_2d.py`.

## Authors

* **Junfang Jiang** - [junfang@berkeley.edu](mailto:junfang@berkeley.edu)
* **Kaiqian Zhu** - [tim3212008@berkeley.edu](mailto:tim3212008@berkeley.edu)
