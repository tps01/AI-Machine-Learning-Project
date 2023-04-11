Instructions on running classifier.py
A description of our implementation of key functions can be found in project3.pdf.

In the command line, type:

python3 classifier.py <attributes> <training-set> <testing-set> <significance>

Note: if the significance value is left out of the command line arguments, the program will run but the tree will not prune.

*Make sure to keep all relevant files in the same folder/directory.

Example calls of the program:
python3 classifier.py house-attributes.txt house-votes-train.csv house-votes-test.csv 0.01
python3 classifier.py house-attributes.txt house-votes-train.csv house-votes-train.csv
python3 classifier.py house-attributes.txt house-votes-train.csv house-votes-test.csv

Invalid calls: 
python3 classifier.py
python3 classifier.py house-attributes.txt
