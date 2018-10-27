This converts data from the British National Corpus to a format that works for us. Use Python 3.

Convert an XML file to a text file to get once sentence per line:
$ python to-text.py **path to XML file** > sentences.txt

Create a file with sliding window batches and a unique word list based on the text file from the previous command:
$ python to-trainingdata.py < sentences.txt
