# Hotels Ranking Adversarial


## Setup
To run the files the following is required:
1) Python with pip (compatible with versions 3.5 - 3.8) 
  * Can be downloaded here: https://www.python.org/
2) Tensorflow
  * Can be downloaded here: https://www.tensorflow.org/install/pip
3) Tensorflow Ranking
  * Can be downloaded here: https://github.com/tensorflow/ranking
4) Matplotlib
  * Use pip to install library
```
pip install matplotlib
```

5) Dataset Files
  * Use wget to download dataset. 
```
wget -O "vocab.txt" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/vocab.txt"
wget -O "train.tfrecords" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking/ELWC/train.tfrecords"
wget -O "test.tfrecords" "http://ciir.cs.umass.edu/downloads/Antique/tf-ranking//ELWC/test.tfrecords"
```
