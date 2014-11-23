# pynm
![build status](https://travis-ci.org/ohtaman/pynm.svg?branch=master)

pynm is a Python Machine Learning Library.
(under development yet.)

## Installation

To install pynm, simply:

```
$ sudo pip install git+https://github.com/ohtaman/pynm.git
```

or from source:

```
$ curl -LO https://github.com/ohtaman/pynm/archive/master.zip
$ unzip master.zip
$ cd pynm-master
$ sudp python setup.py install
```

## As a CommandLine Tool

- show help
```
$ pynm -h
usage: pynm [-h] {metric,bandit} ...

pynm Machine Learning.

optional arguments:
  -h, --help       show this help message and exit

commands:
  {metric,bandit}
    metric         Metric Learning
    bandit         Bandit Agent
```

- do metric learning
```
$ pynm metric -h
usage: pynm metric [-h] [-i FILE] (-l FILE | -p FILE) [-o FILE] [-m FILE]
                   [-w FILE] [-d DELIM] [-s] [--header] [-U DISTANCE]
                   [-L DISTANCE] [-S SLACK] [-N MAX]
                   {itml} ...

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --input_data FILE
                        input data file (default: stdin)
  -l FILE, --input_labels FILE
                        input labels file
  -p FILE, --input_pairs FILE
                        input pairs file
  -o FILE, --output_data FILE
                        output data file
  -m FILE, --output_metric FILE
                        output metric file
  -w FILE, --output_weights FILE
                        output weights file
  -d DELIM, --delimiter DELIM
                        delimiter (default: "\t")
  -s, --sparse          sparse format (not implemented yet)
  --header              has header
  -U DISTANCE, --u_param DISTANCE
                        U parameter (max distance for same labels, default:
                        1.0)
  -L DISTANCE, --l_param DISTANCE
                        L parameter (min distance for different labels,
                        default: 1.0)
  -S SLACK, --slack SLACK
                        slack variable (default: 1.0)
  -N MAX, --max_iteration_number MAX
                        max iteration (default: 1000)

algorithm:
  {itml}
    itml                Information Theoretic Metric Learning
``` 

- do bandit
```
usage: pynm bandit [-h] [-i FILE] [-o FILE] [-a {ucb1,epsilon,thompson}]

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --input FILE
                        input file
  -o FILE, --output FILE
                        output file
  -a {ucb1,epsilon,thompson}, --algorithm {ucb1,epsilon,thompson}
                        algorithm
```

## As a Python Library

```
>>> import pynm
```

### Data Handling

### Regression

### Clustering

### Classification

### Metric Learning

### Bandit

### Closs Validation
