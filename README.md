# CYCASP, understanding ColonY growth and Cell Affect in SPatiotemporal experiments

abstract

CYCASP comprises three steps:
1. preprocessing (state of the art image preprocessing methods)
2. particle (finding, tracking, etc)
3. patch lineage graph ()

## Data

It ran for both real and synthetic data.
The real data is available under The Open Data Commons Attribution License (ODC-By) v1.0.

Schlueter, J. - P., McIntosh, M., Hattab, G., Nattkemper, T. W., and Becker, A. (2015). Phase Contrast and Fluorescence Bacterial Time-Lapse Microscopy Image Data. Bielefeld University. [doi:10.4119/unibi/2777409](http://doi.org/10.4119/unibi/2777409).

The synthetic data can be found under ...

## Particle diameter estimation




## Usage

```bash
# Set file permissions
$ chmod +x cycasp.py 

# Run CYCASP on a folder containing all image files 
# Formatted by channel : red, green, blue as c2, c3, c4 respectively for every time point
$ ./cycasp.py -i img_directory/

# Or on a CSV file containing particle positions and trajectory IDs 
# with default thresholds for the fives metrics (euclidean distance, channel specifc differences and time window)
$ ./cycasp.py -f filename.csv

#  -h, --help            show this help message and exit
#  -v, --version         show program's version number and exit
#  -i, --input           run CYCASP on the supplied directory
#  -f, --file            run only step 3 of the method given as a CSV file
#  -d                    diameter estimate (default 11)
#  -e                    euclidean distance (default 10, cf. section above)
#  -r -g -b              channel specific differences (default 50 for each)
#  -t                    time window for merges (default 10 time points)

```
## Dependencies

For better reproducibility the versions that were used for development are mentioned in parentheses.

* Python (2.7.11)
* OpenCV (3.1.0-dev)
* pyqtgraph (0.9.10)
* trackpy (u'0.3.0rc1')
* networkx (1.9.1)
* Scipy (0.16.0)
* pandas (0.16.2)
* json (2.0.9)

## License
```
The MIT License (MIT)

Copyright (c) Georges Hattab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
```
