# SEEVIS, (S)egmentation-Fr(EE) (VIS)ualisation

A data driven and segmentation-free approach to extract and visualise the positions of cells' features as a lineage tree by using space-time cubes. 
This work was funded by the German-Canadian DFG International Research Training Group GRK 1906/1 and the “Phenotypic Heterogeneity and Sociobiology of Bacterial Populations” DFG SPP1617.

## Data

The employed datasets are available under The Open Data Commons Attribution License (ODC-By) v1.0.

Schlueter, J. - P., McIntosh, M., Hattab, G., Nattkemper, T. W., and Becker, A. (2015). Phase Contrast and Fluorescence Bacterial Time-Lapse Microscopy Image Data. Bielefeld University. [doi:10.4119/unibi/2777409](http://doi.org/10.4119/unibi/2777409).

## Dependencies

For better reproducibility the versions that were used for development are mentioned in parentheses.

* Python (2.7.11)
* OpenCV (2.4.12)
* pyqtgraph (0.9.10)
* trackpy (u'0.3.0rc1')
* pims (0.2.2)
* pandas (0.16.2)

## Usage

```bash
# Set file permissions
$ chmod +x seevis.py 

# Run SEEVIS on a folder containing all image files 
# Formatted by channel : red, green, blue as c2, c3, c4 respectively for every time point
$ ./seevis.py -i img_directory/

# Or on a CSV file containing feature positions
$ ./seevis.py -f filename.csv -s 2

#  -h, --help            show this help message and exit
#  -v, --version         show program's version number and exit
#  -i, --input           run SEEVIS on the supplied directory
#  -f, --file            run the Visualisation of SEEVIS
#  -s                    run colour scheme ranging from 1 to 4 (default is 1)
```

## License
```
The MIT License (MIT)

Copyright (c) 2016 Georges Hattab

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
