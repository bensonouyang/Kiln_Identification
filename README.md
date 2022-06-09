# Deep Learning Algorithm to Recognize Kilns in Images

## Table of Contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Usage](#usage)
* [Source](#source)

## Introduction
According to United News of Bangladesh, the three top three sources of pollution in Bangladesh are cars, brick kilns, and construction. The Bangladeshi government has issued orders to move brick kilns away from schools or populated areas, and replace traditional brick kilns with more modern methods that do not pollute so much. For example, the United Nations Development Programme UNDP/GEF Project: Improving Kiln Efficiency in Brick Making Industry (GEF PIMS 1901) notes that "… the traditional brick making business in Bangladesh is an informal industry which uses old kiln design and coal as the main fuel … unrestricted concentration of brick kilns in certain regions in the country has given rise to a number of environmental issues." Identification of the locations of brick kilns in Bangladesh is required to reduce pollution in Bangladesh and follow government policy. In this project, we determine if chimney kilns are present in satellite imagery (differentiating between images without kilns, or with modernized versions of kilns such as zig-zag kilns and not chimney kilns). 

## Technologies
Project is created with:
* Python

Report is made with:
* R Markdown
* Latex

## Libraries
* Pandas
* Numpy
* Imageio
* PIL
* Tensorflow
* Keras
* Albumentations
* Scikit-learn

## Usage
To run the code and see results:
* Open [A-firstCNN.ipynb](https://github.com/bensonouyang/Kiln_Identification/blob/main/A-firstCNN.ipynb) in Jupyter Notebook. This file contains the results of a CNN with grayscale image data. 
* Open [B-featuresCNN.ipynb](https://github.com/bensonouyang/Kiln_Identification/blob/main/B-featuresCNN.ipynb) in Jupyter Notebook. This file contains the results of a CNN with coloured image data. 
* Open [C-imagesize_augmentation.ipynb](https://github.com/bensonouyang/Kiln_Identification/blob/main/C-imagesize_augmentation.ipynb) in Jupyter Notebook. This file contains the results of a CNN with coloured and augmented image data. 
* Open [D-augmentation.ipynb](https://github.com/bensonouyang/Kiln_Identification/blob/main/D-augmentation.ipynb) in Jupyter Notebook. This file contains the final results of a CNN with augmented image data with cross validation. 

To see the report:
* Open [proj3A-report.pdf](https://github.com/bensonouyang/Kiln_Identification/blob/main/proj3A-report.pdf) for PDF version
* Open [proj3A-report.Rmd](https://github.com/bensonouyang/Kiln_Identification/blob/main/proj3A-report.Rmd) to see how the report was made

## Source
[In class kaggle competition]([https://www.kaggle.com/dgawlik/nyse](https://www.kaggle.com/competitions/stat440-21-project3a))
