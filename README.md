# Lightweight gait based authentication technique for IoT using subconscious level activities
This is pythond code for the paper work published in 2018 IEEE 4th World Forum on Internet of Things (WF-IoT). You can access to the paper through this [link](https://ieeexplore.ieee.org/document/8355210)

## Prerequisites
- Language: Python
- Required Packages: pandas, matplotlib, numpy, sklearn, scipy, tensorflow
- To install the required packages except tensorflow, type the following command:
1) Python 2
```
pip install numpy pandas matplotlib sklearn scipy
```
2) Python 3
```
pip3 install numpy pandas matplotlib sklearn scipy
```
For installation of tensorflow, please visit the official tensorflow webpage [link](https://www.tensorflow.org/install)

## Data Collection Process & Structure
- accelerometer (x,y, and z) and gyroscope (x, y, and z) from 12 participants
- asked participants to walk approximately 45 seconds, wearing smartwatch
- data sampling rate: 100 Hz


## Running the code
1) Run Deep Neural Network for User Classification
```
python3 DNN.py
```
2) Run K-Nearest Neighbor classifier for User Classification 
```
python3 KNN.py
```
3) Run Random Forest Classifier for User Classification 
```
python3 RFC.py
```
