# ATM-Sunglasses-Detection

This project is designed to used in ATM for security reason and could recognize abnormal human faces

In recent years, the level of economic development in China has grown by leaps and bounds. Under the environment of continuous development of modern science and technology, many banks and financial institutions in China have widely promoted the use of ATMs. However, many criminals use masked faces to evade monitoring and investigation. In order to reduce the number and harmfulness of such criminal activities and protect the property safety of the general public, it is urgent to develop a sunglasses detection system that is used in ATM machines with high accuracy and good real-time performance.

the project is researched and developed on the basis of deep learning and convolutional neural network algorithms. The main research and development work completed by the projecy is as follows:
Firstly, the problem of detecting face from the surveillance video stream is solved. The Harr classifier algorithm and the Seetaface algorithm based on deep learning are tested, and the Seetaface algorithm with better speed is selected. The single face 640*640 picture face detection speed is only 0.2s.Then from the open-source face database and actual ATM monitoring video collected more than 20,000 sunglasses wearing positive and negative samples, trained in the classification model under the Caffe framework, the accuracy rate of up to 0.999 or even 1. Finally completed the entire detection system.
