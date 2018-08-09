# Semantic Segmentation
### Introduction
In this project, I label the pixels of a road in images using a Fully Convolutional Network (FCN).

### What is semantic segmentation?
Semantic segmentation is understanding an image at pixel level, using a Fully Convolutional Neural Network. Generally, when we are using a neural network for classification, we consider the image as a whole and classify the full image to a certain category. In semantic segmentation, we classify each pixel of the image to a certain category.

### Implementation
I started following the guidelines of the walkthrough video by Udacity. It gave me all the basics. The class introduced a pre-trained VGG-16 network that had to be converted to a fully convolutional network. The final fully connected layer need a 1x1 convolution and the depth had to be equal to the number of desired classes. In this case the classes were only two (road and not-road).
The entire architecture is in this lines of code:
```
conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2), padding='same',       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
conv_1x1_2 =tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
layer_add = tf.add(output, conv_1x1_2)
output_2 =  tf.layers.conv2d_transpose(layer_add, num_classes, 4, strides=(2, 2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
layer_add_2 = tf.add(output_2, conv_1x1_3)
nn_last_layer = tf.layers.conv2d_transpose(layer_add_2, num_classes, 16, strides=(8, 8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
```

The hyperparameter I used are:
- epochs: 50
- batch size: 8
- keep probability: 0.75
- learning rate: 0.0009

I tried different parameters and these are the results:

| epochs  | batch | learning rate  | keep probability | LOSS |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 10  | 1 | 0.0009 | 0.5 | **0.20** |
| 5  | 5 | 0.0009 | 0.5 | **0.30** |
| 50  | 5 | 0.0009 | 0.5 | **0.035** |
| 20  | 16 | 0.0009 | 0.5 | **0.18** |
| 50  | 8 | 0.0009 | 0.75 | **0.056** |

### Examples
![First example](/images/um_000014.png)
![Second example](/images/um_000015.png)
![Third example](/images/um_000021.png)
![Fourth example](/images/um_000022.png)
![Fifth example](/images/um_000029.png)
![Sexth example](/images/um_000035.png)

### Reflections
The FCN network classifies over 80% of the road as road, and less then 20% of nor road as road. There is still room for improvement. Probably a lower learning rate could be helpful. The epochs and the batch size I think are ok.
I used my GPU (GeForce GTX 1070) for this project. Every training requires between 10 and 30 minutes. Using a faster GPU on the cloud, can decrease this time and, so, parameter tuning could be easier.

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
