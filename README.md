# Semantic Segmentation

### Reflection

Goal of this project is to use VGG-16 image classifier neural network converted tweaked to be fully convolutional to determine drivable areas on images from KITTI dataset.
Or, in other words, perform semantic segmentation task so image pixels will be divided in 2 classes: road and non-road.

Pretrained VGG-16 neural network were used as a base of architecture. Layers 3, 4, 7 and input were extracted as well as keep probability. Following layers were added to 
complete architecture:

1. 1x1 convolutional layer with `kernel=1` from VGG's Layer 7.
2. Upsampling (deconvolutional layer) with `kernel=4` and `stride=(2,2)` from layer #1.
3. 1x1 convolutional layer with `kernel=1` from VGG's Layer 4.
4. Skip connection layer from layer #2 to layer #3.
5. Upsampling layer with `kernel=4` and `stride=(2,2)` from layer #4.
6. 1x1 convolutional layer with `kernel=1` from VGG's Layer 3.
7. Skip connection layer from layer #5 to layer #6.
8. Upsampling layer with `kernel=16` and `stride=(8,8)` from layer #7.

For convolutional and upsampling layers random normal kerne linitializers with deviation of `0.01`
and kernel `L2` regularizer of `0.001` were used.

Softmax-cross-entropy function was chosen as a loss. Adam optimizer was used as it had not so many hyper-parameters and because of decaying learning rate.

Following hyper-parameters were used in training:

1. Keep probability of `0.5`.
2. Learning rate of `0.0001`.
3. Batch size of `2`.
4. Training epochs of `30`.

As a measure of how well semantic segmentation was done mIOU (mean interaction-over-union) metric (`tf.metrics.mean_iou()`) was chosen. Initial goal
was to reach level of `mIOU=0.8`. After `1`-st epoch of training level of `mIOU=0.604` was reached. Desired level was reached after `6`-th epoch
with `mIOU=0.807`. After `15`-th epoch of training level of `mIOU=0.921` was reached and `mIOU` started to grow very slowly from epoch to epoch. 
After final `30`-th epoch of training level of `mIOU=0.933` was reached. Training time was roughly `80` minutes using Udacity GPU-equipped workspace.
So the takeaway from training process is that `10-15` training epochs are enough to reach reasonably good level of `mIOU` and it will take `27-40` minutes
to finish training process (which is much less than `80`).

Some examples of results produced by trained network are shown below. It can be seen that in general network performs quite good but not perfect as it
struggles sometimes to determine drivable road in shadows and near cars and other objects on the road (for example, lane lines).

[image1]: ./pics/good-01.png ""
[image2]: ./pics/good-02.png ""
[image3]: ./pics/good-03.png ""
[image4]: ./pics/good-04.png ""
[image5]: ./pics/good-05.png ""
[image6]: ./pics/good-06.png ""
[image7]: ./pics/good-07.png ""
[image8]: ./pics/good-08.png ""
[image9]: ./pics/good-09.png ""
[image10]: ./pics/good-10.png ""
[image11]: ./pics/bad-01.png ""
[image12]: ./pics/bad-02.png ""
[image13]: ./pics/bad-03.png ""
[image14]: ./pics/bad-04.png ""
[image15]: ./pics/bad-05.png ""

##### Examples of images on which network performs well:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

##### Examples of images on which network performs not so good:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

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
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
