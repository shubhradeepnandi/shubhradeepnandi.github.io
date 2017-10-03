---
title: "Setting Up Python Environment Computer Vision and Machine Learning(Ubuntu)"
categories:
  - Tutorial
tags:
  - machine-learning
  - computer-vision
  - development
  - python
  - environment
  - essential
---

So, I am back, in the new post of Machine Learning tutorial series. Last time, I told you about Linear Regression and then promised to talk further more in the next post. And I am also gonna ask you to do some coding, remember? But before doing that, we need more than our hands. If you are gonna have a show, then you must have the stage prepared, right? Similarly, if you want to write some lines of code, then you need to prepare the developing environment. Today, I will show you how to do that.

In this post, I am assuming that you are using a computer running on Linux based OS (like Ubuntu, Red Hat, ect). For ones who are sticking to Windows, I am so sorry.

We will install OpenCV for Computer Vision, and Keras (Theano backend) for Machine Learning. I will make it short and clear, just to make sure that you will have a working environment, without spending so much time on struggling with unknown errors.

**1. INSTALL OPENCV**

Firstly, open your *Terminal*, and type the lines below to Update and Upgrade the packages:

{% highlight Bash shell scripts %}
sudo apt-get update
sudo apt-get upgrade
{% endhighlight %}

From now on, just hit *y* when prompted. Next, we will install the necessary packages for OpenCV:

Next, we will need library to optimizing purpose (there will be a lot of simultaneous computation in Computer Vision and Machine Learning which mainly bottleneck the performance). We will use ATLAS (among many other BLAS implementation):

{% highlight Bash shell scripts %}
sudo apt-get install libatlas-base-dev
{% endhighlight %}

After that, we will install *pip*, a tool used for installing Python packages by PyPA:

{% highlight Bash shell scripts %}
sudo apt-get install python3-pip 
{% endhighlight %}

Now, let's create our environment for installing OpenCV-Theano-Keras


Next we need to install python development tools package:

{% highlight Bash shell scripts %}
sudo apt-get install python3-dev
{% endhighlight %}

Then we will install *opencv*:

{% highlight Bash shell scripts %}
sudo apt-get install python-opencv
pip3 install opencv-python opencv-contrib-python
{% endhighlight %}


Now let's test if **OpenCV** was installed successfully (make sure that you are still standing in *opencv* virtual environment):

{% highlight Bash shell scripts %}
python
>>> import cv2
>>> cv2.__version__
'3.3.0'
{% endhighlight %}

If you see the output like above, then Congratulations! You have successfully installed OpenCV! 

**2. INSTALL KERAS**

Now let's move on to install Keras. Actually we will almost use the *scikit-learn* library, not Keras in the near future, but Keras is not so hard to install, so I decided to add it here so that I won't have to make another post for Keras.

Now let's install necessary packages for Keras:

{% highlight Bash shell scripts %}
pip install numpy scipy scikit-learn pillow h5py
{% endhighlight %}

Keras actually works on top of Theano and Tensorflow (if you don't give a damn about what they are, then that's just fine!). Suppose that you are using Keras with Theano backend, so you will have to install Theano first:

{% highlight Bash shell scripts %}
pip install theano
{% endhighlight %}

Wait until the Theano installation completes, then we can install keras:

{% highlight Bash shell scripts %}
pip install keras
{% endhighlight %}


Now let's test our installation. Again, make sure that you are still in *keras* enviroment.

{% highlight Bash shell scripts %}
python
>>> import keras
Using Theano backend.
{% endhighlight %}

If you had the exact output as above, then Congratulations again! Wait... what if you didn't, but received *Using TensorFlow backend* error instead. Well, sometimes Keras fails to choose the right backend, and you must do it manually. But don't worry, it should be just a piece of cake!

All you have to do is modify the file *~/.keras/keras.json*, replace 'TensorFlow' with 'Theano', then try to test it again. It should work just fine!


So now you have installed all you need for learning Computer Vision and Machine Learning. Great job guys! It was not as hard as you imagined right?

So now you have Tea Leaves, Sugar, and Milk, so lets make tea. We are ready to be on the field. I know you guys don't want to wait no more. So I will post the next post soon. Before that, make sure you revise all the stuff about Linear Regression. And I will be back. See you soon!
