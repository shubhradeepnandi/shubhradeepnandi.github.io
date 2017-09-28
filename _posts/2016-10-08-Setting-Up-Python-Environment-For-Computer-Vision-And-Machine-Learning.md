---
title: "Machine Learning Part 4: Setting Up Python Environment for Computer Vision and Machine Learning"
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

So, I am here with you again, in the new post of Machine Learning tutorial series. Next time, I told you about Linear Regression and promised to talk further more in the next post. And I am also gonna ask you to do some coding, remember? But before doing that, we need more than our hands. If you are gonna have a show, then you must have the stage prepared, right? Similarly, if you want to write some lines of code, then you need to prepare the developing environment. Today, I will show you how to do that.

In this post, I am assuming that you are using a computer running on Linux based OS (like Ubuntu, Red Hat, ect). For ones who are sticking to Windows, I am so sorry. I will write a post for Windows users soon.

We will install OpenCV for Computer Vision, and Keras (Theano backend) for Machine Learning. I will make it short and clear, just to make sure that you will have a working environment, without spending so much time on struggling with unknown errors.

**1. INSTALL OPENCV**

Firstly, open your *Terminal*, and type the lines below to Update and Upgrade the packages:

{% highlight Bash shell scripts %}
sudo apt-get update
sudo apt-get upgrade
{% endhighlight %}

From now on, just hit *y* when prompted. Next, we will install the necessary packages for OpenCV:

{% highlight Bash shell scripts %}
sudo apt-get install build-essential cmake git pkg-config libjpeg8-dev \
libjasper-dev libpng12-dev libgtk2.0-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev gfortran
sudo apt-get install libtiff5-dev 
{% endhighlight %}

Next, we will need library to optimizing purpose (there will be a lot of simultaneous computation in Computer Vision and Machine Learning which mainly bottleneck the performance). We will use ATLAS (among many other BLAS implementation):

{% highlight Bash shell scripts %}
sudo apt-get install libatlas-base-dev
{% endhighlight %}

After that, we will install *pip*, a tool used for installing Python packages by PyPA:

{% highlight Bash shell scripts %}
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
{% endhighlight %}

In the next step, we will need to install *virtualenv* and *virtualenvwrapper*, which are tools for isolating Python environments. You may ask why we have to do that. Imagine you are using Python for many purposes (like game developing, computer vision, machine learning, etc). Because each requires different configuration, it will be a good idea to work seperately to avoid confliction.

{% highlight Bash shell scripts %}
sudo pip install virtualenv virtualenvwrapper
{% endhighlight %}

To make *virtualenv* work, we will have to apply these lines to our *~/.bashrc* by typing *sudo vim ~/.bashrc* (I prefer Vim, you can use whatever you want). Then add these lines to the end of the file:

{% highlight Bash shell scripts %}
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
{% endhighlight %}

Now, let's create our virtual environment for installing OpenCV
{% highlight Bash shell scripts %}
source ~/.bashrc
mkvirtualenv opencv
{% endhighlight %}

From this step, make sure you are in *opencv* environment (you can know by seeing if there is *(opencv)* before the $ mark).

Next we need to install python development tools package:

{% highlight Bash shell scripts %}
sudo apt-get install python2.7-dev
{% endhighlight %}

Then we will install *numpy*, a very powerful module for dealing with array computation:

{% highlight Bash shell scripts %}
pip install numpy
{% endhighlight %}

**Note that you can not add *sudo* all the time. Because *sudo* means that you are execute as *superuser*, it will install directly into your system environment, not the virtual environment!**

Next, we will download **OpenCV**, then checkout to *3.0.0* branch:

{% highlight Bash shell scripts %}
cd ~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.0.0
cd ~
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.0.0
{% endhighlight %}

Now we got everything ready to be installed, let's compile and install **OpenCV**:

{% highlight Bash shell scripts %}
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..
  
make
sudo make install
sudo ldconfig
{% endhighlight %}

In the last step, we only need to link the installed **OpenCV** to our virtual environment:

{% highlight Bash shell scripts %}
cd ~/.virtualenvs/cv/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
{% endhighlight %}

Now let's test if **OpenCV** was installed successfully (make sure that you are still standing in *opencv* virtual environment):

{% highlight Bash shell scripts %}
python
>>> import cv2
>>> cv2.__version__
'3.0.0'
{% endhighlight %}

If you see the output like above, then Congratulations! You have successfully installed OpenCV! 

**2. INSTALL KERAS**

Now let's move on to install Keras. Actually we will almost use the *scikit-learn* library, not Keras in the near future, but Keras is not so hard to install, so I decided to add it here so that I won't have to make another post for Keras.

As I mentioned earlier, you may not want to mess up your environments. So let's create another virtual enviroment for Keras.

{% highlight Bash shell scripts %}
# Exit the current virtual environment
deactivate
mkvirtualenv keras
{% endhighlight %}

Now let's install necessary packages for Keras:

{% highlight Bash shell scripts %}
pip install numpy scipy scikit-learn pillow h5py
{% endhighlight %}

Keras actually works on top of Theano and Tensorflow (if you don't give a damn about what they are, then that's just fine!). Suppose that you are using Keras with Theano backend, so you will have to install Theano first:

{% highlight Bash shell scripts %}
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
{% endhighlight %}

Wait until the Theano installation completes, then we can install keras:

{% highlight Bash shell scripts %}
pip install keras
{% endhighlight %}

We have just finished our Keras installation. If you want to use OpenCV in *keras* virtual environment, you can link it the way we did above:

{% highlight Bash shell scripts %}
cd ~/.virtualenvs/keras/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
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

So now you have swords and shields. We are ready to be on the field. I know you guys don't want to wait no more. So I will post the next post soon, I promise. Before that, make sure you revise all the stuff about Linear Regression. And I will be back. See you!
