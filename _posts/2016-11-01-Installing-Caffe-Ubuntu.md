---
title: "Installing Caffe on Ubuntu 16.04 (GPU Mode with CUDA)"
header:
  teaser: projects/building-desktop-for-deep-learning/gpu.JPG
categories:
  - Project
tags:
  - machine-learning
  - deep-learning
  - caffe
  - installation
  - gpu
  - cuda
  - cudnn
  - environment
  - essential
---

It's great to be with all you guys again in today's post. As you already knew, it's been a while since I built my own desktop for Deep Learning. Or for ones who missed that post, you can have a look at my build here: 

* [Building up my own machine for Deep Learning](https://chunml.github.io/ChunML.github.io/project/Building-Desktop-For-Deep-Learning/){:target="_blank"}

And in the last few days, I was like a kid who had just received some new toys from his parents (I bought my desktop by my own money, though). I was so excited that I couldn't wait any longer to get started. So right after I put all the parts right into their places, the first thing I got to do was installing the OS, of course. I'm running Ubuntu 16.04 on my laptop, so I couldn't find any reason for not installing the latest Long-Term-Support version of Ubuntu on my desktop. The OS installation was quite easy, especially Ubuntu or any Linux based OS. The next thing to do was to install the necessary drivers. Actually nearly all the drivers were installed during the installation of Ubuntu, so I only had to manually install the GTX 1070 driver, but it was a piece of cake and you would laugh at me if I write it down here. In this post, I want to talk about the three main points below:

* Installing Caffe on Ubuntu 16.04 in GPU Mode
* Comparing the performance between CPU and GPU using MNIST and CIFAR-10 datasets

As you may notice that I once talked about the first one in my previous posts. Actually I didn't have myself a desktop with GPU in it, so that post was mainly about how to make things work only by using CPU. And obviously I can't just do the same thing this time if I want the GTX 1070 to be on the field. In short, there's a great deal of extra work to do if you want to make use the power of your GPU. And in this post I'm gonna show you how.

### Installing Caffe on Ubuntu 16.04 in GPU Mode

The first thing to do before installing Caffe was to install OpenCV, because I wanted to compile Caffe with OpenCV. Installing OpenCV wasn't a big deal at all. You can refer at my previous post here:

* [Installing OpenCV & Keras](https://chunml.github.io/ChunML.github.io/tutorial/Setting-Up-Python-Environment-For-Computer-Vision-And-Machine-Learning/){:target="_blank"}

To make it more convenient for you without having to switch between your browser tabs, I think it's better if I write out the steps for installing OpenCV in this post, too.

First, because I got in my desktop a fresh and new OS, I had to perform the commands below to make sure everything is updated to the latest version:

{% highlight Bash shell scripts %}
sudo apt-get update
sudo apt-get upgrade
{% endhighlight %}

Type the password if prompted. When the process completes, let's install all the necessary packages:

{% highlight Bash shell scripts %}
sudo apt-get install build-essential cmake git pkg-config libjpeg8-dev \
libjasper-dev libpng12-dev libgtk2.0-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev gfortran
sudo apt-get install libtiff5-dev 
{% endhighlight %}

And we need a library for computing optimization purpose. We will use BLAS just like before:

{% highlight Bash shell scripts %}
sudo apt-get install libatlas-base-dev
{% endhighlight %}

Next, we will install **pip**, a useful tool for managing all Python packages:

{% highlight Bash shell scripts %}
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
{% endhighlight %}

In order to use OpenCV and Caffe, we need to install Python Development Tools package:

{% highlight Bash shell scripts %}
sudo apt-get install python2.7-dev
{% endhighlight %}

And of course, a powerful module for dealing with arrays, Numpy:

{% highlight Bash shell scripts %}
pip install numpy
{% endhighlight %}

In the next step, let's download OpenCV 3.0 and its contribution modules:

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

Note that you have to tell **git** to checkout to *3.0.0* branch. Now we have everything ready, let's go and make it:

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

It should take a while to complete the installation. Try to have yourself a cup of coffee or something, because we are just half way there, lol.

After the installation finishes, let's check if everything works:

{% highlight Bash shell scripts %}
python
>>> import cv2
>>> cv2.__version__
'3.0.0'
{% endhighlight %}

If you got a result like above, then OpenCV 3.0 was successfully installed on your machine. If something goes wrong, try to do it all over again, this time make sure that you don't miss any line above. If you still can't make it work, please let me know by dropping me a line below. I'll be glad to help.

Unlike the previous post, I will skip the installation of Keras this time, and focus on installing Caffe instead. If you want to install Keras, please give the link above a look.

So now we have OpenCV 3.0 successfully installed. Next we will continue with Caffe. I'm assuming that you have at least one GPU installed. If you don't, please refer to the post below:

* [Installing Caffe on Ubuntu (CPU-ONLY)](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/){:target="_blank"}

This time we want to make use the power of GPU, we can tell Caffe that we want to use GPU, by commenting out the *CPU_ONLY* option, do you remember that? Unfortunately, it's not that simple. Caffe is just a framework which helps us handle the Network, which means that with Caffe, we can define the Network structure, we can define rules, then Caffe will train and evaluate our Model. In fact, Caffe makes use of CUDA, a superb library provided by NVIDIA, to handle the communication with our GPU.

So, in the next step, we will install the CUDA Toolkit. Let's go to the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads){:target="_blank"}, choose your OS, the OS Distribution and version carefully. The rest is simple, just follow the guide on the download page, and it's done. The installation file's size is pretty large, so it's likely to take a while, so don't lose your patience, lol.

![cuda](/images/projects/installing-caffe-ubuntu/cuda.png)

Next, we will download cuDNN, which is a GPU-accelerated library of primitives for deep neural networks provided by NVIDIA. With cuDNN, the computation speed will be significantly accelerated. All we have to do is going to [cuDNN home page](https://developer.nvidia.com/cudnn){:target="_blank"}, register to the *Accelerated Computing Developer Program* (it's free, but it's a must to download cuDNN). After registering and completing their short survey, you will be redirected to the download page like below:

![cudnn](/images/projects/installing-caffe-ubuntu/cudnn.png)

You may want to download the latest version, as NVIDIA claimed that it's much faster than its predecessor. If you don't have any intention to play around with Faster R-CNN, then you can grab the latest version. But if you want to play around with the most outstanding Object Detection algorithm out there, then I highly recommend you to choose the v4 Library. I will tell you why in later post. Before downloading, make sure that you choose the right version for Linux, the upper most one below the install guide:

![cudnn_down](/images/projects/installing-caffe-ubuntu/cudnn_down.png)

After the download process completes, let's extract the downloaded file (assuming that you're placing it under Downloads folder):

{% highlight Bash shell scripts %}
cd ~/Downloads
tar -xvf cudnn-7.0-linux-x64-v4.0-prod.tgz
{% endhighlight %}

In the next step, you just have to copy the two extracted folders to where CUDA was installed, which is most likely at */usr/local/cuda*:

{% highlight Bash shell scripts %}
sudo cp lib64/* /usr/local/cuda/lib64
sudo cp include/* /usr/local/cuda/include
{% endhighlight %}

That's it. You have just successfully installed CUDA and cuDNN. Let's move on and install Caffe. There won't be much difference with the installation in CPU_ONLY mode.

First, we have to install all the necessary packages and libraries:

{% highlight Bash shell scripts %}
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev \
libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
{% endhighlight %}

Next, let's go to BVLC GitHub repository and grab the latest version of Caffe:

{% highlight Bash shell scripts %}
git clone https://github.com/BVLC/caffe
cd caffe
{% endhighlight %}

Then we will create the *Makefile.config* file from the example file, just like before:

{% highlight Bash shell scripts %}
cp Makefile.config.example Makefile.config
{% endhighlight %}

Let's apply those modifications below:

{% highlight vim %}
# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# Uncomment if you're using OpenCV 3
OPENCV_VERSION := 3

# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_INCLUDE := /usr/include/python2.7 \
        /usr/local/lib/python2.7/dist-packages/numpy/core/include

# Uncomment to support layers written in Python (will link against    Python libs)
WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/
{% endhighlight %}

If you went through my post about installing Caffe in CPU_ONLY mode before, then all the modifications above should sound familiar with you. If you didn't, you might want to take a look at that post to know why we have to make those changes. The only difference with what we did in the previous post is, instead of uncommenting the *CPU_ONLY := 1* line, we uncomment the *USE_CUDNN := 1* to take advantage of cuDNN.

At this point, we can go through the compilation of Caffe without any error:

{% highlight Bash shell scripts %}
make all & make test && make runtest && make pycaffe
{% endhighlight %}

Next, in order to use Caffe inside our Python code, we have to add pycaffe to the PYTHONPATH:

{% highlight Bash shell scripts %}
sudo vim ~/.bashrc

export PYTHONPATH=$HOME/Downloads/caffe/python:$PYTHONPATH
{% endhighlight %}

Then execute the command below to make the change take effect:

{% highlight Bash shell scripts %}
source ~/.bashrc
{% endhighlight %}

Now let's check if we have things work properly:

{% highlight Bash shell scripts %}
python
>>> import caffe
>>>
{% endhighlight %}

If it don't show any import error, then Congratulations, you have just successfully installed Caffe. The installation itself was confusing a little, but it didn't require any complicated modifications, so somehow we still made it till the end. We can finally exhale now, lol.

### Comparing the performance between CPU and GPU

So we have Caffe compiled, and with the support from CUDA & cuDNN, we can take avantage of our GPU to speed up the learning process significantly. But, that's just what we have been told so far. When we speak about the performance term, the words "good", "faster", "much faster" or even "significantly faster" are way too subtle and not much informative. In order to answer the question "How faster?", it's better to consider the difference in computing time between CPU Mode and GPU Mode. I will use two datasets which Caffe provided the trained models: MNIST and CIFAR-10 for comparing purpose. Note that in this post, I just consider the size of the dataset for simplicity, without considering the complexity of the Networks. I will dig more further about it on future posts on Neural Network.

* MNIST Dataset

First, make sure you are in the root folder of Caffe, and run the commands below to download the MNIST dataset:

{% highlight Bash shell scripts %}
cd $CAFFE_ROOT
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
{% endhighlight %}

That's all we have to do to prepare the data. Let's see how much time the CPU need to run each iteration:

{% highlight Bash shell scripts %}
./build/tools/caffe time --model=examples/mnist/lenet_train_test.prototxt
{% endhighlight %}

And here's my result on my Intel Core i7-6700K CPU:

![mnist_cpu](/images/projects/installing-caffe-ubuntu/mnist_cpu.png)

As you can see, my CPU took approximately 55ms to run each iteration, in which 23ms for forward pass and 32ms for backward pass. Let's go ahead and see if the GPU can do better:

{% highlight Bash shell scripts %}
./build/tools/caffe time --model=examples/mnist/lenet_train_test.prototxt
{% endhighlight %}

And here's the result on my GTX 1070.

![mnist_gpu](/images/projects/installing-caffe-ubuntu/mnist_gpu.png)

The result came out nearly right after I hit Enter. I was really impressed, I admit. Each iteration took only 1.7ms to complete, in which 0.5ms for forward pass and 1.2ms for backpropagation. Let's do some calculation here: the computing time when using GPU is roughly 32 times faster than when using CPU. Hmm, not so bad, you may think. 

Because MNIST dataset is pretty small in size, which each example is just a 28x28 grayscale image, and it contains only 70000 images in total, the CPU still can give us an acceptable performance. Also note that in order to make use of the power of GPU, our computer has to take some times to transfer data to the GPU, so with small dataset and simple Network, the difference between the two may not be easily seen.

Let's go ahead and give them a more challenging one.

* CIFAR-10 Dataset

CIFAR-10 is way larger comparing to MNIST. It contains 60000 32x32 color images, which means CIFAR-10 is roughly three times larger than MNIST. That's a real challenge for both to overcome, right?

Just like what we did with MNIST dataset, let's first see how much time it takes using CPU:

{% highlight Bash shell scripts %}
./build/tools/caffe time --model=examples/cifar10/cifar10_full_train_test.prototxt
{% endhighlight %}

And here's the result I got:

![cifar_cpu](/images/projects/installing-caffe-ubuntu/cifar10_cpu.png)

As you can see, with a larger dataset (and a more complicated Network, of course), the computing speed was much slower comparing with MNIST dataset. It took approximately 526ms to complete one iteration: 238ms for forward pass and 288ms for backward pass. Let's go ahead and see how well the big guy can do:

{% highlight Bash shell scripts %}
./build/tools/caffe time --model=examples/cifar10/cifar10_full_train_test.prototxt --gpu 0
{% endhighlight %}

And the result I had with my GTX 1070:

![cifar_gpu](/images/projects/installing-caffe-ubuntu/cifar10_gpu.png)

Look at the result above. Unlike the significant decrease in performance as we saw when running on CPU, my GTX 1070 still brought me an impressive computing speed. It took only 11ms on one iteration, in which 3ms for forward pass and 8ms for backpropagation. So when running on CIFAR-10 dataset, the GPU really did outperform the CPU, which computed 48 times faster. Imagine you are working with some real large dataset in real life such as ImageNet, using GPU would save you a great deal of time (let's say days or even weeks) on training. The faster you obtain the result, the more you can spend on improving the Model. That's also the reason why Neural Network, especially Deep Neural Network, has become the biggest trend in Machine Learning after long time being ignored by the lack of computing power. Obviously not only nowadays, but Deep Neural Network will continue to grow in the future.

### Summary

So in this post, I have just shown you how to install OpenCV and Caffe in GPU Mode with CUDA Toolkit and cuDNN. I really appreciate that you made it to the end with patience. I hope that this post can help you prepare the necessary environment for your Deep Learning projects. 

And I also did some comparison on performance between GPU and CPU using two most common datasets: MNIST and CIFAR-10. Through the results above, I think you can now see how using GPU on Deep Neural Network can bring up a big difference.

Finally, if you come across any compilation error, please kindly let me know. I'll try my best to help. Can't wait to see you soon, in the upcoming post.
