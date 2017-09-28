---
title: "Installing Caffe on Ubuntu (CPU-ONLY)"
header:
  teaser: teaser.jpg
categories:
  - Project
tags:
  - machine-learning
  - deep-learning
  - caffe
  - installation
  - environment
  - essential
---

First, to tell you guys the truth, I had no intention to write this post. You know, because I actually don't have much experience with Caffe. And I am not some kind of experienced tech-guy who can deal with almost developing environment, either.

Remember I told you about how to prepare your AWS g2.x2large instance for Machine Learning? You can have a look at this post [here](https://chunml.github.io/ChunML.github.io/project/Prepare-AWS-Instance/){:target="_blank"}. I recommended you to use AWS's instance because of the fact that most of us can't afford a proper desktop to run Machine Learning stuff. But of course it is always a good idea to set things up similarly on your local machine. Why? Because we can do all the coding on your machine (like configure your network architecture, or test if your codes can be compiled without errors, etc), and just leave the heavy tasks for the GPU instance. It would help you save some money.

With that thought, I decided to install Caffe on my laptop. And guess what? It took me almost a whole working day to finish! What a shame, I thought. But soon I realize how experienced I became after struggling with it. You know, I should have been spending my Saturday wandering around with my guys without thinking a damn thing about Machine Learning related stuff. But I didn't regret a little bit, not even a little bit. Because at least I got it done. And I'm here to share my experience with all you guys.

My laptop is running on Ubuntu 16.04, with OpenCV 3.0.0 installed. The other versions of Ubuntu should work as well, because it is the Caffe and the necessary dependencies matter, not the Ubuntu version.

As you may already knew, Caffe is a powerful framework written in C++ for implementing Deep Neural Network and it is being used almost everywhere out there. For that reason, you can just type *Caffe installation* in Google search bar, and you can find a lot of websites telling you how to do it. But sometimes it may not work to someone, just like my case. So what is the reason then?

First, Caffe does require a lot of necessary dependencies to make it work. And errors occur mostly because of dependencies incompatibility.

As you may be recommended by other sites that you should use *Anaconda*, one distribution of Python which can self-manage packages installation. I am not saying that *Anaconda* is bad. In fact I used it a lot in the past, when I had very little experience in Python package installation. It did help me do most of the installing things.

So what is the problem? In my case, having both Python and Anaconda installed caused some kind of a mess, and Caffe struggled with finding the appropriate libraries it needed. Note that Anaconda itself is a completely seperate Python distribution, which means I had two version of Python installed in my machine!

So I decided to remove *Anaconda* from my disk to try some luck. And guess what? The problem was **SOLVED**! So the first thing I want you to try is: if you have both Python and Anaconda installed, try to remove *Anaconda* first.

{% highlight Bash shell scripts %}
sudo rm -rf ~/anaconda2
{% endhighlight %}

Note that you will have to change the path according to your *Anaconda* path. Next, we will install Boost:

{% highlight Bash shell scripts %}
sudo apt-get install -y --no-install-recommends libboost-all-dev
{% endhighlight %}

In the next step, we will install all the necessary packages for Caffe:

{% highlight Bash shell scripts %}
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev \
libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
{% endhighlight %}

Next, clone Caffe repository, and make a copy of file *Makefile.config.example*:

{% highlight Bash shell scripts %}
git clone https://github.com/BVLC/caffe
cd caffe
cp Makefile.config.example Makefile.config
{% endhighlight %}

Next, we will have to install all the necessary Python packages, using *pip*. Navigate to *python* folder, and type the line below:

{% highlight Bash shell scripts %}
sudo pip install scikit-image protobuf
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done
{% endhighlight %}

Note that I got to add *sudo* to make it work on my laptop. Because what it was supposed to install directly to the real environment.

Next, we have to modify the *Makefile.config*. Uncomment the line *CPU_ONLY := 1*, and the line *OPENCV_VERSION := 3*. I also commented out the lines which indicate using Anaconda just to make sure that Anaconda doesn't mess up with the installation:

{% highlight Bash shell scripts %}
cd ..
sudo vim Makefile.config

# CPU-only switch (uncomment to build without GPU support).
CPU_ONLY := 1
...
# Uncomment if you're using OpenCV 3
OPENCV_VERSION := 3
...
# ANACONDA_HOME := $(HOME)/anaconda2
...
# PYTHON_LIB := $(ANACONDA_HOME)/lib
{% endhighlight %}

Now you got everything ready. Let's make it:

{% highlight Bash shell scripts %}
make all
{% endhighlight %}

It will take a while to finish. But you would probably get some error message like below:

{% highlight Bash shell scripts %}
CXX src/caffe/net.cpp
src/caffe/net.cpp:8:18: fatal error: hdf5.h: No such file or directory
compilation terminated.
Makefile:575: recipe for target '.build_release/src/caffe/net.o' failed
make: *** [.build_release/src/caffe/net.o] Error 1
{% endhighlight %}

Seems like a mess, huh? It is because *hdf5* library and *hdf5_hl* library actually have a postfix *serial* in their names, the compiler cannot find them. To fix this, we just have to make a **link** to the actual files. Remember, we are not changing their names!

But first, let's check out the actual name of the libraries. It may vary on your machines, though.

{% highlight Bash shell scripts %}
cd /usr/lib/x86_64-linux-gnu/
ls -al

...
libhdf5_serial.so.10.1.0
libhdf5_serial_hl.so.10.0.2
...
{% endhighlight %}

You may find the two files like above. Note again that the version may be different. Just take note the ones you saw. Then we will make a link to them:

{% highlight Bash shell scripts %}
sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.1.0 /usr/lib/x86_64-linux-gnu/libhdf5.so
sudo ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
{% endhighlight %}

And note that the postfixes of *hdf5* and *hdf5_hl* are not always the same. In my case, I assumed that they are the same (*10.1.0*) and it made me pay the price with another error (of course it took me another while to figure out what I did wrong).

After doing that, try *make all* again, this time there should be no more errors!

Next, we will run two other commands:

{% highlight Bash shell scripts %}
make test
make runtest
{% endhighlight %}

And these two should be working as well. And you will likely see the result like below:

![Success](/images/projects/installing-caffe-cpu-only/success.png)

If you saw something similar, then Congratulations! You have successfully installed Caffe! Now you can get your hands dirty with some real Deep Neural Network projects and become a part of Caffe community!

Next step is optional but I highly recommend because we are using Python for our works. We will compile the Python layer so that we can use *caffe* directly in our Python source code.

{% highlight Bash shell scripts %}
make pycaffe
{% endhighlight %}

Here, most of your machine will compile without error. But someone may see some error like below (I did too).

{% highlight Bash shell scripts %}
CXX/LD -o python/caffe/_caffe.so python/caffe/_caffe.cpp
python/caffe/_caffe.cpp:10:31: fatal error: numpy/arrayobject.h: No such file or directory
compilation terminated.
Makefile:501: recipe for target 'python/caffe/_caffe.so' failed
make: *** [python/caffe/_caffe.so] Error 1
{% endhighlight %}

The error indicates that it can not find a header file named *arrayobject.h*. It was caused because *numpy* was installed in a different path, and we must manually point to it. Actually, this problem was solved at the time of writing, but the installation path varies, so not everyone will get through it. For ones who encountered the error above, all you have to do is to make a small change to your *Makefile.config* from this:

{% highlight Bash shell scripts %}
PYTHON_INCLUDE := /usr/include/python2.7 \
/usr/lib/python2.7/dist-packages/numpy/core/include
{% endhighlight %}

to this:

{% highlight Bash shell scripts %}
PYTHON_INCLUDE := /usr/include/python2.7 \
/usr/local/lib/python2.7/dist-packages/numpy/core/include
{% endhighlight %}

After that, let's do *make pycaffe* again, and it should work now. Next, we will have to add the module directory to our $PYTHONPATH by adding this line to the end of *~/.bashrc* file.

{% highlight Bash shell scripts %}
sudo vim ~/.bashrc

export PYTHONPATH=$HOME/Downloads/caffe/python:$PYTHONPATH
{% endhighlight %}

Note that you have to change your *caffe* directory accordingly. We are nearly there, next execute the command below to make things take effect:

{% highlight Bash shell scripts %}
source ~/.bashrc
{% endhighlight %}

At this time, you can import *caffe* in Python code without any error. Not so hard, right?

{% highlight Bash shell scripts %}
python
>>> import caffe
>>>
{% endhighlight %}

But we are not done yet. Caffe provides us some examples of the most well-known models. We will use the LeNet model to train the MNIST dataset. Everything was already set up. All we have to do is just make it work:

{% highlight Bash shell scripts %}
cd ~/Downloads/caffe
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
{% endhighlight %}

Note that you have to change the path to where you put *caffe* accordingly. Next, we will train the model:

{% highlight Bash shell scripts %}
./examples/mnist/train_lenet.sh
{% endhighlight %}

At this point, there should be an error telling you that it was trying to use GPU in CPU-only configuration:

{% highlight Bash shell scripts %}
I1009 18:51:42.646926 22536 caffe.cpp:217] Using GPUs 0
F1009 18:51:42.647065 22536 common.cpp:66] Cannot use GPU in CPU-only Caffe: check mode.
*** Check failure stack trace: ***
    @     0x7fd00383f5cd  google::LogMessage::Fail()
    @     0x7fd003841433  google::LogMessage::SendToLog()
    @     0x7fd00383f15b  google::LogMessage::Flush()
    @     0x7fd003841e1e  google::LogMessageFatal::~LogMessageFatal()
    @     0x7fd003c38c00  caffe::Caffe::SetDevice()
    @           0x40ad33  train()
    @           0x4071c0  main
    @     0x7fd0027b0830  __libc_start_main
    @           0x4079e9  _start
    @              (nil)  (unknown)
Aborted (core dumped)
{% endhighlight %}

Well, that's OK. We just have to apply a tiny fix to the file *examples/mnist/lenet_solver.prototxt*, replace *GPU* with **CPU**, and save it. Try to run the command above again, then everything should work just fine!

It will take a while to finish the training (maybe long since we are using CPU). When it completes, you may see something like this:

![Training](/images/projects/installing-caffe-cpu-only/training.png)

Your result may vary but we will likely achieve an accurary value of approximately 99%. Congratulations again! Caffe is now working perfectly on your machine!

After today's post, I hope that you all had Caffe installed successfully on you machines. But if you still had troubles installing Caffe, like some frustrating errors keep coming or something, then feel free to leave me a comment below, and I will try to help you figure out why. See you!
