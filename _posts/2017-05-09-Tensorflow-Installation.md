---
title: "Tensorflow Implementation Note: Installing Tensorflow and Keras on Windows"
header:
  teaser: projects/tensorflow-install/tensorflow_logo.png
categories:
  - Project
tags:
  - machine-learning
  - deep-learning
  - keras
  - tensorflow
  - gpu
  - training
  - note
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hello everyone, it's been a long long while, hasn't it? I was busy fulfilling my job and literally kept away from my blog. But hey, if this takes any longer then there will be a big chance that I don't feel like writing anymore, I suppose. So it might not be a bad idea to just make some simple posts (some kind of implementation note like the one I'm writing right now), until life becomes easier, right?

That's enough talking. Let's get down to business. Today I'm gonna talk about some of my implementation notes when working with Tensorflow, not Tensorflow tutorial exactly :)

### What is Tensorflow anyway?

I think it's a little bit rude to say this but, maybe Google can handle that question way better than me. To make it short for you guys, Tensorflow is:

* open source library for deep learning (mainly)
* developed by Google Brain Team, released in 2015
* well documented
* supposed to replace Theano
* growing fast like hell! Very promising!

I remember when I tried Tensorflow last year, the documents were kind of mess since they weren't up-to-date with newest release, and as the result, the source code for tutorials didn't work as expected. But things have become way easier recently after they re-arranged Tensorflow's repositories and made documents up-to-date. Those it-was-supposed-to-run-but-it-didn't problem won't annoy you anymore.

Still can't find it attracting? You can find more on Google and Youtube for more interesting presentations of Tensorflow. And as I don't intend to make this post unnecessarily long, please DIY :)

### Installing Tensorflow and Keras

For Unix users, there shouldn't be any problems installing both Tensorflow and Keras, I believe, if you follow the instructions on their pages. Just one thing: Tensorflow has two seperate versions for CPU-only and GPU-accelerated PC, so don't download the wrong one, or you will end up in trouble!

* **Tensorflow installation:**
[Tensorflow installation](https://www.tensorflow.org/install/)
* **Keras installation:**
[Keras installation](https://keras.io/#installation)

For Windows users, installing Tensorflow can be done with ease, just like on Linux machine, you can install Tensorflow just by one single command. But it's a little bit tricky, though.

* **Tensorflow installation (Windows):**  

There's a couple of ways to install Tensorflow, as you can find here: [Tensorflow installation](https://www.tensorflow.org/install/install_windows). But there's a tiny problem: Tensorflow only works with Python 3.5.x on Windows, so as you might guess, other versions won't work. Well, you're not wrong.

So if you use a Windows machine, I recommend that you stick with Anaconda to manage Python versions as well as its dependencies (you can use the native **pip** along with Anaconda too). To install Anaconda, please visit their [Windows install](https://docs.continuum.io/anaconda/install-windows).

After Anaconda is completely installed, it will provide you a customized command prompt (called Anaconda Prompt), where you can run Python shell with ease, and it understands some Linux commands too!

![Anaconda Prompt](/images/projects/tensorflow-install/anaconda_prompt.PNG)

So for now, let's get started. Firstly, you need to create a Python 3.5 environment in order to install Tensorflow:

```
>conda create -n tensorflow_windows python=3.5 anaconda
```

After the new environment is created, let's activate it (notice that the environment's name will be added before the prompt):

```
>activate tensorflow_windows
(tensorflow_windows)>
```

Next, we will install Tensorflow, let's suppose we are working on a CPU-only Windows machine (my company's PCs are all CPU-only):

```
(tensorflow_windows)>pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl
```

It couldn't be simpler, right? Now, let's go ahead if Tensorflow is installed properly:

```
(tensorflow_windows)>python
>>> import tensorflow as tf
>>> sess = tf.Session()
>>> print(sess.run(tf.constant('Hello, Guys!')
```

The output should be as follow:

```
b'Hello, Guys!'
```

That's it. Now you have Tensorflow installed on your machine and you can start your first Tensorflow-based project. But I suggest that you keep reading. Yeah, I guess you don't want to miss out Keras, right?

* **Keras installation (Windows):**  

Well, it's a real headache if you want to install Keras into Windows machine. But I highly recommend that! Having a framework built on top of Tensorflow will help your work become much easier. 

The least tedious way to install Keras (as some of you might have done) requires just two commands like below:

```
(tensorflow_windows)>conda install mingw libpython
(tensorflow_windows)>pip install keras
```

But I hardly recommend it! As with Theano, installing Keras like above may result in trouble since the version to be installed is usually not up-to-date with the latest version of Tensorflow. So rather than the ones above, here's the recommended commands:

```
(tensorflow_windows)>conda install mingw libpython
(tensorflow_windows)>pip install --upgrade keras
```

**\-\-upgrade** flag will ensure that you install the latest version of Keras. But guess what, that's where the trouble comes from! The installation should be aborted, and you will see an error telling you that **scipy** failed to install.

The reason is, Keras requires SciPi, a library built on top of Numpy (and of course, for numerical computational optimization purposes), and sometimes, it has some problem with MKL, an optimization library from Intel. So what we're gonna do is, instead of installing Numpy and SciPy using native **pip**, we will download and install from customized wheel files. The download files' URLs are below:

(Remember to select the right one, which has **cp35** in its name!)  
 * Numpy: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy  
 * SciPy: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy

After downloading the two files, navigate to your download folder, and install them like this:

```
(tensorflow_windows)>pip install numpy‑1.11.3+mkl‑cp35‑cp35m‑win_amd64.whl
(tensorflow_windows)>pip install scipy‑0.19.0‑cp35‑cp35m‑win_amd64.whl
```

And for now, we can install Keras:

```
(tensorflow_windows)>pip install --upgrade keras
```

The installation should go smoothly without any error. Let's go ahead and check if it works properly:

```
(tensorflow_windows)>python
>>> from keras import backend
>>> print(backend._BACKEND)
```

The output would be something like this:

```
tensorflow
```

Then we are done!

### Summary

That's it for today. It may not be an informative post as you expected, but I hope you guys, especially Windows users can now install Tensorflow and Keras on Windows with no more annoying errors. So, get your hands dirty and begin your journey with Tensorflow! And see you next time.
