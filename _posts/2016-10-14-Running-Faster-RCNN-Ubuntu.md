---
title: "Compiling and Running Faster R-CNN on Ubuntu (CPU Mode)"
header:
  teaser: teaser.jpg
categories:
  - Project
tags:
  - machine-learning
  - faster r-cnn
  - caffe
  - compile
  - cpu mode
  - essential
---

So today I am gonna tell you about how to compile and run Faster R-CNN on Ubuntu in CPU Mode. But there is a big chance that many of you may ask: What the hell is Faster R-CNN?

In my previous posts, I have done a project Real-time Object Recognition (you can find it here: [Real-time Object Recognition](https://chunml.github.io/ChunML.github.io/project/Real-Time-Object-Recognition-part-one/){:target="_blank"}). The result received was pretty good, but as you might notice, that it got a problem (a big problem). The problem is that the trained Model could recognize one object per frame. So if I want it to recognize two or more objects (and even tell me where each of them locates), it will raise the white flag!

So it came to my next mission that I have to find a way to deal with Object Detection. Of course I knew some of them before, but what I wanted is something which applied Convolutional Neural Network. Among some great papers people had done out there, I chose Faster R-CNN.

In today's post, I have no intention to talk about how Faster R-CNN works. I just leave it for a future post, when I finish my task. Today I am just going to talk about how to compile and run Faster R-CNN on Ubuntu - in CPU Mode. And if you ever wonder why I am doing this post although I can find a great deal of tutorials on the net, the answer is: just like Caffe in CPU Mode, compiling Faster R-CNN was hard like hell too!

Faster R-CNN was originally implemented in MATLAB, but they also provided a Python reimplementation code (phew!). So let's grab it from GitHub:

{% highlight python %}
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
{% endhighlight %}

Just make sure that you didn't forget the *--recursive* flag. After the download completes, jump to the *lib* folder:

{% highlight python %}
cd py-faster-rcnn/lib
{% endhighlight %}

Here we are compiling Faster R-CNN for CPU Mode, so we have to make several changes. Let me guide you through this tough guy. First, let' open the **setup.py** file, then comment out all the lines below:

From this:

{% highlight python %}
CUDA = locate_cuda()

self.set_executable('compiler_so', CUDA['nvcc'])

Extension('nms.gpu_nms',
    ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
    library_dirs=[CUDA['lib64']],
    libraries=['cudart'],
    language='c++',
    runtime_library_dirs=[CUDA['lib64']],
    # this syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc and not with
    # gcc the implementation of this trick is in customize_compiler() below
    extra_compile_args={'gcc': ["-Wno-unused-function"],
                        'nvcc': ['-arch=sm_35',
                                 '--ptxas-options=-v',
                                 '-c',
                                 '--compiler-options',
                                 "'-fPIC'"]},
    include_dirs = [numpy_include, CUDA['include']]
),
{% endhighlight %}

To this:

{% highlight python %}
#CUDA = locate_cuda()

#self.set_executable('compiler_so', CUDA['nvcc'])

#Extension('nms.gpu_nms',
#    ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
#    library_dirs=[CUDA['lib64']],
#    libraries=['cudart'],
#    language='c++',
#    runtime_library_dirs=[CUDA['lib64']],
    # this syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc and not with
    # gcc the implementation of this trick is in customize_compiler() below
#    extra_compile_args={'gcc': ["-Wno-unused-function"],
#                        'nvcc': ['-arch=sm_35',
#                                 '--ptxas-options=-v',
#                                 '-c',
#                                 '--compiler-options',
#                                 "'-fPIC'"]},
#    include_dirs = [numpy_include, CUDA['include']]
#),
{% endhighlight %}

After saving the file, now we can compile the *lib* folder. Let's execute the command below:

{% highlight python %}
make
{% endhighlight %}

You should now go through without any error. If some error still occurs, make sure that you didn't miss any line above.
Next, we will compile **caffe** and **pycaffe**. You may remember that I made a post about how to install **caffe** on Ubuntu in CPU Mode, so why don't we just use the results we got, instead of doing the same thing over again? Well, that is because they had made some changes to the *caffe* original codes (implementing some necessary classes, methods, ...), we have to compile a new set in order to run their codes. But don't worry, there are things which we can re-use. I will tell you below. Now let's jump to the *caffe-fast-rcnn* folder:

{% highlight python %}
cd ../caffe-fast-rcnn
{% endhighlight %}
 
Similar to what we did with **caffe** before, this time we are likely to make some change to the *Makefile* files. But actually, they are just files which tell the compile how to compile things (like where to look for include files, library files, etc), so we can re-use our old *Makefile* which we have already modified. In case you haven't had a look at my old post yet, you can find it here: [Installing Caffe on Ubuntu](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/){:target="_blank"}.

So what we are gonna do is going to where we placed **caffe**, copy the *Makefile* and *Makefile.config* files, and paste them into the *caffe-fast-rcnn* folder.

But even after we did that, there is still one tiny change we have to make, let's open the *Makefile.config* file, uncomment the line below:

From this:

{% highlight python %}
# WITH_PYTHON_LAYER := 1
{% endhighlight %}

To this

{% highlight python %}
WITH_PYTHON_LAYER := 1
{% endhighlight %}

And we have done. Now go ahead and *make* it:

{% highlight python %}
make && make pycaffe
{% endhighlight %}

This time should run smoothly too. Congratulations, you have successfully compiled **caffe** for Faster R-CNN.

It seems like we have just made some tiny changes up to now. But don't be complacent too soon, although we have sucessfully compiled **caffe** for Faster R-CNN, we still cannot run their demo code now. There are some other places we have to modify too.

{% highlight python %}
cd ..
./data/scripts/fetch_faster_rcnn_models.sh
{% endhighlight %}

The command above will download the pre-trained model in order to run the demo code. After the download finishes, let's apply the changes below:

File: *lib/fast_rcnn/nms_wrapper.py*, from this:

{% highlight python %}
from nms.gpu_nms import gpu_nms
{% endhighlight %}

To this:

{% highlight python %}
# from nms.gpu_nms import gpu_nms
{% endhighlight %}


File: *lib/fast_rcnn/config.py*, from this:

{% highlight python %}
__C.USE_GPU_NMS = True
{% endhighlight %}

To this:

{% highlight python %}
__C.USE_GPU_NMS = False
{% endhighlight %}

That is it. We should now be able to run the demo project now. So let's do it:

{% highlight python %}
./tools/demo.py --cpu
{% endhighlight %}

Note that you must provide the *--cpu* flag to tell it to run in CPU Mode. It will take a while to run, because it will process all images before outputting the results (in my case it took approximately 22s per image). The output should look like this:

![3](/images/projects/running-faster-rcnn-ubuntu/3.png)

![1](/images/projects/running-faster-rcnn-ubuntu/1.png)

So, I have shown you how to compile and run the demo code of Faster R-CNN. It was not so hard especially if you experienced the *caffe* installation before. I will be glad if you find this post helpful. And even if you followed all the instructions above but you still couldn't make it through the frustrating errors, don't hesitate to leave me a comment below, I will help you as soon as possible, I promise!

And as I promised above, I will do some posts related to my current work using Faster R-CNN, and I will tell you more about Faster R-CNN too. So stay tuned! I'll be back.

*Reference*:  
Faster R-CNN GitHub page: https://github.com/rbgirshick/py-faster-rcnn



