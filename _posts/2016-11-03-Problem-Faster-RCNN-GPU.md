---
title: "Solving problem when running Faster R-CNN on GTX 1070"
header:
  teaser: projects/problem-faster-rcnn-gpu/gpu_ok.png
categories:
  - Project
tags:
  - machine-learning
  - faster r-cnn
  - gpu
  - gtx 1070
  - image not showing
  - caffe
  - compile
  - cpu mode
  - essential
---

Hello guys, it's great to be here with you today (why do I keep saying that boring greeting, you may ask). To be honest, there are a lot of things I want to share to you, especially since I built my own machine for Deep Learning. Of course, having my own machine is great, it allows me to try every crazy idea which has ever crossed through my mind, without giving a damn thought about the payment. However, good thing is followed by troubles, as it always be.

You might read my last post about my experience in installing Caffe on Ubuntu 16.04, with CUDA and cuDNN to make use of the great power of the GPU. Yeah, we went through so many steps to install so many necessary things. And fortunately, things worked flawlessly in the end. For ones who haven't read it yet, you can find it right below:

* [Installing Caffe on Ubuntu 16.04 ](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/){:target="_blank"}

And as you can guess, the next thing I did right after having Caffe installed on my machine, is grabbing the latest Python implementation of Faster R-CNN. I once talked about how to compile and run Faster R-CNN on Ubuntu in CPU Mode, you can refer to it here:

* [Compiling and Running Faster R-CNN on Ubuntu (CPU Mode)](https://chunml.github.io/ChunML.github.io/project/Running-Faster-RCNN-Ubuntu/){:target="_blank"}

Obviously, once you are able to run Faster R-CNN in CPU Mode, making it work with GPU may not sound like a big deal. Why? Because you had successfully installed Caffe, which means you had gone through all the most confusing steps to get CUDA and cuDNN libraries ready. But to tell the truth, I failed to, on the first try!

It was a shock to me, which took me a while to overcome. Then I soon realized that, it's all on me now because it was me who built my own machine. So I had no choice, but to figure it out myself. And after just more than one hour, the problem was solved. The problem always looks harder than it's supposed to be. I've been taught that simple thing so many times in my life, and I just kept forgetting about it. And that's also the reason why I'm writing this post, to share with you some experience to deal with this kind of troubles.

Let's start from the beginning. I was so excited to get everything ready right after installing Ubuntu, so I immediately jumped into installing Caffe without any consideration. And as you may guess, I grabbed all in the latest version, which means that at first, I installed cuDNN v5.1 to CUDA installation folder.

Things worked just right with Caffe, until it came to Faster R-CNN. To make it more clear, I downloaded the latest Python implementation of Faster R-CNN from their GitHub as before:

{% highlight python %}
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
{% endhighlight %}

After that, I compiled the libraries by jumping into *lib* folder and run:

{% highlight python %}
cd lib
make
{% endhighlight %}

Luckily, there's no need to change anything this time. Phew! Then I went backward, then jumped into the *caffe-fast-rcnn* folder, create *Makefile.config* from *Makefile.config.example* file, then applied some necessary modifications to it. That was exactly the same as what we did with Caffe's installation. You can refer to the link I showed above. And it's time to compile Caffe for Faster R-CNN:

{% highlight python %}
cd ../caffe-fast-rcnn
make -j8 && make pycaffe
{% endhighlight %}

And I received some unexpected result, like below:

![cudnn5_error](/images/projects/problem-faster-rcnn-gpu/cudnn5_error.png)

Obviously, as the errors were self-explained, there was something wrong with the cuDNN v5.1 library. Did I just say *v5.1*? That was all my bad, since I forgot that Faster R-CNN is still **incompatible** with cuDNN v5.1. This wouldn't happen if I considered carefully before installing cuDNN. But that was an easy fix, since it seemed like replacing with *cuDNN v4* helps fix the problem. So I gave that thought a try. The installation of cuDNN v4 is exactly same as cuDNN v5.1 so I omit it from here.

After re-installing cuDNN, I ran it again to see if it works:

{% highlight python %}
make clean # to delete the previous progress
make -j8 && make pycaffe
{% endhighlight %}

No errors shown this time. Thank God, seems like it works now, I thought. 

![cudnn4_ok](/images/projects/problem-faster-rcnn-gpu/cudnn4_ok.png)

So I moved on to prepare the pre-trained model, just like the instructions on their GitHub repository:

{% highlight python %}
cd ..
./data/scripts/fetch_faster_rcnn_models.sh
{% endhighlight %}

It took some minutes to complete, since the model is quite large in size. Now, I got everything ready and couldn't wait any longer to run the *demo*:

{% highlight python %}
python ./tools/demo.py
{% endhighlight %}

And here's the result I had, nothing seems to go wrong, I guessed:

![cudnn4_no_image](/images/projects/problem-faster-rcnn-gpu/cudnn4_no_image.png)

To tell the truth, I think I'm a very patient guy. So I just left it there for a while. "So, where are all the images?", I nearly talked to the screen. The reason why I couldn't stay calm is that, this time there were no errors shown, and no images came out to screen, either. It took me another while to admit that something was going wrong somewhere, and I had to figure it out. Since we ran the *demo.py* file, then looking at that file first may help find something.

Since I had read the paper of Faster R-CNN before, so it was somehow easy to understand what each part of the code is doing. To recap a little bit, as I shown you in the previous post, here's the result we want to see:

![result](/images/projects/running-faster-rcnn-ubuntu/1.png)

So what we want is an image (at least), in which each detected object was bounded by a rectangle, with some text to indicate which class it belongs to.
Therefore, it's not the complicated code used for detecting, but the two parts below that I want you to focus into:

![demo_py](/images/projects/problem-faster-rcnn-gpu/demo_py.png)

You can refer to the paper of Faster R-CNN to find some more details. To make it easy to understand, Faster R-CNN searched for some regions which likely contains an object, then each object was detected with probabilities to indicate how likely that object belongs to a particular class. Let's open the *demo.py* file, and make the following modification for a better understand:

{% highlight python %}
def demo(net, image_name):
     """Detect object classes in an image using pre-computed object proposals."""
 
     # Load the demo image
     im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
     im = cv2.imread(im_file)
 
     # Detect all object classes and regress object bounds
     timer = Timer()
     timer.tic()
     scores, boxes = im_detect(net, im)
     print(scores.shape) # ADD THIS LINE!!!
     timer.toc()

{% endhighlight %}

Add one line like above inside the *demo* method in *demo.py* file so that we can know the shape of the output scores, then run it again, here's what I received:

{% highlight python %}
Loaded network /home/chun/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/000456.jpg
(300, 21)
Detection took 0.069s for 300 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/000542.jpg
(259, 21)
Detection took 0.064s for 259 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/001150.jpg
(223, 21)
Detection took 0.054s for 223 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/001763.jpg
(201, 21)
Detection took 0.052s for 201 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/004545.jpg
(172, 21)
Detection took 0.051s for 172 object proposals
{% endhighlight %}

The model used for demo file was fine-tuned so that it can classify 20 classes, plus one *background* class, so we have 21 classes in total. Let's look at the result of the first image. The algorithm proposed 300 object regions (or we can say 300 rectangles can be drawn at max), and the corresponding scores had the shape of (300, 21). It means that each proposal, the algorithm computed all 21 probabilities for all classes. And we will base on those probabilities to decide which class it belongs to.

Now let's look at the first part which I highlighted above. What it's doing is getting the first probability which is greater than the threshold value (it was set to 0.5 this time). So, let's print out the probabilities to see why it couldn't get through the *if* condition which followed:

{% highlight python %}
def vis_detections(im, class_name, dets, thresh=0.5):
     """Draw detected bounding boxes."""
     inds = np.where(dets[:, -1] >= thresh)[0]
     print(dets[:, -1]) # ADD THIS LINE!!!
     if len(inds) == 0:
         return
{% endhighlight %}

Let's add the line above, right before the *if* condition, then run it again to see what happens:

And here's the result:

![low_proba](/images/projects/problem-faster-rcnn-gpu/low_proba.png)

And at this time, I thought I knew where the problem came from. None of the probabilities was greater than 0.5, so obviously I ended up with no images being shown. Then, an idea came through my mind. "Why don't I try to run on CPU?", I thought.

{% highlight python %}
python ./tools/demo.py --cpu
{% endhighlight %}

The result was like below. I was totally speechless.

![cpu_ok](/images/projects/problem-faster-rcnn-gpu/cpu_ok.png)

So I got to go through such a long way, to find out the place which may cause the problem. And thanks to the CPU, or I have to say: I had to make use of the CPU to be sure whether I got it right. Anyway, at least I knew that the compilation was successfully got through. And, the problem is likely from the GPU. Wait, what? The most expensive part of the machine caused the problem? That was ridiculous, I thought. But I couldn't admit that, so I ran into the net right after. The bad news is: yeah, it was caused by the GPU. And the good news? The good news is: the problem happened because the new GPUs don't support cuDNN older than v5. Well, you gotta be kidding me. Caffe doesn't work with cuDNN newer than cuDNN v4, and my GTX 1070 doesn't support cuDNN older than v5. So does it mean that I won't be able to run Faster R-CNN on my machine?

Thinking it that way did depress me a lot. But fortunately, it turns out that re-compiling Faster R-CNN without cuDNN help solve it. So I moved on and gave it my last shot. I opened the *Makefile.config*, comment out the *USE_CUDNN* option, then compile it again. After the compilation completes, I ran it again, hoped that it works for me this time:

![gpu_ok](/images/projects/problem-faster-rcnn-gpu/gpu_ok.png)

It finally worked! That was like you woke up after a nightmare. But it was fantastic! I mean, the feeling when you could finally make things done is hard to express, right?

### Summary

In today's post, I have shown you a problem when running Faster R-CNN with the GTX-1070, which caused the images not being shown when it's done. Going through every step above, it seemed like the problem was not that hard, and that I didn't have to make such a long post. As I mentioned earlier, sometimes you will have to deal with the problems yourself, and sometimes, searching Google for the answer in the very beginning is not a good practice at all. As an developer, especially a Machine Learning developer, it's likely that you are working with not only codes, but also many hard-to-understand algorithms, at least trying to figure things out yourself in the beginning will help you understand the problem more deeper so that the community can get involved efficiently, it also help you gain some very precious experience. And through times, you will be less likely to give up when struggling problems.

That's all I want to tell you today. I'll be more than happy if you find this post helpful. Now, goodbye and see you in the next post!
