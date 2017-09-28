---
title: "Real Time Object Recognition (Part 2)"
categories:
  - Project
tags:
  - machine-learning
  - VGG
  - object recognition
  - learning
  - project
  - camera
  - real time recognition
---

So here we are again, in the second part of my Real time Object Recognition project. In the previous post, I showed you how to implement pre-trained VGG16 model, and have it recognize my testing images.

You can take a look at the first part here: [Real Time Object Recognition (Part 1)](https://chunml.github.io/ChunML.github.io/project/Real-Time-Object-Recognition-part-one/){:target="_blank"}.
The model performed pretty well, although I didn't even have to do any further pre-processing (such as object localization, or image enhancement, etc).

In the next step, I will use the same model, to predict object from a continuous input, such as video file or input from a camera. If you have ever done some work with Computer Vision before, you will see find it extremely easy. Let's say, just a slight improvement over the last one. Yeah, that's true!

For ones who have no experience in Computer Vision, I'll explain a little bit here. So, if you have an image, and want to have it recognized. Here is what you do:

{% highlight python %}
# Load the image

# Use the model to have the image recognized

# Show the result
{% endhighlight %}

Now, you have a video file, how to make it work? You know that movies, videos are just hundreds or thousands of images shown continuously. So what you have to do, is just read through a video, from its first image, to the last one. And you see that it works exactly the same way with the one above.

{% highlight python %}
# Load the video
for image in video:
  # Use the model to have the image recognized
  # Show the result
{% endhighlight %}

Easy, right? As shown above, the result is shown continuously to your eyes. And your eyes will treat them as a video, not seperate  images.

Here comes something to consider about. Remember the conflict between OpenCV and Keras that I mentioned in the last post? To recall a little bit, After reading the input image, OpenCV and Keras turn it into arrays in different ways. So basically, if we want the model to recognize the input image, we have two choices to pick.

Use the Keras' method:
{% highlight python %}
from keras.preprocessing import image as image_utils

image = image_utils.load_img(file, target_size=(224, 224))
image = image_utils.img_to_array(image)
{% endhighlight %}


Or, use the OpenCV's method, then converting to Keras' format:
{% highlight python %}
import numpy as np

image = cv2.imread(file)
image = image.transpose((2, 0, 1))
{% endhighlight %}

As you can see, Keras' method reads the image from file, then performs conversion. In case of reading from a camera, it may be a little bit odd to read each frame, then immediately save to disk, then have it read by Keras. Doing like so will slow down the performance, and it is obviously not an efficient solution.

Remember I said that I prefered the approach which using OpenCV's method? Because I just needed to perform one matrix transpose, and I had everything ready for Keras, without saving and reading from disk anymore.

So, by just adding two lines of code made things done. Why the hell must I split it into two parts? Well, because there's a tiny problem in performance. Obviously, everyone does prefer a perfect show, right?

What's the problem then? I told you in the first part, that our model is quite a big guy (it can distinguish between 1000 classes). Big guys tend to move slowly, so does the VGG16 model. It took approximately 1.5 ~ 2 seconds to recognize ONE image (on my PC). So what will it be like when dealing with an input from a camera (or a video file)? Definitely, it will be a very horrible scene that you never wish to see.

Therefore, we come to a fact that we must do something here, or we are likely to ruin the show. There may be many tricks out there, I think. But in my case, I just simply, put the recognition task in another thread. Whenever the recognition on a frame is ready to deliver, I updated the result. On the other side, the output was smoothly shown since it no longer had to wait for the recognition result.

Ones with little experience in programming, especially multi-threading, would find what I said hard to understand. Don't worry, I'll explain right below:

In case of no multi-threading:
{% highlight python %}
# Load the video
for image in video:
  # Use the model to have the image recognized
  
  # 2 SECONDS LATER...
  
  # Show the result
{% endhighlight %}


With multi-threading:
{% highlight python %}
# Load the video
for image in video:
  # Show the image with the last result put on it

# Somewhere else on Earth
  # Load the image to recognize
  
  # Perform recognition
  
  # 2 SECONDS LATER...
  
  # Output the result
{% endhighlight %}

Do you get the trick here? When implementing multi-threading, the codes which deal with recognition task is seperated from the output task. It will return the result each 2 seconds. On the other side, because the output won't have to wait for the recognition result, it just simply puts the last received result on, and updates when the new one is delivered.

You make ask, so the recognition process is actually skipping everything between the 2-second periods, so what we see in the output may not be the exact result. For example, your camera was on a dog, and you passed the frame containing the dog to the recognition code, 2 seconds later, the Dog label was delivered, but you are now looking at a lion! What a shame on a real time recognition app! Well, it sounds like an interesting theory. But I think, no one moves their cameras that fast, let's say, abruptly change the view each second. Am I right?

So, you now know about the problem you may face working with real time object recognition, and I showed you how to deal with it by implementing multi-threading. But...
If there's something I need you to know about me, it's that I am not a professional programmer, which means I prefer working on theoretical algorithms to spending hours on coding. To me, coding is simply a way to visualize the theory. Therefore, my code may seem like a mess to you guys. So feel free to tell me if you find something wrong or something which can be improved. I will definitely appreciate that.

Tired of reading? So sorry to make it long. I'll show you the result I got right below:

{::nomarkdown}
<iframe width="560" height="315" src="https://www.youtube.com/embed/70Kv8Rr72ag" frameborder="0" allowfullscreen></iframe>
{:/nomarkdown}


Let's talk a little bit about the result above. You can see that, despite of the bad light condition (my room is only equipped with one damn yellow light!), the model still performed pretty well and totally satisfied me. Sometimes it got wrong (couldn't recognize my G-Shock, or having trouble in distinguishing whether it was a screen or a TV!), but that was far more than expected.

And finally, here's the code in case you need: [Object Recognition Code](https://github.com/ChunML/DeepLearning){:target="_blank"}. Most of them was cloned from François Chollet's repository. I just coded two files below:

* For recognizing images seperately:
test_imagenet.py 

* For real time object recognition with camera:
camera_test.py

Hope you enjoy this project. Feel free to leave me some feedbacks or questions. I will be more than pleased to help.

So we are done with this Real time Object Recognition project, but not with Machine Learning! And I'll see you soon, in the next projects!
