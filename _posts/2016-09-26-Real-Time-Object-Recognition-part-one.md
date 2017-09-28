---
title: "Real Time Object Recognition (Part 1)"
categories:
  - Project
tags:
  - machine-learning
  - VGG
  - object recognition
  - learning
  - project
---

Technology sometimes seems like magic, especially when we don't have any idea about how it was done, or we even think it can't be done at all.
When I was a kid, I was a huge fan of Sci-Fi Films, which were on every TV channel in the 1990s in my country. They did inspire me a lot, help grow up an "engineering mindset" in me.
Being an engineer sometimes makes me feel so great, because I can mostly understand many things that seem to be magical to others. But to be honest, there are still lots of things out there can make me feel like a fool. Just let me tell you one of them.

It was sometime in 2015, I watched a demo video from a startup company in Japan. It was about Object Recognition using Deep Learning. At that time, I didn't even have any idea about what Machine Learning is, not to mention Deep Learning! 
My thought then? "That was impossible. How did they do that?"
I kept questioning myself for a while. I just couldn't get it out of my mind. But unfortunately, because my first thought was "That was impossible", or at least "That was impossible to me", I gave it up, left the mystery unrevealed.

It was not until I paid serious attention to Machine Learning that I somehow made the mystery become clear. As I was working around with Convolutional Neural Network, then I suddenly thought of the mystery I gave up one year ago. I mean, that was really exciting and crazy. (Feels like Mission 6 accomplished to me, lol)

Seems like a pretty long beginning (I didn't mean to write a preface for my book or something, though). So I am here to share my work. In fact, it was not something impresssive (especially when comparing to outstanding projects out there). But I really hope this post can help, especially for ones who have enthusiasm in Machine Learning and once came accross something like me before.

In this project, I will try to reproduce the result I saw in the demo video one year ago. Concretely, I will move the camera around a table, with many objects put on it. The computer will try to recognize each object, and print the object's label in the output of the camera.

Writing it all in one post may hurt, so I separate this project into two parts like below:

* Part 1: Go through the folder containing the images, recognize object in each.
* Part 2: Real time object recognition through input from camera

Before I begin, let's talk a little bit about how Machine Learning works. I had post about what Machine Learning exactly is, and how it exactly works here: [What is Machine Learning?](https://chunml.github.io/tutorial/Machine-Learning-Definition/){:target="_blank"}

The main part of any Machine Learning system, is the Model. It contains the algorithms and all the associated parameters. A good Model will produce a good prediction. Because of that, the training process (i.e. the process which produces the Model) can be heavily computational and take a lot of resources (you know, RAM and memories to store the parameters).

Fortunately, a great work from [François Chollet](https://github.com/fchollet){:target="_blank"} helps make this problem become much more relaxing than ever. I mean that you won't have to spend a great deal of money on a giant computer (with powerful CPU and GPUs), you won't have to spend time on training the data yourself, thanks to François Chollet for providing us the pretrained Model on the most outstanding CNN architectures. Sounds cool, right? Among those, I will use the VGG16 Model for this project.

So, let's get down to business.

Firstly, I downloaded some images for testing purpose, and put them in the same folder.

{% highlight python %}
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True)
args = vars(ap.parse_args())

files = [os.path.join(args["folder"], f) for f in os.listdir(args["folder"])]
random.shuffle(files)
{% endhighlight %}

As written in the code above, I passed the folder name, then got the list of file names stored in *files* variable. I shuffled the list for demonstration purpose (you don't want the same order everytime, right?)

Next, I initiate the VGG16 Model, provided by François Chollet. You will have to clone his repository first, from [here](https://github.com/fchollet/deep-learning-models){:target="_blank"}

{% highlight python %}
from vgg16 import VGG16

model = VGG16(weights="imagenet")
{% endhighlight %}

As you can see we are passing the *weights="imagenet"* parameter, to tell the Model to initialize with pretrained parameter set. It is quite a large file (approximately 550MB), so it will take some time to download on its first run. (The parameter set is saved somewhere in keras' temporary folder)

Then, I looped through the file list created above, in each iteration:

{% highlight python %}
from keras.preprocessing import image as image_utils

image = image_utils.load_img(file, target_size=(224, 224))
image = image_utils.img_to_array(image)
{% endhighlight %}

Keras provides us a method for loading image for training and testing purpose. Note that OpenCV and Keras treat the input image in different ways, so we cannot use image loaded by OpenCV's imread method for Keras. Concretely, let's say we have a 3-channel image (a common color image). OpenCV's imread will produce an (width, height, 3) array, whereas Keras requires an (3, width, height) format.
Another way to solve this is to use Numpy's tranpose method:

{% highlight python %}
import numpy as np

image = cv2.imread(file)
image = image.transpose((2, 0, 1))
{% endhighlight %}

I prefer the second approach, you will understand why when we come to the second part of this project.

Next, we need to add one more dimension to the array obtained above. Why we have to do that? If you have experience with Neural Network, you may find the term *mini_batch* similar. The additional dimension will tell the Model the number of input arrays (for example, you have 70,000 data, so you need to pass an array with shape (70000, depth, width, height) for the Model to run on, let's say SGD or RMSprop or something).

If you don't have any idea about what I've just talked, you can ignore it for now. I'll talk about Neural Network in later posts, I promise.

After converting the input image to Keras's format, I passed it through a pre-processing method:

{% highlight python %}
from imagenet_utils import preprocess_input

image = preprocess_input(image)
{% endhighlight %}

The pre-processing method came along with the models provided by François Chollet. It simply subtracts each channel with its mean value, and solve the ordering conflict between Theano and Tensorflow.

So we now have things ready. Let's pass the preprocessed array to the Model and get it predicted:

{% highlight python %}
preds = model.predict(image)
(inID, label) = decode_predictions(preds)[0]
{% endhighlight %}

Let's talk a little bit about ImageNet's image database (the database on which VGG16 was trained). It was organized according to WordNet hierarchy (you can find more details [here](http://wordnet.princeton.edu/){:target="_blank"}). But the predicting result is always numerical (Neural Network only works with numerical labels), so we have to map between the numerical result, and the noun provided by WordNet.

Once again, François Chollet provided a method to do that: decode_predictions method. It simply map the predicted result with a JSON file, and return the associated noun instead. Here's what the JSON file looks like:

{% highlight json %}
{"0": ["n01440764", "tench"], 
 "1": ["n01443537", "goldfish"], 
 "2": ["n01484850", "great_white_shark"], 
 "3": ["n01491361", "tiger_shark"], 
 "4": ["n01494475", "hammerhead"], 
 "5": ["n01496331", "electric_ray"], 
 "6": ["n01498041", "stingray"], 
 "7": ["n01514668", "cock"], 
 "8": ["n01514859", "hen"], 
 "9": ["n01518878", "ostrich"], 
 "10": ["n01530575", "brambling"], ...}
{% endhighlight %}

Finally, the easiest part: load the input image with OpenCV's imread, then put the label found above on it, then show the result. And we are DONE!

{% highlight python %}
origin = cv2.imread(file)
cv2.putText(origin, "Predict: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Result", origin)
{% endhighlight %}

Here are some results I got from my test images:

![Image_1](/images/projects/real-time-object-recognition/1.jpg)

![Image_2](/images/projects/real-time-object-recognition/2.jpg)

So I've just walked through the first part of the Real time Object Recognition project. You may think that it's not a big deal anyway. But in my opinion, it did help a lot in visualizing some real work on Object Recognition using Convolutional Neural Network. Just implementing the code to use the model may not take you some months to work on, but importantly, it now doesn't seem like magic anymore.

That's all for part 1. Hope you enjoy it. In part 2, I will continue with real time object recognition through continuous input from camera. So stay updated and I'll see you there.
