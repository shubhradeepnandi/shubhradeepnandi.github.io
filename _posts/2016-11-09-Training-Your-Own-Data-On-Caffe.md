---
title: "Training With Your Own Dataset on Caffe"
header:
  teaser: teaser.jpg
categories:
  - Project
tags:
  - machine-learning
  - deep-learning
  - caffe
  - installation
  - gpu
  - training
  - fine-tuning
  - own data
  - essential
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hi, everyone! Welcome back to my Machine Learning page today. I have been playing around with Caffe for a while, and as you already knew, I made a couple of posts on my experience in installing Caffe and making use of its state-of-the-art pre-trained Models for your own Machine Learning projects. Yeah, it's really great that Caffe came bundled with many cool stuff inside which leaves developers like us nothing to mess with the Networks. But of course, there comes sometime that you want to set up your own Network, using your own dataset for training and evaluating. And it turns out that using all the things which Caffe provides us doesn't help Caffe look less like a *blackbox*, and it's pretty hard to figure things out from the beginning. And that's why I decided to make this post, to give you a helping hand to literally make use of Caffe.

Before getting into the details, for ones that missed my old posts on Caffe, you can check it out anytime, through the links below:

* [Installing Caffe on Ubuntu (CPU_ONLY)](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/){:target="_blank"}

* [Installing Caffe on Ubuntu (GPU)](https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/){:target="_blank"}

Now, let's get down to business. In today's post, I will mainly tell you about two points below:

* Downloading your own dataset

* Preparing your data before training

* Training with your prepared data

So, I will go straight to each part right below.

**1. Downloading your data**  
I think there's a lot of ways which everyone of you managed to get your own dataset. If your dataset has been already placed on your hard disk, then you can skip the **Downloading** section and jump right into the **Preparing** section. Here I'm assuming that you do not have any dataset of your own, and you're intending to use some dataset from free sources like ImageNet or Flickr or Kaggle. Then it's likely that: you can directly download the dataset (from sources like Kaggle), or you will be provided a text file which contains URLs of all the images (from sources like Flickr or ImageNet). The latter seems to be harder, but don't worry, it won't be that hard.

* Directly downloading from source:

This kind of download is quite easy. Here I will use the **Dogs vs. Cats** dataset from Kaggle for example. You can access the dataset from the Download page: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats). All you have to do is just register an account, then you can download the whole dataset. There are two of them, one for training purpose, which was named *train*, and one for evaluating, which was named *test1* respectively. I suggest that you should download the training set only. I will explain why when we come to the **Preparing** section. The file size is quite large so it should take a while to finish. And that's it. You have the dataset stored on your hard disk!

* Downloading from URLs

As you could see above, it's great if every dataset was zipped and provided directly to developers. But in fact, due to the copyright of the images (as well as other data types), providing data that way isn't simple, especially when we talk about an extremely large dataset like ImageNet. So data providers have another way, which is providing you the URLs only, and you will have to access to the image hosts yourself to download the data. I will use a very famous site for example, which is ImageNet, the site which holds the annual ILSVRC. You can read more about ILSVRC [here](http://www.image-net.org/challenges/LSVRC/){:target="_blank"}.

First, let's go to the ImageNet's URLs download page: [Download Image URLs](http://image-net.org/download-imageurls){:target="_blank"}. All you need to know to get the URLs is something called **WordNet ID** (or **wnid**). You can read more about ImageNet's dataset and WordNet to grab some more details because this post will be too long if I explain it here. To make it simple right now, ImageNet uses WordNet's synset, such as *n02084071*, *n02121620* which represents *dogs* and *cats* respectively, to name its classes. To find out what the synset of a particular noun, just access [Noun to Synset](http://www.image-net.org/synset?wnid){:target="_blank"}, then search for any noun you want, then you will see the corresponding synset. 

Once you knew the synset, you can download the URLs by going to this page:  
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=[wnid], which *[wnid]* is the synset of the object you want to download data for. For example, let's use two synsets above, to download the URLs of the Dogs and Cats images of ImageNet:

Dogs: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071

Cats: http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02121620

If you access the links above, you will see something like this:

![urls](/images/projects/training-your-own-data-on-caffe/urls.png)

So, the next thing to do is just simple, you have to copy those URLs and paste somewhere, let's say a text file or something. With ones who are familiar with Linux commands, you can see that we can use *wget* to grab all the images with ease. But there's some problem here: using *wget* is hard to rename the images as *wget* will use the name right in each URL to name each image. Training your own data with CNN on Caffe may not require any naming rules, but if you have intention to use your own data in other places, for example, the state-of-the-art Faster R-CNN, then the naming convention does matter! And as far as I know, we can manually rename all the images while downloading using *wget*, but it requires some experience in Linux commands, and to be honest, I tried and failed. But don't worry, I found the solution for that!

You know that Caffe provides us so many useful tools, to help us do all the heavy things so that we can use all the pre-trained Models without worrying about the data preparation, which means that, if you want to play with MNIST, Caffe provides you the script to download MNIST, if you want to play with CIFAR-10, Caffe got a script to download CIFAR-10 too. So, we can make use of the tools Caffe provides, and modify a little to make it work with our data. Not so bad, right?

All you have to do, is to make use of the tool which Caffe uses to download Flickr's images for fine-tuning (I will tell you about fine-tuning in the second part, so don't care about that term). Open your terminal, and type the commands below (make sure that you are in the root folder of Caffe):

{% highlight Bash shell scripts %}
cd data
mkdir DogsCats

cd ../examples
mkdir DogsCats
sudo cp finetune_flickr_style/* DogsCats/*
{% endhighlight %}

What we just did, is to create the neccessary folders for storing the script (*./examples/DogsCats*) and the images (*./data/DogsCats), then we copied the script to download Flickr's images to our new folder. Obviously, we have to make some changes in order to make it work properly, just some minor changes.

First, let's go to *./examples/DogsCats* folder, unzip the *flickr_style.csv.gz* to get a CSV file named *flickr_style.csv*. Open it up, take a look at the file. There are five columns but just three of them are actually used: *image_url*, *label* and *_split*. The *image_url* column stores all the URLs to all the images, the *label* column stores the label values, and the *_split* column tells whether each image is used for training or evaluating purpose.

As I mentioned earlier, we are not only downloading the images, but also renaming it, so we will use an additional column to store the name associating with each image URL, which I chose column *A* for that task. Before making any changes, let's deleting all the records, except the first row. Then, let's name cell A1 *image_name*. Next, in the *image_url* column, paste all the URLs of each class. Note that we won't paste all the URLs of all classes at once, since we have to labeling them. After pasting all the URLs of one class, let's say the Dogs class with *n02084071* synset, we will fill the *image_name* column. Start from cell A2, let's fill that it will *n02084071_0* then drag until you see the last URL in *image_url* column. Don't forget to add the *.jpg* extension when you finish (just use the CONCATENATE function).

Next, we will label the images we have just added URLs for. In the *label* column, let's fill with *0* until the last row containing URL. Since all URLs we pasted belong to Dogs, so they will have the same label. And lastly, let's fill in the *_split* column. In case of the Dogs' images, we have 1603 images in total, so let's fill *train* for the first 1200 images and *test* for the rest (here the train:test ratio I chose is 0.75:0.25). After all, your CSV should look similar to this:

![csv](/images/projects/training-your-own-data-on-caffe/csv.png)

And we can continue with other classes' images, don't forget to increase the value of *label* column each time you add another class's URLs.

So, we have done with the CSV file, let's go ahead and modify the Python script (make sure that you are in the root folder of Caffe):

{% highlight Bash shell scripts %}
cd examples/DogsCats
sudo vim assemble_data.py
{% endhighlight %}

First, let's replace all the phrase *data/finetune_flickr_style* with *data/DogsCats. That value tells where to store the downloaded images, so we have to point to our new created folder. Next, make some changes like below:

{% highlight vim %}
# Line 63
df = pd.read_csv(csv_filename, index_col=None, compression='gzip')

# Line 77
os.path.join(images_dirname, value) for value in df['image_name']
{% endhighlight %}

That's it. And now we are ready to download the images, and have them renamed the way we wanted:

{% highlight Bash shell scripts %}
python assemble_data.py
{% endhighlight %}

It will take a while for the script to run. Note that many of the URLs are inaccessible at the time of writing, since many of them were added quite so long ago. So if you notice that the number of downloaded images is not equal to the number of URLs, don't be confused.

As soon as the script finished running, then your images are all stored on your drive. So now your dataset is ready for the next stage!

**2. Preparing the data before training**  
So we just managed to have the desired dataset stored on your hard disk. And believe me or not, we have just completed the most time-consuming task! Before we can train our Network using the data we have just downloaded, there's some things we need to do. First, we need to convert the downloaded images into the format that the Networks can read. In fact, the Networks in Caffe accepts not just one kind of input data. As far as I know, there are three different ways to prepare our images so that the Networks can read them, and I'm gonna tell you about two of them: normal format and LMDB format. And second, we need to provide one special image called *the mean image*. Okay, let's get into each of them.

* Creating the train.txt and test.txt files

Let's first talk about the data conversion. As I said above, we have two choices. You can choose whether to use the normal format (leave the images untouched after downloaded), or to convert them to LMDB format. In both cases, you have to create two files called *train.txt* and *text.txt*. What the two files do is to tell our Network where to look for each image and its corresponding class. To understand better, let's go and create them.

I'm gonna use the *Dogs vs Cats* dataset which we downloaded from Kaggle (because we haven't touched it yet, have we?). Let's create two similar folders just like we did above with ImageNet's images, one for storing the images, and one for storing the necessary scripts:

{% highlight Bash shell scripts %}
cd examples
mkdir DogsCatsKaggle
cd ../data
mkdir DogsCatsKaggle
{% endhighlight %}

Then, let's place the zip file which we downloaded from Kaggle into *./data/DogsCatsKaggle* folder and upzip it. After unzipped, all of the images will be stored into the subfolder called *train*. Next, we're gonna create the *train.txt* and *test.txt* files. Let's go into the *./examples/DogsCatsKaggle* folder and create a Python file, name it *create_kaggle_txt.py* and fill the codes below:

{% highlight vim %}
import numpy as np
import os
 
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/DogsCatsKaggle/train'))
TXT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/DogsCatsKaggle'))
 
dog_images = [image for image in os.listdir(DATA_DIR) if 'dog' in image]
cat_images = [image for image in os.listdir(DATA_DIR) if 'cat' in image]
 
dog_train = dog_images[:int(len(dog_images)*0.7)]
dog_test = dog_images[int(len(dog_images)*0.7):]
 
cat_train = cat_images[:int(len(cat_images)*0.7)]
cat_test = cat_images[int(len(cat_images)*0.7):]
 
with open('{}/train.txt'.format(TXT_DIR), 'w') as f:
    for image in dog_train:
        f.write('{} 0\n'.format(image))
    for image in cat_train:
        f.write('{} 1\n'.format(image))
    f.close()
 
with open('{}/text.txt'.format(TXT_DIR), 'w') as f:
    for image in dog_test:
        f.write('{} 0\n'.format(image))
    for image in cat_test:
        f.write('{} 1\n'.format(image))
    f.close()
{% endhighlight %}

Then, all you have to do is to execute the Python script you have just created above:

{% highlight Bash shell scripts %}
python examples/DogsCatsKaggle/create_kaggle_txt.py
{% endhighlight %}

Now let's jump into *./data/DogsCatsKaggle*, you will see *train.txt* and *test.txt* has been created. And that's just it. We have finished creating the two mapping files for *Dogs vs Cats* dataset from Kaggle!

So, what about the Dogs and Cats images from ImageNet? Well, you may want to take a look at *./data/DogsCats*. Voila! When were the two files created? - You may ask. They were created when you ran the script to download the images! So with ImageNet's dataset, you don't have to create the mapping files yourself. That was great, right? Now we got the images, the mapping text files ready, there's only one step left to deal with the data: create the *mean image*.

* The need of computing the mean image

Why do we need the mean image anyway? First, that's just one type of *Data Normalization*, a technique to process our data before training. As I told you in previous post, the final goal of the learning process is finding the global minimum of the cost function. There's many factors that affect the learning process, one of which is how well our data was pre-processed. The better it is pre-processed, the more likely our Model will learn faster and better.

The goal of computing the mean image is to make our data have zero mean. What does that mean? For example, we have a set of training data like this: \\(x^{(1)}, x^{(2)}, \dots, x^{(m)}\\). Let's call \\(x_\mu\\) the mean value, which means:

$$
x_\mu=\frac{x^{(1)}+x^{(2)}+\dots+x^{(m)}}{m}=\frac{1}{m}\sum_{i=1}^mx^{(i)}
$$

Next, a new set of data will be created, where each \\(x_{new}^{(i)}=x^{(i)}-x_\mu\\). It's easy to see that the mean value of the new dataset is zero:

$$
\sum_{i=1}^mx_{new}^{(i)}=\sum_{i=1}^mx^{(i)}-mx_\mu=\sum_{i=1}^mx^{(i)}-m\frac{1}{m}\sum_{i=1}^mx^{(i)}=0
$$

So, above I just showed you a short explanation about one type for Data Normalization, which subtracting by the mean value to get a new dataset with zero mean. I will talk more about Data Normalization in future post. Now, how do we compute the mean image? As you may guess, of course Caffe provides some script to deal with some particular dataset. And we're gonna make use of it with some modifications! But before we can compute the mean image, we must convert our images into *LMDB* format first.

* Converting data into LMDB format

But first, why LMDB? Why is LMDB converting considered recommended, especially when we are working with large image database? To make it short, because it helps improving the performance of our Network. At present, performance is not all about accuracy anymore, but required to be both fast and accurate. With a same Network and a same dataset, how the data was prepared will decide how fast our Network learns. And LMDB conversion is one way (among many) which helps accomplish that. And the trade-off? The converted LMDB file will double the size of your downloaded images, since your images were decompressed before being converted (that's one reason why our Network performs faster with LMDB file, right?)

Next, let's copy the necessary script that we will make use of. I will use *Dogs vs Cats* dataset from Kaggle for example.

{% highlight Bash shell scripts %}
sudo cp examples/imagenet/create_imagenet.sh examples/DogsCatsKaggle/
{% endhighlight %}

To convert the downloaded *Dogs vs Cats* dataset to LMDB format using the script above, we will have to make some changes. But it's not a big deal at all because all we have to change is just the correct path to our images and the mapping text files. Below is the lines which I have applied changes for your reference:

{% highlight vim %}
EXAMPLE=examples/DogsCatsKaggle
DATA=data/DogsCatsKaggle
TOOLS=build/tools
 
TRAIN_DATA_ROOT=data/DogsCatsKaggle/train/
VAL_DATA_ROOT=data/DogsCatsKaggle/train/
...
RESIZE=true
...
GLOG_logtostderr=1 $TOOLS/convert_imageset \
...
    $EXAMPLE/dogscatskaggle_train_lmdb

echo "Creating val lmdb..."
 
GLOG_logtostderr=1 $TOOLS/convert_imageset \

...
    $DATA/text.txt \
    $EXAMPLE/dogscatskaggle_val_lmdb
 
echo "Done."
{% endhighlight %}

Next, let's go ahead and run the script above:

{% highlight Bash shell scripts %}
./examples/DogsCatsKaggle/create_imagenet.sh 
{% endhighlight %}

It will take a while for the conversion to complete. After the process completes, take a look at *./examples/DogsCatsKaggle* folder, you will see two new folders which are named *dogscatskaggle_train_lmdb* and *dogscatskaggle_val_lmdb*, and new LMDB files were placed inside each folder, created from the training data and test data respectively.

* Making the mean image

After creating LMDB files, making the mean image is no other than one last simple task to complete. All we have to do is to copy and apply some tiny changes into the script which computes the mean image.

{% highlight Bash shell scripts %}
sudo cp examples/imagenet/make_imagenet_mean.sh examples/DogsCatsKaggle/
{% endhighlight %}

And here's what it looks after modified:

{% highlight vim %}
EXAMPLE=examples/DogsCatsKaggle
DATA=data/DogsCatsKaggle
TOOLS=build/tools
 
$TOOLS/compute_image_mean $EXAMPLE/dogscatskaggle_train_lmdb \
  $DATA/dogscatskaggle_mean.binaryproto
{% endhighlight %}

And, only one last command to execute:

{% highlight Bash shell scripts %}
./examples/DogsCatsKaggle/make_imagenet_mean.sh 
{% endhighlight %}

And that's it. Let's go into *./data/DogsCatsKaggle* folder, you will see one new file called *dogscatskaggle_mean.binaryproto*, which means that the mean image was created successfully!

**3. Training with your prepared data**  
So now you nearly got everything ready to train the Network with the data prepared by yourself. The last thing is, of course, the Network! At this time, you may want to create a Network of your own, and train it using the data above (of your own, too!). But I recommend you try some available Networks which is provided by Caffe, some of which are very famous such as VGG16 or AlexNet. Let's pick AlexNet for now since it's quite simpler than VGG16, which will make it train faster. We need to create one new folder and copy the necessary files for Network definition. And for your information, Caffe uses the *protobuf* format to define the Networks, which you can read for details here: [Protocol Buffers](https://developers.google.com/protocol-buffers/){:target="_blank"}.

{% highlight Bash shell scripts %}
cd models
mkdif dogscatskaggle_alexnet
sudo cp bvlc_alexnet/solver.prototxt dogscatskaggle_alexnet/
sudo cp bvlc_alexnet/train_val.prototxt dogscatskaggle_alexnet/
{% endhighlight %}

Let's first modify the *solver.prototxt* first. This file stores the necessary information which the Network needs to know before training, such as the path to the Network definition file, the learning rate, momentum, weight decay iterations, etc. But all you need to do is just to change the file paths:

{% highlight vim %}
net: "models/dogscatskaggle_alexnet/train_val.prototxt"
...
snapshot_prefix: "models/dogscatskaggle_alexnet/caffe_alexnet_train"
{% endhighlight %}

Next, we will make change to the Network definition file, which is the *train_val.prototxt* file. In fact, it was nearly set up and we only need to modify a little bit. First, we have to tell it where to look for your prepared data. And second, we must change the output layer, since our dataset only contains two classes (change this accordingly if you have a different dataset with me). Now open up the file, you will see the first two layers are the data layers, which provide the input to the Network. Stanford University has an excelent tutorial on defining the Network in Caffe at here: [Caffe Tutorial](http://vision.stanford.edu/teaching/cs231n/slides/caffe_tutorial.pdf){:target="_blank"}. 

Let's change the path to the mean image and two LMDB folders which we created above:

{% highlight vim %}
name: "AlexNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: "data/DogsCatsKaggle/dogscatskaggle_mean.binaryproto" # MODIFIED
  }
  data_param {
    source: "examples/DogsCatsKaggle/dogscatskaggle_train_lmdb" # MODIFIED
    batch_size: 256
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227 
    mean_file: "data/DogsCatsKaggle/dogscatskaggle_mean.binaryproto" # MODIFIED
  }
  data_param {
    source: "examples/DogsCatsKaggle/dogscatskaggle_val_lmdb" # MODIFIED
    batch_size: 50
    backend: LMDB
  }
}
{% endhighlight %}

And there's only one place left to change: the output layer. Let's look through the file to find the layer named *fc8*, that's the last layer of our Network. It now has 1000 outputs because it was created to train on full ImageNet's images. Let's change the number of output according to our dataset:

{% highlight vim %}
layer {
  name: "fc8"
...
  inner_product_param {
    num_output: 2 # MODIFIED
...
}
{% endhighlight %}

Then save the file and that's it, you can now train the Network with your own dataset! We can't wait to do it, can we?

{% highlight Bash shell scripts %}
./build/tools/caffe train --solver=models/dogscatskaggle_alexnet/solver.prototxt
{% endhighlight %}

Our Network should be running flawlessly now. And all we have to do is wait until it's done! We have come a long way until this point. So I think we deserve a cup of coffee or something. That was so fantastic! You all did a great job today.

### Summary

So in today's post, I have shown you how to train the Network in Caffe, using your own dataset. We went through from how to download the data from URLs file (or directly from host), how to prepare the data to be read by the Network and how to make change to the Network to make it work using our dataset. As you could see, it was not so hard, but it did require some time to dig into. I hope this post can save you quite some of your previous times, and instead, you can spend them on improving your Network's performance. And that's all for today. Thank you for reading such a long post. And I'm gonna see you again in the coming post!



