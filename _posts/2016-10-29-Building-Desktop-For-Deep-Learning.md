---
title: "Building up my own machine for Deep Learning"
header:
  teaser: projects/building-desktop-for-deep-learning/all.JPG
categories:
  - Project
tags:
  - machine learning
  - deep learning
  - PC
  - desktop
  - GPU
  - GTX 1070
---

Hello guys, it's me again being here with all you guys today. There will be no tutorial in this post, so at first I'm sorry for that. But Halloween is coming to town, right? The weather is good out there, too. So there's no reason to stay at home spending time on some boring Machine Learning stuff, right? But I did! I spent a whole day on building my own desktop for Deep Learning!

Of course I still remember telling you about using *g2.2xlarge* instance of Amazon Web Service. For ones who are considering using some GPU instance with a reasonable price, please have a look at my previous post:  
[Using AWS EC2 Instance](https://chunml.github.io/ChunML.github.io/project/Prepare-AWS-Instance/){:target="_blank"}.

The main reason why I decided to build my own machine is just simple: I needed a more powerful GPU. Of course I can change to *g2.8xlarge* instance but I will soon get broke. Having a desktop at home also makes sense for one who mainly works with images and videos like me, since I can visualize the output without writing more codes to use the raw results from EC2 instance.

The third reason is: because it's kind of cool, lol. To be honest, I have never done it before, so I was really eager to try.

I was thinking about that for a while. I was surfing on Googles, seeing some cool guys doing some reviews on performance of different GPUs. And below is the build I chose for my own, eventually:

![all](/images/projects/building-desktop-for-deep-learning/all.JPG)

1. CPU: Intel Core i7-6700K 4.0GHz Quad-Core Processor
2. CPU Cooler: Scythe Kotetsu 79.0 CFM
3. Motherboard: ASRock Z170 Extreme4 ATX LGA1151
4. GPU: Gigabyte GeForce GTX 1070 8GB G1
5. RAM: Corsair Vengeance LPX 16GB (2 x 8GB) DDR4-2666
6. HDD: Seagate Barracuda 2TB 3.5" 7200RPM

The most important part is the GTX 1070 GPU. At first I intended to build with GTX 970, but I soon realized that the performance of GTX 970 is just not much different from GTX 960, so it may not a good choice especially if you mostly work with Convolutional Neural Network. Guys on Deep Learning community recommended to use at least GTX 980 for CNN. And because GTX 1070 is just slightly more expensive than GTX 980, whereas GTX 1080 is still a big deal, I thought GTX 1070 was the best choice for me.

Working with Deep Neural Network requires mostly the power of GPU, so we don't necessarily buy a giant-killer CPU. But a little guy won't be a good fit (comparing to the giant GTX 1070), so I chose Intel Core i7-6700K. I also needed a good CPU cooler (not a very big one), too. 

Since GTX 1070 will be with my team, not only the CPU, but I also needed a motherboard which can hold them well, and has an efficient power consumption, too. My choice was Z170 Extreme4, a motherboard mainly used for gaming PC.

The next to consider is storage and RAM. CNN requires a lot of memory to hold the network temporary values (all the parameters and gradients from backpropagation), and a great deal of memory to save the Model, so now I started with 16GB of RAM, and 2TB internal hard disk.

The last one is the PSU (power supply unit). I chose a 600W PSU from a Japanese brand, and they don't sell them outside Japan (as I searched on the net and found no result), so I don't list out the PSU I bought here. You can buy from your local brand, but I recommend that you choose a PSU which can provide from at least 600W, because a big GPU will likely consume a lot of power.

So that's was all about the parts I need. It's time to put things together (actually it was finished at the time of writing).

Here is the Z170 Extreme4, fantastic design, isn't it?

![z170](/images/projects/building-desktop-for-deep-learning/z170.JPG)

And if there's something I forgot to mention, that is the big one below:

![case](/images/projects/building-desktop-for-deep-learning/case.jpg)

Since the GTX 1070 is 28cm long, obviously I needed a big case in order to put all these things in. I chose the NZXT S340 Mid Tower case. The price wasn't good at all, but in the end it turned out that it's worth every *yen*.

Next, let's first put the motherboard into the case. I heard some guys recommend putting everything on the motherboard first, then put all into the case. I think both ways work well. But I think it's easier if in the beginning our motherboard is mounted stably somewhere, and there's no better place than its final shelter, right?

Here's the picture of my Z170 put into the S340 case:

![z170_case](/images/projects/building-desktop-for-deep-learning/z170_case.JPG)

Next, let's take a look at the tiny CPU. Despite of the power it has, it's literally tiny comparing to other parts.

![cpu](/images/projects/building-desktop-for-deep-learning/cpu.JPG)

But it was not until I pull it out that I realized it was heavier than it looks. Maybe that was because the metal which helps it transmit temperature adds more extra weight. Next, let's mount it onto the motherboard.

![cpu_z170](/images/projects/building-desktop-for-deep-learning/cpu_z170.JPG)

When I tried to close the cover afterward, it was a lot heavier than it was supposed to be. And I kept wondering for a while, whether I should add more strength or something. After watching some guys on Youtube mounting the core i7-6700k, and mentioning about how hard it was to close the cover, I decided to add more force, too. It turns out that the shape of the core i7-6700K, LGA 1151 caused that problem.

Next, I put the cooler on the top of the CPU. It was a little bit tall, that I was afraid if it fit my case. Fortunately, it does.

![cooler](/images/projects/building-desktop-for-deep-learning/cooler.JPG)

And here comes the boss. I had to say, it was so amazing. Fantastic, tremendous design that I couldn't have been able to imagine a GPU could be!

![gpu](/images/projects/building-desktop-for-deep-learning/gpu.JPG)

And here it is after put onto the motherboard. It takes a great deal of space and makes everything look small.

![gpu_case](/images/projects/building-desktop-for-deep-learning/gpu_case.JPG)

The rest was so simple: putting the RAMs, the PSU, and carefully plug them into the right places. And my "Monster" is ready to be unleashed:

![done](/images/projects/building-desktop-for-deep-learning/done.jpg)

![done_2](/images/projects/building-desktop-for-deep-learning/done_2.jpg)

![done_3](/images/projects/building-desktop-for-deep-learning/done_3.JPG)

The moment I stood up to plug the PSU's cord, I realized that I was sitting for nearly 7 hours! My Saturday was just an extended Friday! But at least, it worked in the end. And I can draw a smile in my face now. I also realized that I just ate some noodles for breakfast, and I started to feel hungry now. Few more hours left for Saturday, and I've been thinking about curry for a while. So, I'll go get some beef right now. Thank you all for watching. If you need some experience on building your own machine for Deep Learning, feel free to contact me. I'll be glad to help! Goodbye and see you in the next tutorial!


