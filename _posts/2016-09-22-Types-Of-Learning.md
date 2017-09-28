---
title: "Machine Learning Part 2: Types of Learning"
categories:
  - Tutorial
tags:
  - machine-learning
  - supervised
  - unsupervised
  - reinforcement
  - regression
  - classification
---

So here we are again, to continue what we dared to begin: challenging Machine Learning.

Before we move on, let's spend a minute to look back on the last post. You now know what Machine Learning is, it's something that contains a Model, some (Features, Labels) as inputs (remember we needed many Dog's images?) and some Activations as outputs (Dog or Not a Dog).
You also know what exactly Learning means. It's the process of Trial-and-Error repeatedly. Trial makes big mistakes in the first time, but help the computer (as well as human's brain) to learn faster, to become more confident on the next time.

Learning is doing something repeatedly, guys. So don't be ashamed to do some revision until you're ready to move on.
You can find the last tutorial here: [What is Machine Learning?](https://chunml.github.io/tutorial/Machine-Learning-Definition/){:target="_blank"}

So, let's get down to business. I'll make it easy for you today. It means today's post won't be such long as the previous one. Even Cristiano Ronaldo needs some recovery-day, right?

#### Types of Machine Learning

What I want to talk to you today, is about types of Learning. So why the hell must we know about it. So imagine you're trying to learn piano in a mathematical way, it means that you treat those beautifully composed repertoires as some kind of, let's say those annoying matrices you saw yesterday. You'll soon realize it doesn't make any sense, logically. And even if you can play through it, but actually you just can't get it (if you're listening to classical music, then you will know what I mean).
So we have to admit that, different things require different learning approaches. If you choose an inappropriate way of learning, then it'll likely result in some unpleasant tragic.

Sounds like a headache here, right? But don't worry, it's not that complicated in the case of Machine Learning. Above I told you that you won't be able to get progressed in learning piano with some mathematical approach. In the case of Machine Learning, piano repertoires can be treated as Inputs. So different Input types do require different approach of learning. It does make some sense, right?
So, based on Input types, we can divide Machine Learning into three categories like below:

* Supervised Learning
* Unsupervised Learning
* Reinforcement Learning

### Supervised Learning
First, as you may know (in fact I think you don't even care), that I'm currently living in Japan (I'm not Japanese, though). When I tried to learn some Machine Learning's vocabularies in Japanese, I discovered some couple of interesting things. One of those is Supervised Learning term. Why am I telling you this? Because I think it may help you understand this term easily. They call it "Learning with a teacher" in Japanese. Does it make some sense to you? When your parents sat by your side and taught you each picture was about, they acted just like you teachers. When you tried to learn for you SAT or GMAT, it was not you who learned by yourselves! Where did you get those vocabularies from? From your teachers? From you instruction books? Doesn't matter, at least you learned from some sources. Or I can say, you were learning under the supervision of something. It means you knew what were right and what were wrong at the time you were learning.
Everything is exactly the same in Machine Learning, Supervised Learning indicates that the computer knows whether it made a right guess or not, using the Label along with each Feature. Remember the Dog Recognition example on the previous post? Yep, it used the Labels to evaluate its work. So, whenever you see a Machine Learning problem where Labels provided along with Features, that's definitely Supervised Learning.

### Unsupervised Learning
Obviously, Unsupervised Learning is the opposite of Supervised Learning. It means that you will be given a mess of Features without any Labels on them. So it doesn't make any sense here, you may suppose. How can the computer actually learn without knowing whether it's doing right? A picture is worth of thousand words. Let me show you:

![Image_1](/images/tutorials/types-of-learning/Image_1.jpg)

As you can see in the picture above. We have two set of points (just ignore the axises' names, I just want you to focus on the points). Some are red, and some are blue. And it's obvious that we can draw a line between them, let's say the green one. As you might get, it's exactly an another example of Supervised Learning, where the Labels are Red & Blue.
(For ones who find it hard to understand. Just imagine the Red points are images of "Dog", and the Blue ones are images of "Not a Dog". But don't worry, I'll make it more concrete when we come to Logistic Regression!) 
So, what about this one:

![Image_2](/images/tutorials/types-of-learning/Image_2.jpg)

I just simply made them all Blue! So what's the computer supposed to do now? As you could see from the previous example, the Labels may help in the learning process, but they're not something which we can't live without. It's the distribution of Features which matters! For example, the computer may not see the "Dog" label (or the "Not a Dog" label, either), but it can put things with four-leg features, black-nose features, long-tail features, long-tongue features in the same group. So in the end, we can have the same result with what we got from Supervised Learning.
(Of course, actually it can't be always the same! Imagine that the computer will also have a group for things with two-leg features, two-swing features (for birds' images); a group for things with no-leg features, long-tongue features (for snakes' images) and so on. That's because Unsupervised Learning is not limited by the Input Labels, so the result may vary depending on how the computer learns)

It may be too long to write all about these two types of Learning. Actually in this tutorial, I will mainly focus on Supervised Learning because it's very common, not only on research but also in real life projects. And via Supervised Learning you will understand the algorithms much faster, much deeper without hurting your enthuasiasm.

### Reinforcement Learning
To be honest, I rarely talk about something I don't know much about. Actually at the time of writing, I'm just in the beginning of my research on Reinforcement Learning. So for the sake of simplicity, Reinforcement Learning is the learning through the interaction with Environment. Supervised Learning and Unsupervised Learning are not. Why are they not? Remember the way they actually learn? We give them a set of Input for them to learn from. And they use what they learned from that Input to make predictions (another way of saying "Guess", I guess). So the way they predict depends entirely on the Input they learn from. We may see the disadvantage here, right?
So Reinforcement Learning appears as a solution to eliminate that disadvantage. You can simply think that the computers will actually learn from dynamic Input, it makes them much more robust, precise and confident. But the trade off, that kind of learning is very complicated, especially for Machine Learning's newcomers. So just put it in the list, OK?

So, now you know about three main types of Machine Learning (there's a lot out there but you only need to focus on these three, I think). I hope after this post, you can have a (slightly) deeper thought about Machine Learning. In the next post, I'll talk about Linear Regression, the most common and basic algorithm (let's say for almost advanced algorithms). And we'll finally get our hands dirty with some coding (sounds cool?). So stay updated, and I'll see you there. See ya!
