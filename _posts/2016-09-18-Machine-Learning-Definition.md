---
title: "Machine Learning Part 1: What is Machine Learning?"
categories:
  - Tutorial
tags:
  - machine-learning
  - model
  - feature
  - learning
---

You've been hearing about machine learning in the last few years, especially on the day on which AlphaGo was hotter than Donald Trump. You felt excited, you wanted to know what it is, or even more, you wanted to dig into it. But you just still can't get it? Let me help you this time. 

### So, the very big question, and maybe the biggest reason which led you here: What exactly is Machine Learning?

Well, I've got no intention to repeat something that was already written on Wikipedia (you're tired of it, right?). So my definition of Machine Learning is: we don't teach the computers to work anymore, we make them learn, to do things itself!

Still confused? Let's see something cool instead.

![Image_1](/images/tutorials/what-is-machine-learning/1.jpg)

So as you can see in the picture, we have a model (the big one in the middle), an input which takes care of something call (Feature, Label) and an output which gives us something call Activation. So what the hell are all these things?

Let's imagine you want the computer to do you a favor: to tell you whether there is a dog in an image. So it becomes much clearer when I match up everything like the image below:

![Image_2](/images/tutorials/what-is-machine-learning/2.jpg)

Firstly, what is Features?
If this is the first time you hear it, then congratulations, you will see this keyword everywhere, everywhen from now on (as long as you keep up with Machine Learning, of course). So let's make it simple. Look at the dog. He has four legs (like the other dogs on Earth), his nose is black, he is hanging out his tongue, he has a tail, ... Yeah, those are what come up when you see this dog, and importantly, you will see those in other dogs too. So it turns out to be some distinct features
which make human's brain know it just saw a dog. Wait a minute, did I just say Features? Well, I hope you get the point here. So, Features are something distinct that make something different than the others. Human's brain recognises something by their Features, so why don't we teach the computer that trick? (Come on, don't be that selfish man)

I know you're excited now. Ready to move on? So what about Label? Well, it's much simpler than the Feature. Label is a name of guys who have specific features. So we call guy who have four legs, one black nose, one long tongue and one long tail a DOG. Well, I just can't make it simpler. So what about the other guys? At this time, I just call them "Not a dog". Maybe you will ask me why I don't make it more specific, just like "a Cat", "a Bird" or something. You will have the answer when we come to Logistic Regression. Oh my God, please don't care about that name. I'll remind you when the time is up. So please stay patient :) There is a lot of fun ahead, I promise.

So I've talked about Features and Labels, but what to do next? Remember when we were kids, our parents bought us some picture books? Then we spent days after days, kept pointing in and speak out loud names of things we saw, with so much pride. Omg, my parents must be so patient with me back in the day. So what the hell does the childhood's story make sense here? Well, we will do exactly what our parents did. We show them images, a large amount of them, just like this:

![Image_3](/images/tutorials/what-is-machine-learning/3.jpg)

As you can see, each image is labeled whether the image contains a dog. Just like what we did in the past, the computer will try to learn from the images we shown them. I said it LEARNS, not MEMORIZES. We could learn, so can the computer, right?

And lastly, what is Activation? Simply speaking, we can call it a Guess. Imagine your computer knew how to tell whether it saw a dog (we just assume that, we'll back to it right below), then you show it some image it has never seen before, so it has to Guess, based on its own mind. And you can easily figure it out, the Guess (or Activation) must be one of the Labels we taught them. (We can't teach them about Dogs, then force them to recognise a Cat, it doesn't make any sense right?)

To make it more concrete, let's see the image below:

![Image_4](/images/tutorials/what-is-machine-learning/4.jpg)

Yeah, at least the computer can now distinguish between a Dog and a Crab. Wait! It doesn't know about a Crab, at this point it only know the second image is NOT a Dog. Okay, whatever, though. It gets better someday, definitely.

### When we talk about Machine Learning, we talk about Features, Labels, Activations, and ...

So up to this point, I assume that you can draw your own sketch about what Machine Learning actually is, what Machine Learning actually does (or be done). Quite simple right? Then I have something for you:

![Image_5](/images/tutorials/what-is-machine-learning/5.jpg)

WAIT! Where're you going? Grabbing your algebraic book or something? Or just finding your PS4's controller? Take it easy, man. You won't even have to grab a pencil to do some maths today, I promise!
So why the hell I shown you this? Because I'm afraid after you read this post, you'll run outside, tell your friends about Machine Learning, yeah you'll tell them about something like "four-leg feature", "long-tail feature", "a Dog", or "not a Dog"... Please don't do that. Sorry, I'm just joking, kind of over-joking. Let's get back to business. We all know about binary number (something like a switch with On-Off state, 0 or 1). And sadly, that's what computer can understand. So it won't see something like "four-leg feature", "long-tail feature". The things they actually see is just boring repeating 0s and 1s. All I want to say here is, Features, Labels & Activations, they're all matrices to computers. At this point I just want you to know it. (Maybe I don't even need to speak that long, right? Sorry, my bad again!)

Let me summarize all the things we talked above by the image below:

![Image_7](/images/tutorials/what-is-machine-learning/7.jpg)

Sounds familiar, right? Let's imagine the Dog recognition is taken place somewhere in the computer, we call it a Model. The final goal is to help create this. We don't create, we're just the helpers, the computer must handle it alone. Don't forget that it's the one that learns :)

### When we talk about Machine Learning, we talk about Features, Labels, Activations, and of course, a Model.

So next step is, Training? Sounds like we're in the middle of the pitch and do some crazy training, just like Cristiano Ronaldo? No, of course we don't (or actually we can't). We're not training muscles, we're training the brain, the computer brain. That's the process when the computer learn from the images we shown it. But let's talk a little bit about how it learns. We won't even teach them (although we said machine LEARNING, so learning from what?)
Imagine the label of each image is printed on the back so the computer can't see. I know you got it. Exactly what we do with our flashcards, the things you can't live without to prepare for your SAT, GMAT or something like that. Firstly, the computer will try to guess, without knowing anything about what a Dog should look like. Then it flips the back over, sees the answer, and learns. Yep, it learns from making MISTAKES. Just like human does, the bigger the mistake is, the faster it learns.

Then next step is Evaluation? I assume we gave the computer 100 images (roughly 50 images of Dog, and 50 images of Not a dog). It will try to guess each image until it finish the 100th one. Then it counts the number of right guesses. For example, there are 69 right guesses, so it has a probability of 69% in recognising whether there is a dog. Not so bad, right? That's what we call Evaluation. The computer evaluates its own work:

![Image_8](/images/tutorials/what-is-machine-learning/8.jpg)

But what if the evaluation result can't satisfy us? We just tell the computer about that, it will update its own Parameters. But what the hell are 'Parameters' here? So much new keywords today! (Maybe many of you want to tell me that!) Remember the Model I said above? Parameters are just something put into the Model, just like machines in the factory. If something goes wrong with the product, they just have them fixed. Exactly the same with Parameters. Simple, right? Just as the image above, let's call it 'θ'. Some other places you will see thay use 'W' for Weights, and 'b' for Biases, but right now for the sake of simplicity, just remember 'θ'. Don't ask me about the naming convention, I don't know! Everyone uses that, and you don't want everyone to look at you just like "What the hell that guy is talking about!" when you talk about Machine Learning, right?

So the loop Training-Evaluating-Parameter Updating is actually the core of Machine Learning. Until now, I think you can accept this easily without hurt. Thanks again, cool Dogs in the beginning :)

### The core of Machine Learning: Training - Evaluating - Parameter Updating then repeat!

So last, but not least, I just want you to do me a favor. I'm definitely sure you can. Did you notice the question mark on the fifth image (The annoying image with some kind of algebraic matrices)? So that's the last thing I want to reveal in the end of this post. And once again I'll show you some algebraic stuff, but stay calm, you won't get your hands dirty today!

![Image_9](/images/tutorials/what-is-machine-learning/9.jpg)

Okay, such a long function which likely comes from hell. Please ignore it right now. All I need you to know are just: Activation Function, Bias and Weights. I won't make it long this time.
So from Feature to Activation, we'll go through Activation Function, very easy to remember, right?
The parameter standing alone is call Bias, the other parameters which have their own Xs, are called Weights. That's all for today. Why are they called Bias & Weights, I'll reveal it in the next tutorial.

So, finally we made it till the end. You may ask why I had to make such a long post, and sometimes talked so much about simple things. Well, the later posts will definitely contain a lot of things you don't want to face (such as algebraic stuff, ...), you will even get your hands dirty with some coding work (Cool!). So I have to make sure you don't make any wrong assumption in the beginning, you know, people make mistakes by inappropriate assumptions, right?

Hope this post is helpful to you on the long-run towards your Machine Learning's targets. 

So stay updated, and I'll meet you on the next post of this tutorial. See ya!
