---
title: "Machine Learning Part 8: Decision Tree"
header:
  teaser: tutorials/decision-tree/graph.png
categories:
  - Tutorial
tags:
  - machine-learning
  - decision-tree
  - classification
  - essential
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hello guys, I'm here with you again! So we have made it to the 8th post of the Machine Learning tutorial series. The topic of today's post is about **Decision Tree**, an algorithm that is widely used in classification problems (and sometimes in regression problems, too). *Decision Tree* is well-known not only for its great performance on classification, but also for its easy-to-understand algorithm.

As we have already seen up to now, in [Linear Regression](https://chunml.github.io/ChunML.github.io/tutorial/Linear-Regression/){:target="_blank"} and [Logistic Regression](https://chunml.github.io/ChunML.github.io/tutorial/Logistic-Regression/){:target="_blank"} tutorials, understanding how a learning algorithm works is somehow irritated. We got to go through boring theories, boring mathematical explanations and so on. Although I tried my best to make the explanation as simple as I can, but you know, functions is still functions, matrices are still matrices, there is no other way to get rid of those terms. But you won't have to go through that pain today. *Decision Tree* can totally be explained using human-understandable natural language. So keep reading, okay?

And before we get started, it's great to know that we have made it to the 8th post of the Machine Learning tutorial series. If you take a look back, I'm quite sure you will be surprised by how much you have progressed. We have learned two learning algorithms: *Linear Regression* and *Logistic Regression* respectively. We have worked on some simple dataset and visualized how your Model performed. At the moment, I'm quite sure that you are now familiar with Machine Learning. So in this post, we will use a more complicated set of data and see how our learning algorithms handle it.

So, let's talk about *Decision Tree*. You are somehow a real Model of Decision Tree algorithm yourself! In your daily life, you make many decisions exactly the same way *Decision Tree* does, subconsciously (of course). For example, your friend Joe invited you to his party. You may asked him back: "Any girls tonight?". He said yes. You asked him again: "Will Miley join too?". "Of course, homie!", he replied. And you accepted his invitation. The example I have just showed you is one of many situations that you may face everyday, in which you have to make your own decision on something. For that example, I can express it using the graph below:

![graph](/images/tutorials/decision-tree/graph.png)

Simply enough, right? You may be wondering: Is this real that such simple algorithm can solve complicated classification problem? The answer is: Yes! To make it more clear to you, let's consider a bigger one:

|Weather|Temperature|Humidity|Injure|Mood|RUN|
|-------|-----------|--------|------|----|---|
|clear|<10|<70|slightly|happy|NO|
|shower|20~30|>80|fit|stressed|YES|
|storm|10~20|>80|fit|happy|NO|
|shower|10~20|>80|slightly|stressed|YES|
|clear|>30|70~80|fit|lazy|YES|
|storm|20~30|>80|fit|stressed|NO|
|clear|>30|70~80|severe|happy|NO|
|clear|10~20|<70|severe|stressed|NO|
|shower|10~20|70~80|slightly|happy|NO|
|shower|>30|>80|fit|happy|YES|
|storm|20~30|70~80|slightly|happy|NO|
|clear|10~20|<70|slightly|happy|?|

Let's see what we have here. Here's the dataset I created for demonstration. I actually created based on my running experience, but some examples may sound weird to you, please ignore them for now, lol. 

Let's say we want to predict whether I am gonna go for a run, based on some factors such as Weather, Temperature or even my Mood! And as you has learned so far, Weathers, Temperature, Humidity, Injure, Mood are just the features, and the YES or NO in the RUN column is the targets (or labels).

Before going further into an appropriate explanation on how the algorithm works, I will first show you how we can solve the problem using the approach above.

First, let's randomly pick one feature from the features above. We will use that feature as the first condition, exactly the same way you asked "Any girls tonight?" above. To make it easy, let's pick the first one, Weather. As you can see, Weather can be either *clear* or *shower* or even *storm*. We will then split our data into three groups according to the Weather feature. In terms of Decision Tree, each time we use a feature to split the data, we create one **node**, and each group is called one **subset**. Let's first see how the *clear* Weather subset looks like:

|Weather|Temperature|Humidity|Injure|Mood|RUN|
|-------|-----------|--------|------|----|---|
|*clear*|<10|<70|slightly|happy|NO|
|*clear*|>30|70~80|fit|lazy|YES|
|*clear*|>30|70~80|severe|happy|NO|
|*clear*|10~20|<70|severe|stressed|NO|

Let's take a look at the RUN column. Obviously, just knowing the Weather is clear is not enough to decide whether to make a run or not. Just like the example above, you didn't make your decision after just one question, right? So we will have to look for another feature, hoping it will help us decide. Let's pick the *Temperature* feature. Now we can omit the Weather column (because it contains only *clear*), and just like what we did, we see that the Temperature feature can be either *<10* or *10~20* or *> 30*, so we will now have three more subsets, let's say, subsets of the subset above (where Weather is *clear*):

|Temperature|Humidity|Injure|Mood|RUN|
|-----------|--------|------|----|---|
|*<10*|<70|slightly|happy|NO|

|Temperature|Humidity|Injure|Mood|RUN|
|-----------|--------|------|----|---|
|*10~20*|<70|severe|stressed|NO|

|Temperature|Humidity|Injure|Mood|RUN|
|-----------|--------|------|----|---|
|*>30*|70~80|fit|lazy|YES|
|*>30*|70~80|severe|happy|NO|

Let's take a look at three new subsets above. The first two subsets are already clear, because the RUN column in each subset contains only one value. In terms of Decision Tree, we call the result in those subsets **Leaf Nodes**, and when they are now **pure**, which means that the output contains only one value. Meanwhile, the third subset still requires some further work. But it's quite easy now. Let's go ahead and use the *Injure* feature to create two new subsets:

|Humidity|Injure|Mood|RUN|
|--------|------|----|---|
|70~80|*fit*|lazy|YES|

|Humidity|Injure|Mood|RUN|
|--------|------|----|---|
|70~80|*severe*|happy|NO|

Up to now, all the nodes in the *clear* Weather subset are clear, since there's no any node in which the RUN column contains more than one value. We can now move on to the rest two subsets: the *shower* Weather subset and the *storm* Weather subset. By doing exactly the same way, we will get a result in the end, where all the subsets are clear. I have created a another graph for a better visualization:

![run_graph](/images/tutorials/decision-tree/run_graph.png)

Using the graph above, we can now predict the value of the RUN column for our last row above:

![run_graph_pred](/images/tutorials/decision-tree/run_graph_pred.png)

So, it's likely that I'm gonna stay at home with my PlayStation 4 that night, lol.

So that's it. Just easy to understand like I said earlier, right? Of course, we have some kind of mathematical explanation for how Decision Tree actually does. For example, choosing which Feature to split in the beginning is not done randomly, but depends on some considerations. And each time we need to create new subsets from the parent subset, the process is repeated again. Why do we have to make things such complicated, you may ask. Technically say, Decision Tree is a greedy algorithm, which means that it's likely to fall into local-minimum rather than the desired global-minimum, which means we may get an ugly result if we run out of luck. I will give you a simple explanation for Decision Tree algorithm below for ones who concern. You can skip it to jump directly to the Python Implementation because the explanation is just optional.

### Decision Tree: ID3 Algorithm

So you decided to give Decision Tree's algorithm a look. I really appreciate that. Knowing what is behind the scenes in somehow unrejectable, right? I won't waste any other minute of your time. Let's go straight into the algorithm. Actually, since Decision Tree was introduced quite long ago, the original algorithm has been revised and improved so many times, which the successor became more complex and robust than its predecessor. Among those, ID3 may probably be the best well-known algorithm. I think once you understand ID3, you can understand all its successors without problems.

So, how does ID3 algorithm work? To make it as simple as possible, I list out all the steps below:

### 1/ Call the current dataset \\(S\\). We will then compute the **Entropy** \\(H(S)\\) on S as follow:

$$
H(S)=-\sum_{j=1}^Kp(y_j)\log_2p(y_j)
$$

, where \\(K\\) is the number of classes, \\(p(y_j)\\) is the proportion of number of elements of \\(y_j\\) class to the number of entire elements in output of \\(S\\):

\\(H(S)\\) tell us how uncertain our dataset is. It ranges from \\(0\\) to \\(1\\), which \\(0\\) is the case when the output contains only one class (pure), whereas \\(1\\) is the most uncertain case.

And in case you may ask, yes, that's exactly the same as the entropy cost function that I showed you in Logistic Regression tutorial (except the \\(\frac{1}{m}\\) term). As you already knew, the smaller the entropy function, the better classification result we can achive.

### 2/ Next, we will compute the *Information Gain* \\(IG(A,S)\\). 

Information Gain is computed seperately on each feature of the current dataset \\(S\\), whose value indicates how much the uncertainty in S was reduced **after** splitting \\(S\\) using feature \\(A\\). We can see that it looks like some kind of derivative, where we take the difference of the Entropy before and after splitting:

$$
IG(A, S)=H(S)-\sum_{i=1}^np(t)H(t)
$$

, where \\(A\\) is the feature used for splitting, \\(n\\) is the possible number of values of \\(A\\), \\(p(t)\\) is the proportion of number of elements whose values is \\(t\\) to the number of all elements of feature \\(A\\).

### 3/ After compute all the Information Gains of all features, we will then split the current dataset \\(S\\) using the feature which has the highest Information Gain.

### 4/ Repeat from step 1 with the new current dataset until all nodes are clear.

That's it. But just skimming through the algorithm may not make any sense about how the algorithm works, right? So, let's use the dataset above as an example.

First, our current \\(S\\) would be the entire original table, right? We will compute its Entropy \\(H(S)\\). Look at the RUN column (which is our output), it has \\(4\\) YES and \\(7\\) NO over \\(11\\) examples, so its Entropy will be as follow:

$$
H(S)= -p(YES)\log_2p(YES)-p(NO)\log_2p(NO)=-\frac{4}{11}\log_2(\frac{4}{11})-\frac{7}{11}\log_2(\frac{7}{11})=0.9457
$$

Next, we will compute Information Gain on each feature. Let's first look at the Weather feature. It has three possible values: *clear*, *shower* and *storm*. *clear* Weather has \\(4\\) examples, so we have: \\(p(clear)=\frac{4}{11}\\). In those \\(4\\) examples, we have \\(1\\) YES and \\(3\\) NO, so the Entropy of *clear* Weather will be:

$$
H(clear)= -p(YES)\log_2p(YES)-p(NO)\log_2p(NO)=-\frac{1}{4}\log_2(\frac{1}{4})-\frac{3}{4}\log_2(\frac{3}{4})=0.8113
$$

We now can do similarly to the rest two values to obtain values like below:

$$
p(shower)=\frac{4}{11}\\

H(shower)=-p(YES)\log_2p(YES)-p(NO)\log_2p(NO)=-\frac{3}{4}\log_2(\frac{3}{4})-\frac{1}{4}\log_2(\frac{1}{4})=0.8113\\

p(storm)=\frac{3}{11}\\

H(storm)=0
$$

Note that in the case of *storm* Weather, its output contains only *NO* value, so its Entropy will be \\(0\\), you don't need to compute it by hand (and even if you do, you'll soon realize that it is impossible because of the term \\(\log_2(0)\\)!).

And I would like to talk a little bit about the case when \\(H(t)=1\\), which I mentioned it as the most uncertain case. We can only obtain that value when the proportion of each class is equal to others'. In the case of our current dataset, if the number of YES is equal to the number of NO on the considered subset, then it's easy to see that there is a big chance that it can't be fully classified (that's why we call it the most uncertain case).

So now we can compute the Information Gain on the Weather feature as follow:

$$
IG(Weather, S)=0.9457-\frac{4}{11}*0.8113-\frac{4}{11}*0.8113-\frac{3}{11}*0=0.3557
$$

Continue repeat this process with other features, you will likely end up with results like this:

$$
IG(Weather, S)=0.3557\\

IG(Temperature, S)=0.1498\\

IG(Humidity, S)=0.2093\\

IG(Injure, S)=0.2093\\

IG(Mood, S)=0.2275
$$

From the results above, IG on Weather has the highest value, so use Weather as a splitting condition will have the highest chance to reduce the uncertainty of dataset \\(S\\), and may lead to a good classification in the end. 

So that's all I have to tell you about Decision Tree's ID3 algorithm. Hopely this explanation somehow can help you have a deeper understanding about what was actually done behind the scenes. You may want to read more about its successors such as C4.5 or C5.0 algorithms. And you will find them not so hard to understand at all!

Now, let's jump to the implementation! Can't wait no more, can you?

### Decision Tree with scikit-learn

As I mentioned in the previous tutorials, the **scikit-learn** library comes bundled with everything you need to implement Machine Learning algorithms with ease. Furthermore, the library provides us many methods to generate data for learning purpose. And today I will use one of them to create a more complicated dataset, just to see how well Decision Tree can handle that.

First, let's import necessary modules as usual:

{% highlight python %} 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
{% endhighlight %}

If you went through my last tutorial, you may now know a proper way to evaluate the performance of a Machine Learning Model using the *cross_val_score* method. And if you didn't, you can always give it a look here: [Cross Validation](https://chunml.github.io/ChunML.github.io/tutorial/Cross-Validation/){:target="_blank"}.

Next, let's using *make_classification*, a method provided by scikit-learn library to generate data for classification problems.

{% highlight python %} 
X, y = make_classification(n_samples=100, n_features=2, 
	n_redundant=0, n_classes=2, n_clusters_per_class=1)
{% endhighlight %}

Let's plot the data we created to see what it looks like:

{% highlight python %} 
X1_min, X1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
X2_min, X2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
axes = plt.gca()
axes.set_xlabel('X1')
axes.set_ylabel('X2')
axes.set_xlim([X1_min, X1_max])
axes.set_ylim([X2_min, X2_max])
plt.show()
{% endhighlight %}

In my case, the data looks like below. Note that your graph may be way different, since the *make_classification* method generates data in some unpredictable way.

![data](/images/tutorials/decision-tree/data.png)

As you might notice, I also included Logistic Regression. That is because I want to compare the performance of the two classification algorithms that we have known so far. From now on, you will learn some more algorithms, and it's always a good practice to compare between them, to see what algorithm work best for a particular problem.

Let's train our Model using one algorithm at a time, and print out the mean accuracy of each algorithm respectively. We will, of course, use the *cross_val_score* method for evaluation:

{% highlight python %} 
clf = DecisionTreeClassifier()
score = np.mean(cross_val_score(clf, X, y, cv=5))
print('Decision Tree: {}'.format(score))

clf = LogisticRegression()
score = np.mean(cross_val_score(clf, X, y, cv=5))
print('Logistic Regression: {}'.format(score))
{% endhighlight %}

So let's run our code a few times and see how the two algorithms perform. Note that you have to run the *make_classification* too, or you will likely get the same results!

Here's my results:
{% highlight python %} 
Decision Tree: 0.97999
Logistic Regression: 0.99

Decision Tree: 0.98999
Logistic Regression: 0.99487

Decision Tree: 0.91
Logistic Regression: 0.935
{% endhighlight %}

As you see in the result above, Decision Tree's performance is not any better (or even worse) than Logistic Regression's. Let's go ahead and try customizing the parameters of *make_classification* method (like increasing number of features, number of classes or number of samples, etc), you will likely realize that the two algorithms don't have any significant difference. That is because the data generated by *make_classification* is linear, which means that Logistic Regression shouldn't find any difficulty in fitting the data. Also note that we initialized both with default hyper-parameters (don't mess with our \\(\theta\\) parameters, I will have a post on tuning hyper-parameter soon!), which means that our algorithms can perform even better!

So, let's try another dataset. This time I will create a non-linear dataset, and see if these two algorithms can handle as well as they did above. Just like *make_classification* method for generating data for classification problems, scikit-learn library provides us some more methods to generate some particular dataset. One of those is *make_circles* method, which helps generate data with circular distribution. So let's use it to create a new dataset, and see how it looks like:

{% highlight python %} 
from sklearn.datasets import make_circles

X, y =  X, y = make_circles(n_samples=200, noise=0.2, factor=0.5)
{% endhighlight %}

The code for drawing the graph is exactly same as above so I omit it for now. I highly recommend you to put everything in a python file, so you just need to modify a few lines, without having to re-run the code for drawing the graph!

In my case, I have a graph like below:

![circle_graph](/images/tutorials/decision-tree/circle_graph.png)

As you could see, our data is now no longer linearly distributed. I can't wait to see how the two algorithms handle it. After I run the code three times, I got the result below:

{% highlight python %} 
Decision Tree: 0.84
Logistic Regression: 0.455

Decision Tree: 0.86
Logistic Regression: 0.465

Decision Tree: 0.83
Logistic Regression: 0.45
{% endhighlight %}

Now what? The accuracy of Logistic Regression seems horrible, right? That was somehow predictable, because we have already talked in the previous tutorial that a straight line may not work well with non-linear dataset. But look at the results of Decision Tree, they are satisfying, I think, since we haven't tuned any hyper-parameters of the algorithm. So now you have your first tool to deal with non-linear dataset! Decision Tree is simply incredible. The algorithm is easy to understand, which you can explain to your friends just like telling them a story, and yet the performance is impressive.

### Summary

So today, we have just talked about Decision Tree and used it to solve a few classification problems. We have seen an outstanding performance on different kinds of data. It doesn't matter whether our dataset is linear or non-linear, the algorithm can do the jobs just fine. That is definitely a great tool that you have just added to your Machine Learning toolbox!

In the next post, I will tell you about another powerful classification algorithm, the one which nearly turned Neural Network into history (Neural Network was introduced quite very long ago, though). If you want to know what it is, stay updated for the next post! See you!
