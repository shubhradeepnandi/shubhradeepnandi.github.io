---
title: "Machine Learning Part 7: Cross Validation"
categories:
  - Tutorial
tags:
  - machine-learning
  - cross validation
  - data splitting
  - overfitting dianosic
  - stratifiedkfold
  - cross_val_score
  - train_test_split
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hello there, so I'm here again to bring you the 7th post on my Machine Learning tutorial series. Today I will talk about **Cross Validation**, a term that you must know and mostly apply in all Machine Learning problems.

So, to go straight to the main topic, what is Cross Validation anyway? And why must we know about it? Why must we apply it to our Machine Learning problems? The simple answer is: because of *Overfitting*.

Maybe my answer is so straight, and somehow confusing. Okay, let me fix that. Remember in the previous post on [Underfitting and Overfitting Problems](https://chunml.github.io/ChunML.github.io/tutorial/Underfit-Overfit/){:target="_blank"}, I talked about the two most common problems in Machine Learning? Underfitting is the problem when our algorithm failed to create a model that can fit the data, which means that our Model tends to give bad predictions even on the training dataset. Conversely, Overfitting is the problem which our Model fit training data so well that it tends to memorize all the features, therefore it performs badly on the data it has never seen.

So, how do we at least foresee those problems before we can think about any solutions for them? As you might notice in the end of that post, I mentioned a little bit about **splitting the dataset**. Through the examples I showed you, it's clear that we cannot just rely on the performance over the training data. We need a seperate dataset to use for performance evaluation, which assures that our Model has never seen before. That is the main idea of the term **Cross Validation**, which I will talk more further about today.

I think that's enough of talking. Let's go down to business. Open your terminal, and get it started by initializing the data used in this post:

{% highlight python %}
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

X1_1 = np.linspace(0, 5, 100)
X1_2 = np.linspace(0.2, 5.2, 100)
X2_1 = np.random.rand(100) * (15 - 3 * X1_1)
X2_2 = np.random.rand(100) * 3 * X1_2 + 15 - 3 * X1_2

X1 = np.append(X1_1, X1_2)
X2 = np.append(X2_1, X2_2)
X = np.concatenate((X1.reshape(len(X1), 1), X2.reshape(len(X2), 1)), axis=1)
y = np.append(np.zeros(100), np.ones(100))
{% endhighlight %}

Let's have a look at our data by plotting it onto the coordinate:

{% highlight python %}
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.show()
{% endhighlight %}

![Graph_1](/images/tutorials/cross-validation/graph.png)

You can see that now we have data of two classes which are linearly seperated (which means you can seperate them by a straight line). I don't want to use any complicated data today, because I have just shown you two basic learning algorithms. Furthermore, the thing I want you to focus is **cross validation**, not how to create a sophisticated Model.

Now with the data all set. Now what to do next? As I mentioned before, we will split our original dataset into two portions: one for training and one for testing. The standard splitting ratio is 70% for training data and 30% for test data (in other places, you can see they use another ratio such as 6:4, etc). So let's do it. Here I will use the first 140 elements as training data, and the rest as test data.

{% highlight python %}
X_train = X[:140, :]
X_test = X[140:, :]
y_train = y[:140]
y_test = y[140:]
{% endhighlight %}

Next, let's create our Logistic Regression object and train the model using the split data above:
{% highlight python %}
clf = LogisticRegression()

clf.fit(X_train, y_train)
{% endhighlight %}

The training should take less than one second as usual. Now we have our Model learned. Let's see how well it performs over the training dataset:

{% highlight python %}
clf.score(X_train, y_train)
{% endhighlight %}

Using the training data to evaluate our Model gave me an accuracy of approximately 92.14%. Not bad, right? But I think you got experience now. You won't speak a word until you see the performance over the test data, right?

So let's go on and grab the result on the test data:

{% highlight python %}
clf.score(X_test, y_test)
{% endhighlight %}

In my case, I got a result of 40%. That was a real **NIGHTMARE**, wasn't it. You may shout out loud. Why do we still get that frustrating result, despite the fact that we used cross validation? Well, let's calm down and figure out why.

Obviously, our current Model performs unacceptably bad. In real life projects, that will be a headache that we must stay calm and find where the problem is (though it may not be easy). There are a lot of reasons which cause a bad performance over the test data, but the main reason is the data itself. So it's a good idea to take a look at the test data, maybe we will find something wrong with it.

{% highlight python %}
y_test
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
{% endhighlight %}

Wow! Since this is a two-class classification problem, having a test dataset which only contains labels from one class is somehow unsual. To make it more clear, let's print out the training data:

{% highlight python %}
y_train
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
{% endhighlight %}

I'd better show the training data, I think. As you can see, the training data above was split unequally between two classes. It means that one class has much more data than the other. What does it affect the learning process? The more data you provide, the better it learns. So our Model is likely to learn the \\(0\\) class better than the \\(1\\) class. It is even worse that our test data contains only data of the \\(1\\) class, which resulted in a very bad predicting accuracy as you saw. So obviously, it's necessary to split the data equally between classes, or we will achive some Model that we cannot trust.

### Randomly data shuffling

One simple way to get our data equally split between classes is shuffling it. Shuffling data is a good practice, not only to make sure the data is split equally, but also make our learning algorithm work properly in other places such as stochastic gradient descent or something (you will know about it in later posts).

Fortunately, **scikit-learn** library comes bundled with *train_test_split* method, which help us split the original data into training data and test data with a specific ratio, and of course, it makes sure that our data is well shuffled.

So let's import *train_set_split* method and use it to split our data:

{% highlight python %}
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
{% endhighlight %}

You can specify the *random_state* parameter to get a specific random sample. Here I omitted it so that I got a different set of data each time I call the *train_test_split* method.

Let's go ahead and make sure our data is now split equally between classes:

{% highlight python %}
len(np.where(y_test==0)[0])
32

len(np.where(y_test==1)[0])
28
{% endhighlight %}

It's not necessary that the number of data of each class is exactly the same as the other's. So in my case, \\(33:27\\) is a reasonable ratio to me.If you find your data is not okay (let's say \\(35:25\\)), then just re-run the method to get a better split. Simple enough, right?

Next, let's train the Model using our new dataset:

{% highlight python %}
clf.fit(X_train, y_train)
{% endhighlight %}

Check the result again:

{% highlight python %}
clf.score(X_train, y_train)
0.9357142857142857

clf.score(X_test, y_test)
0.9166666666666663
{% endhighlight %}

Obviously, that's just the result that we longed for. You may now realize that, it's not just the algorithm which is important, the data itself does matter. With our dataset properly prepared, we can obtain a better result without tuning any learning parameters.

I'm now so excited. Let's do it one more time:

{% highlight python %}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf.fit(X_train, y_train)

clf.score(X_train, y_train)
0.94285714285714284

clf.score(X_test, y_test)
0.84999999999999999
{% endhighlight %}

The result varied a little bit. But it does make me concern. Is there any chance that we missed something? Because we randomly split our data, what if we have been lucky to have a "good" split? Obviously, we cannot tell anything unless we have some way to cover all the cases, which means that all the data was split into test data at least once.

Well, we will do that with a *for* loop. Since we are splitting with the ratio 7:3, we will call the *train_test_split* method four times in order to cover all the cases:

{% highlight python %}
scores = []
for i in range(4):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  clf.fit(X_train, y_train)
  scores.append(clf.score(X_test, y_test))

print(scores)

[0.93333333333333335, 0.96666666666666667, 0.83333333333333337, 0.98333333333333328]
{% endhighlight %}

Doing this way, we can have a better visualization of testing accuracy over the whole testing data. Obviously, the results varied a little, but that was something we could predict. And since there's no significantly difference between those numbers, we can now feel relieved a little, right?

Here comes another question. Will the *train_test_split* perform well with multi-class classification problem? Let's say now we have five classes, let's see how well it can do:

{% highlight python %}
y = np.append(np.zeros(40), np.ones(40))
y = np.append(y,2 * np.ones(40))
y = np.append(y,3 * np.ones(40))
y = np.append(y,4 * np.ones(40))
{% endhighlight %}

Here we got five classes from \\(0\\) to \\(4\\), each class has 40 examples. Now we will use the *train_test_split* to split our data. Let's see if it can split equally between five classes:

{% highlight python %}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
len(np.where(y_test==0)[0])
11

len(np.where(y_test==1)[0])
5

len(np.where(y_test==2)[0])
20

len(np.where(y_test==3)[0])
8

len(np.where(y_test==4)[0])
16
{% endhighlight %}

You might not expect it, but the performance was poor. Now we may wonder it the **scikit-learn** provides us some other cool module instead. Yeah, it does!

### StratifiedKFold and cross_val_score

**scikit-learn** library provides us a module called **StratifiedKFold**. As its name is self-explained, it will divide the original data into *f* folds, make sure that each fold contains the same amount of data from all classes.

{% highlight python %}
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
{% endhighlight %}

Let's see how our data is split using StratifiedKFold. Since we have 200 examples, with 40 examples each class, we will set *n_folds* equal to 5:

{% highlight python %}
k = StratifiedKFold(y, n_folds=5)

for i, j in k:
  print(y[j])

{% endhighlight %}

Here's the first line of the output I got:

{% highlight python %}
[ 0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  2.  2.
  2.  2.  2.  2.  2.  2.  3.  3.  3.  3.  3.  3.  3.  3.  4.  4.  4.  4.
  4.  4.  4.  4.]
{% endhighlight %}

As you can see, using StratifiedKFold we got our data split equally between five classes. With the data well split, we can be sure that our Model will likely learn much better.

**scikit-learn** library also provides us a method called **cross_val_score**. What it does is first, splitting our Model using StratifiedKFold, and second, looping through all *k* folds then compute the corresponding accuracy score, which means that we don't have to write the loop ourselves anymore. So great, right?

All we have to do is to pass our Model, our original \\(X\\), our original \\(y\\), number of folds we want it to split through the *cv* parameter. And it does all the rest for us!

Note that before that, we must restore our \\(y\\) vector, since I used it above for demonstration. My bad, sorry!

{% highlight python %}
y = np.append(np.zeros(100), np.ones(100))

scores = cross_val_score(clf, X, y, cv=4)

print(scores)
[ 0.78  0.96   0.98  0.9]
{% endhighlight %}

The result we got here is not much different than the one above, since this is a two-class classification problem, using StratifiedKFold is not likely to improve that much. But it is recommended using StratifiedKFold (through the cross_val_score method), rather than use the train_test_split and then write the loop yourselves.

### Summary

So, today I have told you about **cross validation**. Now you know the right way to split the data for training and testing purposes. You also know the need of shuffling the dataset before splitting. And after all, you know how to efficiently split the data using StratifiedKFold and use cross_val_score method to help us perform testing over the whole dataset.

With that, you will know have a valuable tool to evaluate your trained Model, and can tell whether it tends to overfit or not. In future post, after I show you some more powerful algorithms, we will continue with some techniques to prevent overfitting. Until then, stay updated and I will be back soon! See you!
