---
title: "Machine Learning Part 5: Underfitting and Overfitting Problems"
header:
  teaser: tutorials/underfit-overfit/poly_2_test.jpg
categories:
  - Tutorial
tags:
  - machine-learning
  - linear-regression
  - implementation
  - underfit
  - overfit
  - problems
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Here we are again, in the fifth post of Machine Learning tutorial series. Today I will talk about two common problems you may face in Machine Learning: Underfitting and Overfitting. Wait! There is something wrong, isn't it? - You may wonder...

Of course I remember promising you in the previous post, that today I will dig deeper into Linear Regression, and together we will do some coding. Actually, I intended to name today's post "Implementing Linear Regression" or something, but I soon realized that it would be inappropriate. Good news is today's post will mainly focus on implementation of Linear Regression, and what we can do to improve the quality of the Model. By doing that, I am actually leading you to go into some concept which is more general, and can be applied not only in Linear Regression, but every spot which Machine Learning takes place.

That is enough of talking. First, let's get our hands dirty. If you went through my previous post, you would now have everything set up. But if you didn't, you might want to take a look at it here: [Setting Up Python Environment](https://chunml.github.io/ChunML.github.io/tutorial/Setting-Up-Python-Environment-For-Computer-Vision-And-Machine-Learning/){:target="_blank"}.

### Implementing Linear Regression

Open Terminal and go into Python console mode:

{% highlight Bash shell scripts %} 
python
{% endhighlight %}

Now let's import *sklearn* module for Linear Regression. *sklearn* is a shortname of **scikit-learn**, a great Python library for Machine Learning.

{% highlight python %} 
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
{% endhighlight %}

We are not using only *LinearRegression*. We will work with arrays, so here I also imported *numpy* for dealing with arrays. We will also draw some graphs to visualize the data, so that's why I imported *pyplot*, a great module for graph drawing.

Remember the data I used in the previous post on Linear Regression? I will show it right below for you:

| X       | y           |
| ------------- |-------------| 
| 1      | 7 | 
| 2      | 8 |
| 3      | 7 |
| 4      | 13 |
| 5      | 16 |
| 6      | 15 |
| 7      | 19 |
| 8      | 23 |
| 9      | 18 |
| 10      | 21 |

Now let's use that to prepare our training data:

{% highlight python %} 
X = np.arange(1, 11).reshape(10, 1)
y = np.array([7, 8, 7, 13, 16, 15, 19, 23, 18, 21]).reshape(10, 1)
{% endhighlight %}

Nearly every Machine Learning library requires data to be formatted in the way which each row is one training example (or testing example), and each column represents one feature's data. So we have to reshape our data accordingly.

Now let's plot our training data. You will receive the same figure with the one in the previous post:

{% highlight python %} 
plt.plot(X, y, 'ro')
plt.show()
{% endhighlight %}

![Training_data](/images/tutorials/linear-regression/1.jpg)

Next, let's initialize the Linear Regression model:

{% highlight python %} 
model = LinearRegression()
{% endhighlight %}

Then we will train our Model, using the training data above. You can do that by simply calling the **fit** function, which takes feature matrix \\(X\\) and label vector \\(y\\) as parameters:

{% highlight python %} 
model.fit(X, y)
{% endhighlight %}

Our training data is quite simple, so the learning process finished so fast as if it never happened. All the change during training (like weights and bias), was stored in the model object. Let's see what we got:

{% highlight python %} 
model.coef_
array([[ 1.77575758]])

model.intercept_
array([ 4.93333333])
{% endhighlight %}

Obviously, you can get more information through other attributes of model object, but now we will only focus on *coef_*, which stores the weight parameter, and *intercept_*, which stores the bias parameter.

Next, let's compute the prediction vector *a*, using the obtained weight and bias:

{% highlight python %} 
a = model.coef_ * X + model.intercept_
{% endhighlight %}

Now let's draw all \\(X\\), \\(y\\) and \\(a\\) on the same plot. Here we got a straight line, which fit the data better than what we did before (which is easy to understand, since we only went through 4 iterations).

{% highlight python %} 
plt.plot(X, y, 'ro', X, a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
{% endhighlight %}

![poly_1](/images/tutorials/underfit-overfit/poly_1.jpg)

So simple, right? Just a few lines of code, we have just prepared our training data, trained our Model, and visualized the result we got! Yeah, **scikit-learn** helps us do all the heavy things. In later posts, you will see that it can even handle more complicated jobs.

### Improving the performance of Linear Regression

Obviously, we can see that the straight line above fits pretty well, but not good enough. And we need a more suitable approach. But first, let's evaluate how well the Model is performing numerically, by computing the accuracy over the training data:

{% highlight python %} 
print(model.score(X, y))
0.84988070842366825
{% endhighlight %}

We cannot always evaluate something just by seeing it, right? We need something which is more concrete, yeah, a number. By looking at numbers, we will have a better look, and easily compare different things. **scikit-learn** provides us the *score* function, whose parameters are similar to the *fit* function.

And you can see that, our Model now has the accuracy of 85% over the training data. Commonly,  we demand a higher accuracy, let's say 90% or 95%. So by looking at the current accuracy, we can tell that our Model is not performing as we are expecting. So let's think about an improvement. But how can we do that?

Remember I told you about Features in the first [Post](https://chunml.github.io/ChunML.github.io/tutorial/Machine-Learning-Definition/){:target="_blank"}? Features are something we use to distinguish one object from others. So obviously, if we have more Features, then we will likely have a better fit model, since it can receive more necessary information for training. But how we can acquire more Features?

#### Polynomial Features
The easiest way to add more Features, is to computing *polynomial features* from the provided features. It means that if we have \\(X\\), then we can use \\(X^2\\), \\(X^3\\), etc as additional features. So let's use this approach and see if we can improve the current Model. First, we have to modify our \\(X\\) matrix by adding \\(X^2\\):

{% highlight python %} 
X = np_c[X, X**2]
X
array([[   1,    1],
       [   2,    4],
       [   3,    9],
       [   4,   16],
       [   5,   25],
       [   6,   36],
       [   7,   49],
       [   8,   64],
       [   9,   81],
       [  10,  100]])
{% endhighlight %}

Similar to previous step, let's train our new Model, then compute the prediction vector \\(a\\):

{% highlight python %} 
model.fit(X, y)
x = np.arange(1, 11, 0.1)
x = np.c_[x, x**2]
a = np.dot(X, model.coef_.transpose()) + model.intercept_
{% endhighlight %}

Mathematically, we will now have \\(a=\theta_0 + \theta_1X + \theta_2X^2\\). Note that now we have more complicated matrix *X*, so we will have to use the *dot* function. An error will occur if we just use the multiply operator like above. I also created a new \\(x\\) variable, which ranges from 1 to 10, but with 0.1 step. Use the new \\(x\\) to compute \\(a\\) will result in a smoother graph of \\(a\\), since \\(a\\) is no longer a straight line anymore.

Now let's plot things out and see what we got with new feature matrix:

{% highlight python %} 
plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
plt.show()
{% endhighlight %}

![poly_2](/images/tutorials/underfit-overfit/poly_2.jpg)

As you can see, now we obtain a curved line, which seems to fit our training data much better. To be more concrete, let's use the *score* function:

{% highlight python %} 
model.score(X, y)
0.87215506914951546
{% endhighlight %}

You see that? Now we got a new accuracy of 87%, which is a huge improvement right? At this point, you may think that we can improve it a lot more by continuing to add more polynomial features to it. Well, don't guess. Let's just do it. This time we will add up to degree 9.

{% highlight python %} 
X = np.arange(1, 11)
X = np.c_[X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9]
x = np.arange(1, 11, 0.1)
x = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]

model.fit(X, y)
a = np.dot(x, model.coef_.transpose()) + model.intercept_

plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
{% endhighlight %}

![poly_9](/images/tutorials/underfit-overfit/poly_9.jpg)

Now we just obtained a new curve which fit our training data perfectly. Let's use the *score* function again to get an exact number:

{% highlight python %} 
model.score(X, y)
0.99992550472904074
{% endhighlight %}

Wow, let's see what we have here, an accuracy of 100%. This is real magic, you may think.

But that is just where the tragic begins...

### OVERFITTING & UNDERFITTING

Now let imagine our data has total 15 examples, and I just showed you the first 10. I will reveal the last 5 examples like below:

| X       | y           |
| ------------- |-------------| 
| 11      | 24 | 
| 12      | 23 |
| 13      | 22 |
| 14      | 26 |
| 15      | 22 |

So actually our data will look like this:

{% highlight python %} 
X = np.arange(1, 16)
y = np.append(y, [24, 23, 22, 26, 22])

plt.plot(X, y, 'ro')
plt.show()
{% endhighlight %}

![full_data](/images/tutorials/underfit-overfit/full_data.jpg)

Let's see what happens if we use the Model obtained from degree 9 polynomial features:

{% highlight python %} 
plt.plot(X, y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
{% endhighlight %}

![poly_9_overfit](/images/tutorials/underfit-overfit/poly_9_overfit.jpg)

Do you see what I am seeing? What a tragic! It doesn't seem to fit the new data at all! We don't even feel the need of computing the accuracy on the new data! So what the hell this is all about?

As I told you before, in the first post, that we only provided a fixed set of training data, and the Model will have to deal with new data which it has never seen before. New data, which may vary in unpredictable way in real life, penalized our trained Model this time! In Machine Learning term, we call it **OVERFITTING** problem (or High Variance). Overfitting, as the name is self-explained itself, means that the Model fits the data very well when we prodived a set of data containing a lot of features. We can see that the Model tends to memorize the data, rather than to learn from it, which makes it unable to predict the new data.

In contrast, what will happen if we use just one feature like we did in the beginning (or we can say that we provided a set of data which is poorly informative)? You have already seen that it resulted in a very low accuracy, which is not what we expected, either. We call this problem **UNDERFITTING** (or High Bias).

Overfitting & Underfitting, in both cases, are something that we try to avoid. And you will mostly face these problems all the time you work with Machine Learning. Of course, there are many ways to deal with them, but I will leave all the details for a future post. This time I will tell you the simplest way, which can be seen as a "must-do" in the very first step of any Machine Learning problem.

### Splitting dataset for training and testing

The first thing to do to prevent the problems above, is always splitting the dataset into training data and testing data. Never just count on the accuracy on training data! Why? Because even though we obtained a high accuracy, it does not mean that our Model is doing a good job. Conversely, we need to watch out for *Overfitting* problem. By splitting our dataset into two seperate parts, we will use one part for training, and the other for evaluating the trained Model. Because we evaluate the performance on a separate data, we can know if our Model can work well with new data that it has never seen. And we can somehow tell whether our Model has *Overfitting* problem or not. 

*Underfitting*, on the other hand, can easily be discovered just by looking at the accuracy over the training data, because if our Model has *Underfitting* problem, then it will perform poorly on both dataset.

Finally, we will pick the Model which has the highest accuracy on the testing data.

With the approach I have shown you, let's decide which Model to choose among three models above. We will use first ten examples as training data, and the last five examples for testing.

#### Model 1
{% highlight python %} 
X = np.arange(1, 16).reshape(15, 1)
model.fit(X[:10], y[:10])
model.score(X[10:], y[10:])
-12.653810835629017

a = np.dot(X, model.coef_.transpose()) + model.intercept_
plt.plot(X, y, 'ro', X, a)
plt.show()
{% endhighlight %}

![poly_1_test](/images/tutorials/underfit-overfit/poly_1_test.jpg)

#### Model 2
{% highlight python %} 
X = np.arange(1, 16).reshape(15, 1)
X = np.c_[X, X**2]
x = np.arange(1, 16, 0.1)
x = np.c_[x, x**2]
model.fit(X[:10], y[:10])
model.score(X[10:], y[10:])
-0.51814820280729812

a = np.dot(x, model.coef_.transpose()) + model.intercept_
plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
{% endhighlight %}

![poly_2_test](/images/tutorials/underfit-overfit/poly_2_test.jpg)

#### Model 3
{% highlight python %} 
X = np.arange(1, 16).reshape(15, 1)
X = np.c_[X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9]
x = np.arange(1, 16, 0.1)
x = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]
model.fit(X[:10], y[:10])
model.score(X[10:], y[10:])
-384608869887696.81

a = np.dot(x, model.coef_.transpose()) + model.intercept_
plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
{% endhighlight %}

![poly_9_overfit](/images/tutorials/underfit-overfit/poly_9_overfit.jpg)

As you can see, both by visualizing and by looking at the accuracy. Our first model is too simple, which didn't fit our data well. This is an example of *Underfitting* problem. In contrast, our third model is way too complicated, which performed very well on training data, but failed to fit the testing data. This is what we called *Overfitting* problem.

The second model may not fit as well as the third model, but it is the one that actually learned, which results in good performance over the testing data. And we can somehow say that, it will also predict well with any other data which it has never seen during training.

### Conclusion

So today, through implementing Linear Regression, I led you through the most common problems you may face when working with Machine Learning, which are *Underfitting* and *Overfitting*. I also showed you the easiest way to avoid those problems, which is always splitting the dataset into two parts: one for training purpose, and one for testing.

Hope you find today's post helpful and you now put a further step into Machine Learning world, I think. There is no stopping as you has gone this far. In the next post, I will continue with **Logistic Regression**, which is an extremely important algorithm that you must understand, because it is the key which leads you to the most powerful learning technique nowadays: Neural Network. So stay updated, and I will be with you soon. See you!
