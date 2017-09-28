---
title: "Machine Learning Part 6: Logistic Regression"
header:
  teaser: tutorials/logistic-regression/graph_3.png
categories:
  - Tutorial
tags:
  - machine-learning
  - logistic-regression
  - classification
  - essential
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hello there, I am back with you today. In the 6th post on Machine Learning tutorials series, I will tell you about Logistic Regression, a very important and must-know algorithm. Before I go any further, there is one thing I want you to be clear at first. The algorithm's name, Logistic Regression, is somehow confusing a little bit, but we are not dealing with a regression problem, but a **classification problem**.

But what is the difference between regression problems and classification problems? - You may ask. I am not a statistical expert, and you may not want any detailed academic explanations, so I will make it simply like below.

A regression problem is where you have a **continuous** dataset of label \\(y\\), and your goal is to make predictions for unlabeled data. Because \\(y\\) is continuous, it can take any value between a specific range. Concretely, if our problem's output range is \\(1 ~ 10\\), then \\(y\\) can be any value between that range, such as \\(1.2, 2.5\\) or even \\(4.23424324\\), etc. A regression problem's graph will mostly look like the graph I showed you in [Linear Regression](https://chunml.github.io/ChunML.github.io/tutorial/Linear-Regression/){:target="_blank"} tutorial:

![Update 4](/images/tutorials/linear-regression/7.jpg)

A classification problem, as you may guess, is where our labels \\(y\\) can only take a particular value, or we can say that \\(y\\) is a discrete dataset. For example, if we want to solve a spam mail detecting problem, obviously we will have only two labels which is either *spam* or *non-spam*. Another example, you are a magician who is trying to guess the suit of a randomly picked card, so you only have four choices among Hearts, Diamonds, Clubs and Spades, right? So we are now dealing with a different problem from the one we did before. Our graph will now look like this:

![Graph_1](/images/tutorials/logistic-regression/graph_1.png)

Okay, so I hope you know can see the difference between a regression problem and a classification problem. Let's see how we will do to deal with a classification problem. To make it easily for you to understand, it is better that we should now consider a two-class classification problem like above.

### Activation Function

You still remember how Machine Learning actually works? It is important for any Machine Learning algorithms to have a way to compute the Predictions from the feature data \\(X\\) which we called Activation Functions. In the case of Linear Regression, the activation function is just as simple as below:

$$
h_\theta(X) = \theta_0 + \theta_1X_1 + \theta_2X_2 + \dots + \theta_nX_n
$$

As you can see, the function above is suitable for regression problems, as it doesn't apply any restriction onto the output. Obviously, this function won't work well with our classification problem. But we can also see that, Logistic Regression is somehow similar to Linear Regression (that is why there is *Regression* in its name), so we may be able to solve the Logistic Regression if we somehow find a way to restrict the output value of the function above.

One simple way we can think about is, to use a threshold value. The idea is like below:

$$
h_\theta(X^{(i)}) = \cases{ 1 & \text{if } h_\theta(X^{(i)}) \ge 0.5 \cr 0 & \text{if } h_\theta(X^{(i)}) \lt 0.5}
$$

And we will obtain a graph like this:

![Graph_2](/images/tutorials/logistic-regression/graph_2.png)

Seems OK, huh? At least we can ensure that the activation function now only outputs \\(0\\) or \\(1\\). But in fact, using a threshold directly on regression's activation function is not a good idea at all. The reason is, now we have a sharp change in our graph, which means that the value of \\(y\\) will change immediately in an unpredictable way, which results in a bad Model in the end.

So now we know what to do next. We need an activation function which not only restricts the output value but also contains no sharp change. That shouldn't be a big problem, in fact, we have quite a lot of functions which can satisfy both conditions. And the one which is mostly used is **Sigmoid** function, which has the form as below:

$$
g(z)=\frac{1}{1+e^{-z}}
$$

And the graph of sigmoid function looks like this:

![Graph_3](/images/tutorials/logistic-regression/graph_3.png)

Perfect, right? That is exactly what we need. So how are we supposed to apply this to our case. It's very simple, we just do like below:

$$
z = \theta^TX = \theta_0 + \theta_1X_1 + \theta_2X_2 + \dots + \theta_nX_n
$$  

$$
h_\theta(X) = g(z) = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-\theta^TX}}
$$

Next, I want to talk a little bit about the output of the function about, \\(h_\theta(X)\\). \\(h_\theta(X)\\) is interpreted as the probability that \\(y=1\\) on input \\(X\\), which can be expressed in mathematical terms like this:

$$
h_\theta(X) = P(y=1|x;\theta)
$$

And obviously, since we always have:

$$
P(y=1|x;\theta)+P(y=0|x;\theta)=1
$$

So we can also rewrite the probability that \\(y=0\\) like this:

$$
1 - h_\theta(X) = P(y=0|x;\theta)
$$

**The output of activation function is the probability that \\(y = 1\\).**

Why did I emphasize that? Do you still remember that I left one mystery unrevealed in my very first post about [What is Machine Learning?](https://chunml.github.io/ChunML.github.io/tutorial/Machine-Learning-Definition/){:target="_blank"}. I labeled two classes Dog and Not-a-dog, not Dog and Cat, or Dog and Bird either. And the reason is clear now, I think. In a classification problem, the output is usually interpreted as the probability that the tested object belongs to a particular class. Obviously, if one object is not a Dog, then I would rather say that it is Not-a-dog than say that it is a Cat or it is a Bird, right? Although in real world projects, let's say we have ten classes, then each class will be labeled as a specific name, rather than an obscure name like *Not-something*, but that's OK in case you already had some basic understanding about Machine Learning (I will talk about multiclass classification later). And I thought that it would be better to name two classes Dog and Not-a-dog in the very first example on Machine Learning, so that you would make no misunderstanding about what is actually outputted from a Machine Learning Model. 

So we have done with sigmoid function, the function which we chose as our Activation Function. To recall a little, sigmoid function is a great choice here because it can satisfy both conditions: restricting the output's values and ensuring there is no sharp changes in graph.

### Cost function
After choosing the activation function, let's move to our next target: the Cost Function. As you may remember, the cost function evaluate our Model performance based on the difference between its Predictions and the actual Labels. In the case of Linear Regression, our cost function looks like this:

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)})^2
$$

Can we use the same cost function in Logistic Regression? The answer is: definitely **NO**. You got the right to ask me why. Well, in the learning process, what the computer tries to do is to minimize the cost function. Therefore, our cost function must be some kind of convex functions so that we can find the minimum by using Gradient Descent. In the case of Logistic Regression, if we use the same cost function like above, we will end up with a non-convex function. And for sure, the computer has no way to find the minimum.

So we are likely to find a new form of cost function which can both evaluate our Model's performance and tends to converge to some minimum. I don't want to talk deeper into how they found the appropriate cost function for Logistic Regression because it requires a lot of mathematical explanations. To make it as easily to understand as possible, you can see that our activation function contains an exponential element \\(e^{-z}\\), and one way to linearize that kind of function is to use logarithm. That is why the cost function for Logistic Regression was defined like below, and called the log-likelyhood cost function or the cross-entropy cost function:

$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}\log(h_\theta(X^{(i)})) + (1 - y^{(i)})\log(1 - h_\theta(X^{(i)}))]
$$

Let's talk a little bit about the cost function above. First, note that now we divide the sum by \\(\frac{1}{m}\\), because our new cost function is no longer a quadratic function. And don't forget the **minus** sign, either! It's very easy to explain why there is minus sign there. The logarithm of a number whose value is from \\(0\\) to \\(1\\) is a minus number, so adding a minus sign will make sure our cost function will always greater than or equal to \\(0\\).

Next, you can see that our new defined cost function has two seperate part: \\(y^{(i)}\log(h_\theta(X^{(i)}))\\) and \\((1 - y^{(i)})\log(1 - h_\theta(X^{(i)}))\\). Since the Label \\(y\\) can only be \\(0\\) or \\(1\\), so one of the two terms above will be \\(0\\). So we have a cost function which can cover both two cases: \\(y=0\\) and \\(y=1\\). Furthermore, our new cost function can also accumulate the output errors in each case properly, as explained above.

### Gradient Descent

So I have just talked about the cost function used in Logistic Regression problems. After we have a cost function, we will compute Gradien Descent. Do you still remember what Gradien Descent is? We need to compute Gradient Descent in order to update the parameter \\(\theta\\) (After assuring our cost function is convex, we need a way to go downhill, right?). So, let's compute Gradient Descent. Our new cost function seems very complicate this time, which you may think that it would take a day to compute all its partial derivatives. Don't worry, things are not that bad. In fact, computing the log-likelihood cost function's partial derivatives is very easy, all you have to do is using the *chain rule* which I mentioned before in earlier post. Writing it our here will make this post long and boring, so I leave it for you, lol. Here, I just want to show you the result. And it may make you surprised. Yeah, it looks just like what we had with Linear Regression:

* For weights (\\( \theta_1, \ldots, \theta_n\\))

$$\frac{\partial}{\partial \theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)}).x_j^{(i)}$$

* For bias (\\( \theta_0 \\))

$$\frac{\partial}{\partial \theta_0}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)})$$

As I mentioned above, you can compute the partial derivatives yourselves (I highly recommend you to do so, though). And with a little patience, I'm quite sure that you will have the same result.

So now you know everything you need to know about Logistic Regression: the sigmoid function, the log-likelihood cost function and its gradient descent. Note that not only Linear Regression and Logistic Regreesion, knowing these three terms also help you understand and use any other Machine Learning algorithms as well (even those complicated algorithms such as Neural Network!). 

### Decision Boundary

I will be more than happy if you are still keeping my page open. I really appreciate your attitude and persistence, too. And we will do something more interesting right now: programming!

But before getting our hands dirty. Let's consider the figure below:

![3d](/images/tutorials/logistic-regression/3d.png)

I have just prepared some data. This time we will work with two-feature dataset. In practice, it's normal that our data has a great deal of features (can be up to thousands in case of image data), so working with one-feature data makes it irritating when dealing with some real problems. On the other hand, working with many-feature data when learning (let's say, data which has more than three features) will make you unable to visualize the result. So, it's a good idea to use data with two features (or three features) when learning.

So as you saw in the figure above, our data has two features: \\(X_1\\) and \\(X_2\\) with the label \\(y=0\\) or \\(y=1\\). Since we are dealing with a classification problem, where our label can take a particular value only (in this case \\(0\\) of \\(1\\), we don't need to plot the label \\(y\\). It would be better if we just plot a graph of \\(X_1\\) and \\(X_2\\) only like below:

![2d](/images/tutorials/logistic-regression/2d.png)

So what about \\(y\\)? We can use colors to indicate different values of \\(y\\). Doing this way, we can have a better visualization of our data, and make it possible to visualize three-feature data as well.

With the figure above, we can see that our data is linearly distributed, which means we can seperate them out by a straight line. In a classification problem, that straight line is called **Decision Boundary**. Imagine we have drawn a decision boundary to the graph above, then to every new point, if it lies above the decision boundary, then we will label it with blue triangle. Conversely, if the new point lies below the decision boundary, it will become a red circle. Simple, right? A picture is worth a thousand words!

But what is exactly behind the decision boundary? Just to recall you, the value of the activation function \\(h_\theta(X)\\) is the probability that \\(y=1\\), we will now define a new variable to hold the prediction made by our Model like below:

$$
y^{(i)}_{predict} = \cases{ 1 & \text{if } h_\theta(X^{(i)}) \ge 0.5 \cr 0 & \text{if } h_\theta(X^{(i)}) \lt 0.5}
$$

But the activation function is a sigmoid function, which means that:

$$
\cases{ g(z) \ge 0.5 & \text{if } z \ge 0 \cr g(z) < 0.5 & \text{if } z \lt 0}
$$

So we can have the relation between the prediction \\(y_{predict}\\) and \\(\theta^TX\\) like this:

$$
y_{predict} = \cases{ 1  & \text{if } \theta^TX \ge 0 \cr 0 & \text{if } \theta^TX \lt 0}
$$

Now what's next? How the hell can all of this lead to a decision boundary we need? Well, let's consider the example below:

Let's pick some random parameters for our activation function, like \\(\theta_0 = -1\\), \\(\theta_1 = 2\\) and \\(\theta_2 = 3\\). So we will have \\(z = -1 + 2X_1 + 3X_2\\). Let's compute the prediction \\(y_{predict}\\):

$$
\cases{y = 1 \; \text{if }  -1 + 2X_1 + 3X_2 \ge 0 \Rightarrow 2X_1 + 3X_2 \ge 1 \cr y = 0 \; \text{if }  -1 + 2X_1 + 3X_2 \lt 0 \Rightarrow 2X_1 + 3X_2 \lt 1}
$$

Let's visualize the result above:

![boundary](/images/tutorials/logistic-regression/boundary.png)

As you can see, with the constraint above, we can draw a straight line, in this case: the \\(2X_1 + 3X_2 - 1 = 0\\) line and have it seperate our hyperplane into two parts. The area above the line, which have \\(2X_1 + 3X_2 - 1 \ge 0\\) is where our prediction \\(y_{predict} = 1\\) and the other one is where \\(y_{predict} = 0\\).

So I hope you now understand what decision boundary is. It is a very important concept in classification problems. In the near future, you will see that the decision boundary is not necessarily a straight line. Depending on the algorithm you use, you can achieve a curve decision boundary (as I already shoed you, a curve line is fit the data much more better).

Now, we are ready to code! Let's first import all the necessary modules:

{% highlight python %} 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
{% endhighlight %}

Next, we will create the data to be trained:

{% highlight python %} 
x1_1 = np.linspace(0, 2, 100)
x1_2 = np.linspace(0.1, 2.1, 100)
x2_1 = np.abs(np.random.rand(100))*4
x2_2 = np.abs(np.random.rand(100))*4 + 4

y1 = np.zeros(100)
y2 = np.ones(100)

x = np.ones((200, 2))
x[:, 0] = np.append(x1_1, x1_2)
x[:, 1] = np.append(x2_1, x2_2)

y = np.append(y1, y2)
{% endhighlight %}

If you plot the data above, you will get a graph that looks nearly the same as the one above. Now let's train the Logistic Regression model:

{% highlight python %} 
clf = LogisticRegression()

clf.fit(x, y)
{% endhighlight %}

It should take no longer than one second to complete training. Let's see how well our Model performs on the training dataset:

{% highlight python %} 
clf.score(x, y)

0.98
{% endhighlight %}

Note that your result may vary, since the values of \\(X_2\\) were randomly initialized. I will omit the overfitting problem in this tutorial for simplicity. And similar to Linear Regression, after finishing training, the Logistic Regression object now contains the final parameters in *coef_* and *intercept_* attributes, we will use them to draw our decision boundary:

{% highlight python %}
t = np.linspace(0, 2, 100)

y_pred = (-clf.intercept_ - clf.coef_[0][0]*t) / clf.coef_[0][1]

X1_min, X1_max = X[:, 0].min(), X[:, 0].max()
X2_min, X2_max = X[:, 1].min(), X[:, 1].max()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.plot(t, y_pred, 'g-')
axes = plt.gca()
axes.set_xlabel('X1')
axes.set_ylabel('X2')
axes.set_xlim([X1_min, X1_max])
axes.set_ylim([X2_min, X2_max])

plt.show()
{% endhighlight %}

We will obtain a decision boundary like below, note that your result may be different from mine:

![fit](/images/tutorials/logistic-regression/fit.png)

As you can see, the final decision boundary somehow can seperate the blue triangles and the red circles pretty well. And obviously, we see that there is much room for improvement, but I will leave it for the later post. After typing some codes and visualizing the result, I hope you now know how to implement your code to use the Logistic Regression algorithm.

### Summary

So today, we have talked about Logistic Regression. We talked about the difference between a regression problem and a classification problem. We talked about the activation function, the cost function used in Logistic Regression and how to compute its gradient descent as well. In the next post, I will continue with **Cross Validation**, another very important technique that you must know to deal with Overfitting problem, as you will use that technique nearly in every Machine Learning problem you may face in the future! So stay updated, and I will be back with you soon!

