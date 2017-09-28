---
title: "Machine Learning Part 10: Linear Support Vector Machine"
header:
  teaser: tutorials/support-vector-machine/maximum-margin.png
categories:
  - Tutorial
tags:
  - machine-learning
  - support vector machine
  - svm
  - classification
  - regularization
  - SVC
  - essential
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hi guys! It's been while since my last tutorial post about Regularization. And today, as I promised, I'm gonna talk about one supervised learning algorithm which took the throne of Neural Network a decade ago. It was fast, agile and outperformed almost the algorithms back in the days. Guys, today I want to tell you about Support Machine Learning, or SVM for short.

Many of you may have heard about the term SVM. For example, if you have experience in Computer Vision, especially using OpenCV to accomplish the task, you may have seen something like this on OpenCV's page in HOG Descriptor section:

> OpenCV provides an Linear SVM Model for People detection

And you will likely come across in other places with the same content, which gives us some proof of the irresistible power of SVM. Despite the fact that there are a great deal of supervised learning algorithms out there nowadays, SVM is still among the mostly applied algorithms. And everytime I face a new Machine Learning problem, the first algorithm I apply is SVM, not only for its performance, but also for its speed and easy-to-implement mechanism, which can give me an overview of the problem as fast as I expect.

Above is a brief introduction about SVM. Now let's go finding the anwser for the question we are longing for: What is SVM?

SVM is a supervised learning algorithm which is mostly used for classification problems. It can perform well no matter our dataset is linear or non-linear distributed. But first, to make it easy to understand, in today's post I'm gonna talk only about how SVM work when dealing with linear data, which can also be called Linear SVM algorithm.

And you may remember that I had made a post about one learning algorithm which can give awesome result when dealing with linear dataset. Yeah, I'm talking about Logistic Regression. So, to have a better understanding about Linear SVM, it's a great idea to recall a little bit about Logistic Regression, and see what they differ from each other. For ones who haven't skimmed through my post about Logistic Regression, you can find it right below:

* [Machine Learning Part 6: Logistic Regression](https://chunml.github.io/ChunML.github.io/tutorial/Logistic-Regression/){:target="_blank"}

When we talk about Logistic Regression, we may all think of the sigmoid function, which we use as the activation function. Below is what a sigmoid function looks like:

$$
h_\theta(X)=\frac{1}{1+e^{-\theta^TX}}
$$

As I told you before, using sigmoid function will ensure that the output will be restricted in the range between \\(0\\) and \\(1\\), which then assigned to either \\(0\\) or \\(1\\) depends on its value and the threshold value. And of course, after getting the predictions with the help of the sigmoid function, we cannot evaluate the Model's performance without a cost function (or loss function). And the cost function we use in Logistic Regression is the cross-entropy function, which can also be called the log-likelihood cost function:

$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}\log(h_\theta(X^{(i)}))+(1-y^{(i)})\log(1-h_\theta(X^{(i)})\right]
$$

So now if I set \\(cost_1(\theta^TX)=-log\left(\frac{1}{1+e^{-\theta^TX}}\right)\\), and \\(cost_0(\theta^TX)=-log\left(1-\frac{1}{1+e^{-\theta^TX}}\right)\\) in case of \\(\theta^TX>=0\\) and \\(\theta^TX<0\\) respectively, we can re-write the cost function in a simple form like below:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}cost_1(\theta^TX)+(1-y^{(i)})cost_0(\theta^TX)\right]
$$

Not let's consider the graph of each seperate part which I divided above:

* \\(cost_1(\theta^TX)\\)

![cost_1](/images/tutorials/support-vector-machine/cost_1.png)

As you can see, the \\(cost_1(\theta^TX)\\) term will be very large when \\(\theta^TX\\) is close to zero, and decrease toward zero as the value of \\(\theta^TX\\) increases. What does this mean? Before answering that question, let's consider the other one:

* \\(cost_0(\theta^TX)\\)

![cost_0](/images/tutorials/support-vector-machine/cost_0.png)

Similar to the \\(cost_1(\theta^TX)\\) above, the \\(cost_0(\theta^TX)\\) term will be extremely large when \\(\theta^TX\\) is close to zero, but this time decrease toward zero as \\(\theta^TX\\) goes toward to the left.

The two terms above were divided from our cost function, which means that their values will be accumulated to the cost function. And our target is to minimize the cost function, you remember that? So, the smaller the two terms are, the smaller the cost function becomes. The smaller the cost function is, the closer our Predictions are comparing to the Label \\(y\\).

Now, let's consider the \\(cost_1(\theta^TX)\\) term. We compute this term only when the corresponding label \\(y=1\\). As we saw in the graph above, when \\(\theta^TX\approx0\\), \\(cost_1(\theta^TX)\\) becomes very large. That is because we now have the probability that our Model predict the label \\(y=1\\) is very small, and may be even worse if it predict the label to be \\(0\\). As the result, the cost function will become large as a penalty. In contrast, if \\(\theta^TX\\) is much greater than \\(0\\), then the probability that \\(y=1\\) will be higher. And as the probability becomes nearly \\(1\\), we will have a nearly \\(0\\) cost value.

You can explain the \\(cost_0(\theta^TX)\\) term in the same way. As a conclusion, we will have a result like this:

$$
y_{predict} = \cases{ 1 & \text{if } \theta^TX \ge 0 \cr 0 & \text{if } \theta^TX \lt 0}
$$

The conclusion above is something which I had shown you in the end of the Logistic Regression tutorial, right? This time I just want to make it more clear if we explain it with considering the effect toward the cost function. Now let's move to the case of Linear SVM.

### Linear SVM's Cost Function

After doing some revision on Logistic Regression. Let's see what the cost function of Linear SVM looks like. First, let me re-write the cost function above, with the use of the two \\(cost_1(\theta^TX)\\) and \\(cost_0(\theta^TX)\\) terms, but this time without omitting the regularization term:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}cost_1(\theta^TX)+(1-y^{(i)})cost_0(\theta^TX)\right]+\frac{\lambda}{2m}\sum^m_{j=1}\theta_j^2
$$

To make it even simpler, I will omit the \\(\frac{1}{m}\\) factor. It may affects the value of our cost function, but it doesn't affect the way our algorithm works, since we are just eliminating a constant from a function.

$$
J(\theta)=\sum_{i=1}^m\left[y^{(i)}cost_1(\theta^TX)+(1-y^{(i)})cost_0(\theta^TX)\right]+\frac{\lambda}{2}\sum^m_{j=1}\theta_j^2 
$$

Now, the new cost function looks like above. We can see that it has the form of: \\(\mathbf{A}+\lambda\mathbf{B}\\). Let's talk a little bit about the \\(\lambda\\) term, as we call it the weight of regularization, which control how much we want to regularize our parameters. If it's large, then our parameters will become much smaller and vice versa. Skipped my previous post? You can find it right below:

* [Machine Learning Part 9: Regularization](https://chunml.github.io/ChunML.github.io/tutorial/Regularization/)

Now how about saying we want to put more weight on the actual cost value? The answer may be: just decrease \\(\lambda\\). It's a little bit confusing to someone, so instead of using the form of \\(\mathbf{A}+\lambda\mathbf{B}\\), many people prefer the \\(\mathbf{C}\mathrm{A}+\mathrm{B}\\) form. So now we can say, if we want to emphasize on the actual cost value, we can do it by increasing \\(\mathbf{C}\\). And that way of expression is also the standard which the scikit-learn library are using. For example, here's the full description when initializing Logistic Regression I grabbed on scikit-learn's homepage:

![sklearn_reg_c](/images/tutorials/support-vector-machine/sklearn_reg_c.png)

With that approach, let's re-write our cost function again, using the inverse of regularization weight \\(\mathbf{C}\\) instead of \\(\lambda\\), here's what we will have:

$$
J(\theta)=\mathbf{C}\sum_{i=1}^m\left[y^{(i)}(-\log(h_\theta(X^{(i)})))+(1-y^{(i)})(-\log(1-h_\theta(X^{(i)}))\right]+\frac{1}{2}\sum^m_{j=1}\theta_j^2
$$

And as I told you above about deciding the value of the Predictions by considering its effect on the value of our cost function, which I can also say that the way we compute our cost function, or the two \\(cost_1(\theta^TX)\\) and \\(cost_0(\theta^TX)\\) terms will affect the way our Model predicts the output. And here's what Linear SVM differs from Logistic Regression. We will modify the two cost terms a little bit, to have the new graphs like below:

* Linear SVM's \\(cost_1(\theta^TX)\\) graph

![cost_1_svm](/images/tutorials/support-vector-machine/cost_1_svm.png)

* Linear SVM's \\(cost_0(\theta^TX)\\) graph

![cost_0_svm](/images/tutorials/support-vector-machine/cost_0_svm.png)

Obviously, as you can see, the two cost terms of Linear SVM looks different from what we saw in Logistic Regression. It the difference in how we define the cost terms in Linear SVM makes it predict in a different way. Telling you about this now may makes you feel confusing a little. But in fact, many cannot tell the difference between Linear SVM and Logistic Regression, since they seem to work the same way. So before I talk further, I think it's good to notice the difference right this time, so you won't make any unexpected assumption.

### Predictions of Linear SVM

So, as you see from the graphs of the Linear SVM's cost terms above. They look pretty much like what we saw in Logistic Regression except two things. First: instead of the non-linear graph which we obtained by the logarithmic function, now we have a new graph with two parts, one part which the cost values are \\(0\\), and the other part which values are not \\(0\\) is now linear. The second thing that Linear SVM differs from Logistic Regression is, the constraint to decide the value of the Prediction is now a little bit harder. Linear SVM requires a **safety margin** when deciding the Prediction, which we can express like below:

$$
y_{predict} = \cases{ 1 & \text{if } \theta^TX \ge 1 \cr 0 & \text{if } \theta^TX \lt -1}
$$

Intuitively, this safety margin is the reason why SVM is called the Maximum Margin Classifier, which I told you earlier in this post. What the algorithms does is to find a decision boundary which can obtain the maximum margins from the nearest point of each class. We will have a better visualization of **maximum margin** right below, in the Implementation section. 

So, that's all about Linear SVM. As you can see, if you have already known about Logistic Regression, it's pretty easy to understand Linear SVM since they have some similar behavior in between. And the main point which drives Linear SVM apart from Logistic Regression is how we define the cost terms in Linear SVM, which then affects the way it decides the value of our Predictions. And now, after reading through a great deal of "lecture", let's jump into the Implementation section!

### Implementation

So, we finally come to the Implementation of Linear SVM. And just like Logistic Regression and Decision Tree, scikit-learn library provides us a well pre-implemented Linear SVM. All we have to do is... just use it!

You may now became very familiar with scikit-learn library as well as some Python codes we used for data initialization or graph drawing, so I won't talk about those so much this time. And now, let's get our hands dirty, by import all the necessary stuff we're gonna use:

{% highlight python %} 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import numpy as np
{% endhighlight %}

Next, we will create our data using *make_classification* method:

{% highlight python %} 
X, y = make_classification(n_samples=60, n_informative=2, n_features=2, n_redundant=0, random_state=94)
{% endhighlight %}

To make it simple, this time we just created a dataset with two features and two classes only. Next, let's create our Linear SVM model, and train it using the data created above:

{% highlight python %} 
clf = SVC(kernel='linear')
clf.fit(X, y)
{% endhighlight %}

You may wonder what the *kernel* parameter means. But I will talk about it more in the later post, so now just implement as I did. We set its value "linear" so that scikit-learn knows we want to create a Linear SVM Model.

Next, we will draw the decision boundary which seperates the points of two classes for better visualization of the Model's performance. To help you recall a little bit, the decision boundary in Logistic Regression seperates the coordinate plane into two parts like below (in case we have a dataset of two features and two classes):

$$
y_{predict} = \cases{ 1 & \text{if } \theta_0+\theta_1X_1+\theta_2X_2 \ge0 \cr 0 & \text{if } \theta_0+\theta_1X_1+\theta_2X_2 \lt0}
$$

And through the *coef_* and *intercept_* attributes of the trained Model, we can use them to draw the decision boundary, just like what we did with Logistic Regression.

{% highlight python %} 
xx = np.linspace(x1_min, x1_max)
w = clf.coef_[0]
a = -w[0] / w[1]
yy= a * xx - (clf.intercept_[0]) / w[1]
{% endhighlight %}

And remember that Linear SVM is different from Logistic Regression by the way it defines the cost terms, which then affects the way it decides the value of our Predictions. Concretely, SVM will tend to keep a safety margin when making Predictions, so we're gonna compute the upper boundary and the lower boundary to help visualize the term **maximum margin** better:

{% highlight python %} 
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + margin/w[1]
yy_up = yy - margin/w[1]
{% endhighlight %}

And finally, we now have everything ready, let's go ahead and plot everything on:

{% highlight python %} 
ax = plt.gca()
ax.set_ylim(x2_min, x2_max)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.show()
{% endhighlight %}

And here's the result I received:

![maximum_margin](/images/tutorials/support-vector-machine/maximum-margin.png)

As you could see in the graph above, what Linear SVM did is to find a decision boundary which can keep the maximum margins between the nearest point of each class. And that's the reason why SVM is usually called the **maximum margin classifier**. And through implementing Linear SVM as well as drawing both the upper and lower boundaries, I hope you now have a better visualization of what Linear SVM does.

### Summary

So, thank you for staying with me until the end, in my longest post ever. We have talked about Linear SVM, and of course, a little bit deeper about Logistic Regression, just to help you understand better how the two algorithms differ from each other. I hope after this post, you can both have a deep understand about Logistic Regression, and add Linear SVM, one of the most powerful algorithm to your Machine Learning toolbox. In the next post, I will continue with Support Vector Machine, but there won't be any linear data any more. Next time we will see how SVM can deal with non-linear distributed data, by using something called: the kernel trick. Until then, stay tuned and I will be right back!
