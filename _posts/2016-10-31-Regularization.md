---
title: "Machine Learning Part 9: Regularization"
header:
  teaser: tutorials/regularization/reg_graph.png
categories:
  - Tutorial
tags:
  - machine-learning
  - overfitting
  - regression
  - classification
  - regularization
  - essential
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Welcome back to my 9th tutorial on Machine Learning! I have been kept busy in last weekends, struggling in getting my desktop ready for Deep Learning. You may consider giving that post a look if you are planning to build your own "Monster" too: [Building up my own machine for Deep Learning](https://chunml.github.io/ChunML.github.io/project/Building-Desktop-For-Deep-Learning/){:target="_blank"}.

Of course I got a lot to tell you about things I've been doing with my new desktop. But it will be a little bit selfish of mine and unfair to all of you if I let dust cover my Machine Learning tutorial series. So let's say, I'm back on track. And today, I will talk about **Regularization**, a technique to deal with Overfitting problem. You may still remember that I mentioned eariler in previous posts, Overfitting is a big headache in all Machine Learning problems. Beside Cross Validation that I told you before, **Regularization** is a must-know technique that you will nearly apply in all of your Machine Learning problems. Furthermore, applying **Regularization** is a default setting in all algorithms provided by *scikit-learn* library.

And I'm not going to waste any minute of yours. Let's go straight into the most likely asked question: What is Regularization?

When we hear the word **Regularization** without anything else related to Machine Learning, we all understand that Regularization is the process of regularizing something, or the process in which something is regularized. The problem is: what is exactly *something*. In terms of Machine Learning, we talk about learning algorithms or models, and what is actually inside the algorithms or models? That's the set of parameters. In short, **Regularization** in Machine Learning is the process of regularizing the parameters.

After knowing that Regularization is actually to regularize our parameters, then you may wonder: Why regularizing the parameters help prevent Overfitting? Let's consider the graph that I had prepared for this tutorial. A picture is worth a thousand words, right?

![reg_graph](/images/tutorials/regularization/reg_graph.png)

As you can see in the graph I have just shown you, we got two functions represented by a green curve and a blue curve respectively. Both curve fit those red points so well that we can consider they both incur zero loss. And if you followed all my previous tutorials, you would be able to point out that the green curve is likely to overfit the data. Yeah, you are totally right. But have you ever wondered why the green curve (or any curve which is similar to it) is overfitting the data?

To understand that in a more mathematical way, let's consider the two functions that I used to draw the graph above:

The green curve:

$$
h_1(x)=-x^4+7x^3-5x^2-31x+30
$$

The blue curve:

$$
h_2(x)=\frac{x^4}{5}-\frac{7x^3}{5}+x^2+\frac{31x}{5}-6
$$

Does it sound similar to you? I once told you about one way to improve the performance of Linear Regression model, that is adding polynomial features. You can refer it here: [Underfitting and Overfitting Problems](https://chunml.github.io/ChunML.github.io/tutorial/Underfit-Overfit/){:target="_blank"}. You knew that by adding more features, we will have a more well learned model which can fit our data far better. But everything has its drawback, if we add so many features, we will be purnished with Overfitting. That was what I told you in the earlier tutorial. Not so hard to recall, right?

If you look at each function's equation, you will find that the green curve has larger coefficients, and that's the main caution of Overfitting. As I mentioned before, Overfitting can be interpreted that our model fits the dataset so well, which it seems to memorize the data we showed rather than actually learn from it. Intuitively, having large coefficients can be seen as an evidence of memorizing the data. For example, we got some noises in our training dataset, where the data's magnitude is far difference than the others, those noises will cause our model to put more weight into the coefficient of higher degree, and what we received is a model that overfits our training data!

Some of you may think, if adding so many features causes Overfitting, than why don't we just stop adding features when we got an acceptable model? But think about that this way. If your customer or your boss wants a learned model with \\(95%\\) accuracy, but you can't achieve that result without adding some more features, which results in overfitting the data. What will you do in the next step?

Or think about it in one more other way. You are facing a problem where you are provided with a large dataset, which each of them contains a great deal of features. You don't know which features to drop, and even worse if it turns out that every feature is fairly informative, which means that dropping some features will likely ruin the algorithm's performance. What do you plan to do next?

### The regularization term

Therefore, it's not always a good idea to drop some features just to prevent Overfitting. And as you saw in the example above, it requires further analysis to know whether you can remove some less informative features. So, it's a good practice that you use all features to build your first model in the beginning. And **regularization** comes out as a solution. To make it more clear for you, let's consider our MSE cost function:

$$
J = \frac{1}{2m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)})^2
$$

I once introduced the MSE cost function before in [Logistic Regression](https://chunml.github.io/ChunML.github.io/tutorial/Linear-Regression/){:target="_blank"} tutorial. And as you know, the objective of learning is to minimize that MSE function. It means that our parameters can be updated in anyway, just to lower the MSE value. And as I told you above, the larger our parameters become, the higher chance our Model overfits the data. So the question is: can we not only minimize the cost function, but also restrict the parameters not to become too large? The answer is: we **CAN**, by adding the regularization term like below:

$$
J = \frac{1}{2m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)})^2+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
$$

, where \\(\lambda\\) is a constant to control the value of regularization term, \\(n\\) is the number of the features.

Take a look at our new cost function after adding the regularization term. What the regularization term does is to penalize large parameters. Obviously, minimize the cost function consists of minimizing both terms in the right: the MSE term and the regularization term. So each time some parameter is updated to become significantly large, it will increase the value of the cost function by the regularization term, and as a result, it will be penalized and updated to a small value.

Also note that we only compute the regularization term with the weights only, DON'T include the bias in the regularization term! You may ask why? Well, we can re-write our activation function like below (in case of polynomial function):

$$
h_\theta(X)=\theta_0X^0+\theta_1X+\theta_2X^2+\dots+\theta_nX^n
$$

As you can see, we can think that the bias term goes with the \\(X^0\\) term, which means that it doesn't affect to the form of our function, so include the bias term into the regularization term doesn't make any sense.

With the new added regularization term, obviously we have to make some change to the way we update the parameters too. But it's not a big deal at all, just take all the partial derivatives and we will achieve the result below very easily:

* For weights (\\( \theta_1, \ldots, \theta_n\\))

$$
\frac{\partial}{\partial \theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)}).x_j^{(i)}+\frac{\lambda}{m}\theta_j
$$

* For bias (\\( \theta_0 \\)) it remains unchanged:

$$\frac{\partial}{\partial \theta_0}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)})$$

Next, let's put things together to see how the parameters are updated after adding regularization term:

$$
\begin{bmatrix}\theta_0\\\theta_1\\\vdots\\\theta_n\end{bmatrix}=\begin{bmatrix}\theta_0\\\theta_1(1-\alpha\frac{\lambda}{m})\\\vdots\\\theta_n(1-\alpha\frac{\lambda}{m})\end{bmatrix}-\frac{\alpha}{m}\begin{bmatrix}\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)})\\\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)}).x_1^{(i)}\\\vdots\\\sum_{i=1}^m(h_\theta^{(i)}(x^{(i)})-y^{(i)}).x_n^{(i)}\end{bmatrix} 
$$

That's it. All we have to change is just adding the factor \\((1-\frac{\lambda}{m})\\) to the parameter when updating. You can prove the result above yourselves as an assignment. It's very easy, but I recommend you do it yourselves to become more familiar with those mathematical terms like the Chain Rule or partial derivatives, which you will use a lot in the next tutorials.

### More about regularization

Above I have shown you about adding the regularization term in our MSE function, and how to apply regularization in updating parameters, too. But it doesn't mean that when applying regularization, you always stick to the term I have shown you. In fact, it's just one among many forms of regularization. It's just like the way we have many options for the cost function (or you can call it the loss function), we have MSE function, we have log-likelihood cost function, etc. To a particular problem, there will always be more than one approach, and each one will likely work best in a particular set of conditions.

Below I will introduce to you the two which are mostly used in real world projects:

* L2 Regularization:

$$
J = \frac{1}{2m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)})^2+\frac{\lambda}{2m}\sum_{j=1}^n(\theta_j^2)
$$

This is actually the one I have shown above. It's also easy to remember: L2 means degree \\(2\\) regularization term. Before talking any further, let's consider the other one first:

* L1 Regularization:

$$
J = \frac{1}{2m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)})^2+\frac{\lambda}{m}\sum_{j=1}^n|\theta_j|
$$

Another form of regularization, called the L1 Regularization, looks like above. As you can see, instead of computing mean value of squares of the parameters as L2 Regularization does, what L1 Regularization does is to compute the mean magnitude of the parameters. Also note that in L1 Regularization term, we multiply the sum with a fraction of \\(\frac{\lambda}{m}\\), not \\(\frac{\lambda}{2m}\\). Remember that the term \\(\frac{\lambda}{2m}\\) helps vanish the \\(2\\) factor in the derivative of polynomial of degree 2. So I hope you won't be confused which one to use.

Looking at two cost functions above, adding L1 Regularization term or L2 Regularization term have nearly the same effect, that is to penalize large parameters. As a result, we end up with a learned model with all parameters being kept small, so that our model won't depend on some particular parameters, thus less likely to overfit.

To understand how the two terms differ from each other, let's see how they affect the parameter update process:

* L2 Regularization:

$$
\theta_j = \theta_j-\frac{\alpha}{m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)}).X_j^{(i)}-\alpha\frac{\lambda}{m}\theta_j=(1-\alpha\frac{\lambda}{m})\theta_j-\frac{\alpha}{m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)}).X_j^{(i)}
$$

To make it simple, I just computed for one parameter \\(\theta_j\\). As you can see, when applying regularization, we have the new \\((1-\alpha\frac{\lambda}{m})\\) factor. What does it affect the update for \\(\theta_j\\) anyway? After each learning iteration, \\(\theta_j\\) decreases by an amount of \\(\alpha\frac{\lambda}{m}\\), which means that L2 Regularization tends to shink it by an amount proportional to \\(\theta_j\\). The larger \\(\theta_j\\) becomes, the more it will shrink. Now, what it's like in the case of L1 Regularization:

* L1 Regularization:

$$
\theta_j = \theta_j-\alpha\frac{\lambda}{m}sign(\theta_j)-\frac{\alpha}{m}\sum_{i=1}^m(h_\theta(X^{(i)})-y^{(i)}).X_j^{(i)}
$$

When we use L1 Regularization, our parameters shrink in a different way. Because \\(sign(\theta_j)\\) can only be either \\(-1\\) or \\(1\\), \\(\theta_j\\) now shrinks by a constant amount, and it tends to move toward zero. This makes the update process different from what we saw in L2 Regularization. Therefore, we can easily see that L1 Regularization tends to penalize small parameters more than L2 Regularization does. In short:

* If \\(\theta_j\\) is large, L2 Regularization shinks \\(\theta_j\\) more than L1 Regularization does.
* If \\(\theta_j\\) is small, L1 Regularization shinks \\(\theta_j\\) more than L2 Regularization does.

### Last, but not least...

Above I have just shown you two mostly applied forms of regularization, and how each of them affect the learning process. The last thing I want to tell you in this post is about the constant \\(\lambda\\). As I mentioned earlier, \\(\lambda\\) controls how much we want to regularize our parameters. But what is really behind this? Let's consider the two scenarios below:

* \\(\lambda\\) is very small (nearly zero):

When \\(\lambda\\) is nearly zero, then the regularization term will become nearly zero. As a result, the cost function mostly depends on the MSE term just like before applying regularization. Or we can say that, when \\(\lambda\\) is nearly zero, the regularization term won't have any significant effect on shrinking the parameters. Therefore the model is more likely to overfit.

* \\(\lambda\\) is very large:

Conversely, let's consider the case where \\(\lambda\\) becomes extremely large. This time we put much weight in the regularization term. And as a result, the parameters will shrink to a very small values. That approach, however, brings a real problem, that is, rather than preventing Overfitting, our parameters now become so small so that it can't even fit the training data well, or we can say that: applying so much regularization cause our model to underfit the dataset.

Through the two scenarios above, you can see that choosing the right value for \\(\lambda\\) is not an easy task. But with the right choice of \\(\lambda\\), the regularization term can have a significant effect on the learning process, which results in a model which both fits the dataset very well, but not likely to overfit.

### Summary

In today's post, we have talked about **Regularization**, an important technique applied in every Machine Learning model in the real world to deal with the Overfitting problem. I hope after this tutorial, you can have a deeper understanding about what actually causes Overfitting, and the right way to deal with that headache. In the next post, I will continue with one last supervised learning algorithm, the one that I should have showed you in this post instead. But I soon realized that with the lack of knowledge about Regularization, it will be pretty hard to fully understand that algorithm. I hope you not blame me for this. So stay updated and I will be right back!
