---
title: "Creating A Language Translation Model Using Sequence To Sequence Learning Approach"
header:
  teaser: projects/sequence-to-sequence/repeated_vector.png
categories:
  - Project
tags:
  - machine-learning
  - deep-learning
  - keras
  - recurrent neural network
  - gpu
  - training
  - RNN
  - LSTM
  - GRU
  - seq2seq
  - translator model
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
Hello guys. It's been quite a long while since my last blog post. It may sound like an excuse, but I've been struggling with finding a new place to move into. And I had to say, it's a real problem for a foreigner to find a reasonable apartment in Japan. Luckily, I somehow managed to find one, and I have just moved in for nearly two weeks. Anyway, the toughest time has gone, and now I can get myself back to work, to bring to you guys new interesting (and maybe boring as usual) blog posts on Deep Learning.

So, in my previous blog post, I told you about how to create a simple text generator by training a Recurrent Neural Network model. What RNNs differ from normal Neural Networks is, instead of computing the output prediction on each input independently, RNNs compute the output of timestep \\(t\\) using not only the input of timestep \\(t\\), but also involving the input of previous timesteps (say, timestep \\(t-1\\), \\(t-2\\), \\(\dots\\)). 

As you already saw in my previous post, inputs are actually sequences of characters, and each output was simply the corresponding input shifted by one character to the right. Obviously, you can see that each pair of input sequence and output sequence has the same length. Then, the network was trained using the famous Harry Potter as training dataset and as a result, the trained model could generate some great J.K. Rowling-style paragraphs. If you haven't read my previous post yet, please take a look at it by the link below (make sure you do before moving on):

* [Creating A Text Generator Using Recurrent Neural Network](https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/){:target="_blank"}

But here comes a big question: What if input sequence and output sequence have different lengths?

You know, there are many types of Machine Learning problems out there where input and output sequences don't necessarily have the exact same length. And in terms of Natural Language Processing (or NLP for short), you are more likely to face problems where their lengths are totally different, not only between each pair of input and output sequence, but also between input sequences themselves! For example, in building a language translation model, each pair of input and output sequence are in different languages, so there's a big chance that they don't have the same length. Moreover, I can bet my life that there is no known language where we can create all sentences with the exact same length! Obviously, that is a really big big problem, because the model I showed you in the last post required all the input and output sequences have the same length. Sounds impossible, huh?

The answer is: NO. Big problems only got smart people attracted, and as a result, solved by not only one, but many solutions. Let's go back to our problem. A lot of attempts were made, each of them has its own advantages and disadvantages when compared to others. And in today's post, I will introduce to you one approach which received great attention from NLP community: The Sequence To Sequence Networks (or seq2seq for short), great work by Ilya Sutskever, Oriol Vinyals, Quoc V. Le from Google.

I will talk briefly about the idea behind seq2seq right below. For ones who want to understand deeply about the state-of-the-art model, please refer to the link to the paper at the end of this post.

At this point, we have already known the problem we must deal with, that we have input and output sequences of different lengths. To make the problem become more concrete, let's take a look at the graph below:

![figure](/images/projects/sequence-to-sequence/figure.png)  
(Image cut from the original paper of Sequence to Sequence Learning with Neural Networks)

As illustrated in the graph above, we have "ABC" as the input sequence, and "WXYZ" as the output sequence. Obviously, the lengths of the two sequences are different. So, how does seq2seq approach to solve that problem? The answer is: they create a model which consists of two seperate recurrent neural networks called **Encoder** and **Decoder** respectively. To make it easy for you, I drew a simple graph below:

![encode_decode](/images/projects/sequence-to-sequence/encode_decode.png)

As the names of the two networks are somehow self-explained, first, it's clear that we can't directly compute the output sequence by using just one network, so we need to use the first network to **encode** the input sequence into some kind of "middle sequence", then the other network will decode that sequence into our desire output sequence. So, what does the "middle sequence" look like? Let's take a look at the next graph below:

![repeated_vec](/images/projects/sequence-to-sequence/repeated_vector.png)

The mystery was revealed! Concretely, what the Encoder actually did is creating a temporary output vector from the input sequence (you can think about that temporary output vector as a sequence with only one timestep). Then, that vector is repeated \\(n\\) times, with \\(n\\) is the length of our desire output sequence. Up to this point, you may get all the rest. Yep, the Decoder network acts exact the same way with the network I talked about in the last post. After repeating the output vector from the Encoder \\(n\\) times, we obtain a sequence with exact the same length with the associated output sequence, we can leave the computation for the Decoder network! And that's the idea behind seq2seq. It's not as hard as it might seem, right?

So we now know about how to output a sequence from an input of different length. But what about the lengths of input sequences? As I mentioned above, the input sequences themselves don't necessarily have the exact same length, either! Sounds like an other headache, doesn't it? Fortunately, it's far more relaxing than the problem above. In fact, all we need to do is just something called: **Zero Padding**. To make it easier for you to understand, let's see the image below:

![five_sentences](/images/projects/sequence-to-sequence/five_sentences.png)

Here I prepared five sentences (they were actually from a great song of Twenty One Pilots, link provided at Reference) and let's imagine that they will be the input sequences to our network. As you could see, three sentences are not equal in length. To make them all equal in length, let's take the length of the longest sentence as the common length, and we only need to add one same word some times to the end of the other two, until they have the same length as the longest one. The added word must not resemble any words in the sentences, since it will cause their meaning to change. I will use the word **ZERO**, and here's the result I received:

![ZERO_added](/images/projects/sequence-to-sequence/ZERO_added.png)

You might get this now. And that's why it is called **zero padding**. In fact, what I did above is not exactly zero padding, and we will likely implement it differently. I'll tell you more in the Implementation section. For now, all I wanted to do is just to help you understand zero padding without any hurt.

We are half way there! We now know all we need to know about the-state-of-the-art Sequence to Sequence Learning. I can't help jumping right into Implementation section. Neither can you, right?

### Implementation

(You can find the whole source files on my GitHub repository here: [seq2seq](https://github.com/ChunML/seq2seq){:target="_blank"})

So, now we are here, finally, right in the Implementation section. Working with NLP problems is literally abstract (than what we did in Computer Vision problems, which we could at least have some visualization). Even worse, deep neural network in common is kind of abstract itself, so it seems that thing's gonna get more complicated here. That's the reason why I decided not to dig into details in the previous section, but to explain it along with the corresponding part in the code instead so that you won't find it difficult to understand the abstract terms (at least I think so). And now, let's get your hands dirty!

As usual, we will start with the most tedious (and boring) but important task, which is **Data Preparation**. As you already saw in my previous post, it is a little bit more complicated to prepare language data rather than image data. You will understand why soon.

Before doing any complicated processing, first we need to read the training data from file. I defined *X_data* and *y_data* variables to store the input text and the output text, respectively. They are all raw string objects, which means that we must split them into sentences:

{% highlight python %}
X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
{% endhighlight %}

Let's break it down for a better understanding. I will use the three sentences above as our *X_data*, here's what happened after *X_data.split('\n')*

![split_sentences](/images/projects/sequence-to-sequence/split_sentences.png)

The easiest way to split a raw text into sentences is looking for the line break. Of course, there are many other better ways, but let's make it simple this time. So, from a raw text we now obtained an array of sentences.

Next, for each sentence in the array, we must then split it into an array of words, or say it in a more proper way, a sequence of words. We will do this by using Keras' predefined method called **text_to_word_sentence**, as illustrated below:

![split_words](/images/projects/sequence-to-sequence/split_words.png)

Splitting a sentence into a sequence of words is harder than splitting text into sentences, since there are many ways to seperate words in a sentence, e.g. spaces, commas and so on. So we should not self-implement it but make use of predefined method instead. **text_to_word_sentence** also helps us remove all the sentence ending marks such as periods or exclamation marks. Quite helpful, isn't it?

So, here's what we received, an array of sequences of words:

![sequences](/images/projects/sequence-to-sequence/sequences.png)

But wait! There's one minor change which needs to be made to the input sequences, as mentioned from the paper as follow:

> We found
it extremely valuable to reverse the order of the words of the input sentence. So for example, instead
of mapping the sentence a, b, c to the sentence α, β, γ, the LSTM is asked to map c, b, a to α, β, γ,
where α, β, γ is the translation of a, b, c. This way, a is in close proximity to α, b is fairly close to
β, and so on, a fact that makes it easy for SGD to “establish communication” between the input and
the output. We found this simple data transformation to greatly boost the performance of the LSTM.

If you noticed the graph I drew above, you would have some doubt about the order of the input sequence. Yeah, as you might guess, the order of the input sequence is reversed before going into the network. And that's the reason why I added **[::-1]** to reverse the sequence split from the raw text. So, the final input sequences look like below:

![reverse_sequences](/images/projects/sequence-to-sequence/reverse_sequences.png)

Seems like we're done, right? But sadly, we are only half way there before we can actually have the network train our data. As computers can only understand the gray scale values of pixels in an image, inputting sequences of raw human-alike words will make no sense to computers. For that reason, we need to take a further step, which is converting the raw words into some kind of numeric values. To do that, we need a dictionary to map from a word to its corresponding index value, and another dictionary for the same purpose, but in reverse direction.

But first, what we need is a vocaburaly set. You can think of vocabulary set as an array which stores all the words in the raw text, but each word only appears once.

{% highlight python %}
dist = FreqDist(np.hstack(X))
X_vocab = dist.most_common(vocab_size-1)
dist = FreqDist(np.hstack(y))
y_vocab = dist.most_common(vocab_size-1)
{% endhighlight %}

In real deep learning projects, especially when we're dealing with NLP problems, our training data is pretty large in size, which the number of vocabularies may be up to millions. Obviously, that's too much for our computers to handle. Furthermore, words which appear only a few times (typically once or twice) in the whole text may not have a significant impact on the learning of our network. So, what we do first is to count the frequency which a word appears in the text, then we create the vocabulary set using only 10000 words with highest frequencies (you can change to 20000 or more, but make sure that your machine can handle it).

The result may look like below:

![vocab_set](/images/projects/sequence-to-sequence/vocab_set.png)

So we just have created the vocabulary set from the input text. In the next step, we will create two dictionaries to map between each word and its index in the vocabulary set, and vice versa.

{% highlight python %}
# Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
X_ix_to_word = [word[0] for word in X_vocab]
# Adding the word "ZERO" to the beginning of the array
X_ix_to_word.insert(0, 'ZERO')
# Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
X_ix_to_word.append('UNK')
{% endhighlight %}

With the vocabulary set we created above, it's pretty easy to create an array to store only the words, and eliminate their frequencies of occurrence (we don't need that information after all). But you may wonder, we were supposed to create some kind of dictionary here in order to convert each index to its associated word, and now what I told you to create is an array. Well, since we want to create the index-to-word dictionary, and we can access any element of an array through its index, it's better just to create a simple array instead of a dictionary where keys are all indexes! I'm sure you get that now.

Next, we will need to add two special words. As I mentioned earlier, we need a word called **ZERO** in order to make all sequences have the exact same length, and another word called **UNK**, which stands for **unknown words** or **out of vocabulary** in order to represent words which are not in the vocabulary set. There's nothing special with the word "UNK", which we can just append it to the end of the index-to-word array. But I want you to pay attention to the word "ZERO", **it must be the element of index 0**! You will understand why as we move on to the next steps.

So here's what the index-to-word array looks like:

![ix_to_word](/images/projects/sequence-to-sequence/ix_to_word.png)

As I told you above, don't forget to confirm that the word **ZERO** always be the first element before moving to the next step!

Our next step is pretty simple which is creating the word-to-index dictionary from the array above, so all we need is just a single line of code!

{% highlight python %}
# Create the word-to-index dictionary from the array created above
X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}
{% endhighlight %}

Let's confirm the dictionary we have just created. Once again, make sure the word **ZERO** is associated with the index 0! After that, we can move on to the next step.

![word_to_ix](/images/projects/sequence-to-sequence/word_to_ix.png)

So now we got the two dictionaries ready. The next step is pretty simple: we will loop through the sequences and replace every word in each sequence by its corresponding index number. And also remember that we're only putting 10000 words with highest frequencies into the vocabulary set, which also means that our network will actually learn words from that vocabulary set only. So here comes the question: What happens to the other words and how can we converse them to numeric values? That's where the word **UNK** makes sense. It stands for "Unknown words", or it's sometimes called **OOV**, which means "Out Of Vocabulary". So, for words which are not in the vocabulary set, we will simply assign them as **UNK**. And as you may guess, they will all have the same index value.

{% highlight python %}
# Converting each word to its index value
for i, sentence in enumerate(X):
    for j, word in enumerate(sentence):
        if word in X_word_to_ix:
            X[i][j] = X_word_to_ix[word]
        else:
            X[i][j] = X_word_to_ix['UNK']
{% endhighlight %}

And here's what we obtained. Obviously, our sequences don't contain any **ZERO**, so the converted sequences only contain numbers from \\(1\\) (which are the associated indexes of words in the sequences).

![index_sequence](/images/projects/sequence-to-sequence/index_sequence.png)

And now we got an array of sequences which all elements are numeric values instead of raw words. In the next step, we will use Keras' **pad_sequences** method to pad zeros into our sequences, so as all the sequences will have a same length. I told you about zero padding above, so there's not much left to talk here, I think.

{% highlight python %}
X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
y = pad_sequences(y, maxlen=y_max_len, dtype='int32')
{% endhighlight %}

And here's what our sequences looks like, after zero padded.

![zero_pad_index_sequence](/images/projects/sequence-to-sequence/zero_pad_index_sequence.png)

As you could see from the image above, what **pad_sequences** method did is just add additional \\(0\\) to each sequence, to make all the sequences have a same length with the longest one. So it's very important that the original sequences don't contain any \\(0\\). That's the reason why we must add the word **ZERO** to the beginning of the index-to-word array, so that the index of every word in the vocabulary set is not \\(0\\). If we don't, and have some word with index \\(0\\) instead, then our network won't be able to decide whether that \\(0\\) is padded zero, or index of a particular word. And it will definitely lead to a really bad learning.

So we now got a new array of sequences which all the lengths are the same. But it still can't be understand by the network. Concretely, we have to do a final processing step called vectorization:

{% highlight python %}
sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
for i, sentence in enumerate(word_sentences):
    for j, word in enumerate(sentence):
        sequences[i, j, word] = 1.
{% endhighlight %}

Explaining the process of vectorization (especially in terms of NLP) is kind of tedious, so I think it's better help you guys have a visualization of it. I'm quite sure you will get it just by having a look at the image below. A picture is worth a thousand words!

![vectorization](/images/projects/sequence-to-sequence/vectorization.png)

So, we have finished the toughest part and got our training data ready. Phew! You'd better take a break, we all deserve it!

In the next step, we will create the **encoder** network. Since we need to compute only a single vector from the input sequences, the **encoder** network is pretty simple, just a network with a single hidden layer is far from enough.

But wait! What the heck is **Embedding**, you may probably ask. In fact, we are supposed to input directly the vectorized array from above step into some kind of recurrent neural network like LSTM or vanilla RNN. But what we're gonna do is slightly different. We will vectorize only the output sequences, and leaving the zero padded input sequences unchanged. Then, we will put that input sequences into a special layer called **Embedding** first. Remember that you don't necessarily use that **Embedding** layer, instead you can just vectorize the input sequences and put it directly to the LSTM layer. Talking further into Word Embedding is beyond the scope of this post. The reason I use that layer is just to obtain a better result, from the fact that the size of vocabulary set is pretty small. I will definitely talk about Word Embedding in the coming post, I promise. For now, you have two choices, and it's all on you.

{% highlight python %}
model = Sequential()
model.add(Embedding(X_vocab_len, 1000, input_length=X_max_len, mask_zero=True))
model.add(LSTM(hidden_size))
{% endhighlight %}

Next, we will create the **decoder** network, which does the main job. First, we need to repeat the single vector outputted from the **encoder** network to obtain a sequence which has the same length with the output sequences. The rest is similar to the **encoder** network, except that the **decoder** will be more complicated, which we will have two or more hidden layers stacked up. For ones who are not familiar with Recurrent Neural Networks and how to create them using Keras, please refer to my previous post from the link in the beginning of this post.

{% highlight python %}
model.add(RepeatVector(y_max_len))
for _ in range(num_layers):
    model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(y_vocab_len)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
{% endhighlight %}

So we finally got everything ready. Let's go ahead and train our models. Due to some limitations of memory, I was able to train 1000 sequences, which means 1 batch at a time (with batch size 1000). I still can't find another better solution to this probem. If you guys have some ideas about it, please kindly let me know :)

{% highlight python %}
for k in range(k_start, NB_EPOCH+1):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    for i in range(0, len(X), 1000):
        if i + 1000 >= len(X):
            i_end = len(X)
        else:
            i_end = i + 1000
        y_sequences = process_data(y[i:i_end], y_max_len, y_word_to_ix)

        print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))
        model.fit(X[i:i_end], y_sequences, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
    model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
{% endhighlight %}

At the time of writing, the model is on its third day of learning and everything seems promising. I will continue to update the result, maybe after letting it learn for four or five more days!

### Summary

So, in today's blog post, I have talked about the incapability of normal RNN networks to deal with complicated NLP problems, where sequences differ in length. And through a project of creating an English-Finnish Language Translating Model, I also introduced to you a solution to this big problem by using Sequence To Sequence Learning Approach. It's just a simple experiment, so obviously, there are many places that you can improve. Feel free to play with the model and modify it for your own purposes.

After all, language modeling is a quite complicated problem, I think, and so is Sequence To Sequence Approach. For that reason, I don't expect you to fully understand the idea behind it just by reading this blog post (I myself can't say that I fully understand it, either!). So I recommend you to take not just one look at the paper, but to read it many times to grab a better understanding. And just don't forget that we can always discuss here to help each other learn better. Hope you all enjoy my post, and I'm gonna see you guys soon, on my next post.

### Reference

* Ilya Sutskever, Oriol Vinyals and Quoc V. Le. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf){:target="_blank"}
* Keras Addition RNN (Sequence to Sequence Learning based implementation) [addition_rnn](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py){:target="_blank"}
* The sentences above were from Heathens, an addicting song of Twenty One Pilots which I kept repeating recently. Watch it here: [Heathens](https://www.youtube.com/watch?v=UprcpdwuwCg){:target="_blank"}
