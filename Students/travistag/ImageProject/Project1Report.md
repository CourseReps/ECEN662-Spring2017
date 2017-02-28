# Team Members
Viswam Nathan, Travis Taghavi, Daniel Whitten

# 1. Modeling

The problem was modeled as a binary hypothesis testing problem, with the observation being a set of pixels with RGB color information constituting an image. The two hypotheses were defined as follows:

* Hypothesis H0: The set of pixels constitutes a digital image of real life natural scenery
* Hypothesis H1: The set of pixels constitutes a computer generated image

We assume that there is some image-related feature alpha such that, the set of values that alpha assumes given H0 is true is distinguishable and seperable from the set of values that alpha assumes when H1 is true. The distributions of values that alpha assumes for each hypothesis will be established using the provided training data sets. We can then compare the probabilities P(alpha|H0) and P(alpha|H1) and choose the greater one to be the likelier hypothesis. This will require fitting distributions around the observed values of alpha from the training set and using these distributions as the basis to calculate the two probabilities mentioned earlier. Depending on the feature, a reasonable assumption could be that the distribution is normal and we can simply find the mean and variance of the set of values of alpha from each training set (scenery vs. computer generated) and fit Gaussian distributions with these parameters on the observed alpha values from each set.

If the observed distributions of alpha are such that the two distributions are well separated and have largely similar variances, we can assume that going from one hypothesis to the other is reflected by a monotonic trend in alpha. This will further simplify the hypothesis testing computation, as we can simply train a threshold value tau on the feature alpha such that (alpha >= tau) ===> H0 is true, else H1 is true.

The methods described here exhibit an equivalent structure to that of the log likelihood ratio, and we know that this structure is optimal for a binary hypothesis testing problem. Even though the provided training sets are lopsided, we assume that the priors for each hypothesis are equal and do not consider them as part of our algorithm.

# 2. Algorithm Structure

[Explain the feature here: sobel derivative, or most prevalent color or both]

[Explain how we find the threshold, or how we fit a distribution and take the ratio of probability]

# 3. Code

[Include the code here]

# 4. Presentation of Results

[Here we report accuracy, and also put in any plots that we generate from our data and summarize with any conclusions about what worked and what didn't work]