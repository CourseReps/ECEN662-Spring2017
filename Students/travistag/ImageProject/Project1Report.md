# Team Members
Viswam Nathan, Travis Taghavi, Daniel Whitten

# 1. Modeling

The problem was modeled as a binary hypothesis testing problem, with the observation being a set of pixels with RGB color information constituting an image. The two hypotheses were defined as follows:

* Hypothesis H0: The set of pixels constitutes a digital image of real life natural scenery
* Hypothesis H1: The set of pixels constitutes a computer generated image

We assume that there is some image-related feature alpha such that, the set of values that alpha assumes given H0 is true is distinguishable and seperable from the set of values that alpha assumes when H1 is true. The distributions of values that alpha assumes for each hypothesis will be established using the provided training data sets. We can then compare the probabilities P(alpha|H0) and P(alpha|H1) and choose the greater one to be the likelier hypothesis. This will require fitting distributions around the observed values of alpha from the training set and using these distributions as the basis to calculate the two probabilities mentioned earlier. Depending on the feature, a reasonable assumption could be that the distribution is normal and we can simply find the mean and variance of the set of values of alpha from each training set (scenery vs. computer generated) and fit Gaussian distributions with these parameters on the observed alpha values from each set.

If the observed distributions of alpha are such that the two distributions are well separated and have largely similar variances, we can assume that going from one hypothesis to the other is reflected by a monotonic trend in alpha. This will further simplify the hypothesis testing computation, as we can simply train a threshold value tau on the feature alpha such that (alpha >= tau) ===> H0 is true, else H1 is true.

The methods described here exhibit an equivalent structure to that of the log likelihood ratio, and we know that this structure is optimal for a binary hypothesis testing problem. Even though the provided training sets are lopsided, we assume that the priors for each hypothesis are equal and do not consider them as part of our algorithm.

For each image in the training set, the value of the feature was calculated.
A histogram of feature values was constructed for images in both the real image and synthetic image set.
A normal distribution can be fit to both sets.
The real image set has a significant peak around a feature value of 0.014.
However, the "tails" of this distribution are very long.
Fitting a normal distribution to all data points does not capture the peak around 0.014.
This distribution is shown with the dashed red line in the figure.
Instead, feature values greater than 0.04 were considered to be outliers in this data set and a new normal distribution was constructed (solid red line).

<img src="plots/model_distributions.png" alt="modeled pdfs" style="width: 600px;"/>

The PDFs for H0 and H1 then having the following parameters:

|         | H0           | H1  |
| ------------- |:-------------:| :-----:|
| Mean     | 0.0145 | 0.153 |
| Std     | 0.00528      |   0.0306 |

# 2. Algorithm Structure

After trying a couple of image processing features, we settled on one that we thought had the right balance between computational efficiency and classification performance. The image is first converted to greyscale, and we then collect the respective intensity values of each pixel and create a histogram. This histogram is then normalized so that we now have, for each intensity bin, the percentage of the overall number of pixels that correspond to that intensity. Finally, the feature alpha is defined as the percentage of pixels comprised of the most common intensity value, i.e., the max value of the normalized histogram. This feature alpha is then computed for all the images in each of the two training data sets - real scenery and computer generated - so that we have distributions of alpha, f0 and f1, for each of the two hypotheses.

As alluded to before, we assume that the distribution of the percentage of most prevalent intensity in grayscale is Gaussian. Therefore, we calculate the mean and variance for each of the two distributions of alpha and fit two Gaussian waveforms around the observed alpha values from the two training sets. 

Then for each new testing image, we calculate the feature alpha and then find the probability of this alpha value with respect to the Gaussian distribution f0. This represents the likelihood L0 that H0 is the true hypothesis. Similarly, we calculate L1 using the distribution f1 and the same alpha value. A decision of H0 (real image) is returned when L0 > L1, else a decision of H1 (computer generated image) is returned.

# 3. Code

[Include the code here]

# 4. Presentation of Results

[Here we report accuracy, and also put in any plots that we generate from our data and summarize with any conclusions about what worked and what didn't work]
