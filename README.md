# Day to Night GAN Converter
* Ethan Cramer (EthanCramer2@my.unt.edu)
* Bobby Meyer (bobbybujan-meyer@my.unt.edu)
* Wesley DeLoach (WesleyDeLoach@my.unt.edu)
* Richard Tran (RichardTran2@my.unt.edu)

## INTRODUCTION
The issue of transposing an image from night to day or day to night is a common issue. The introduction of general adversarial networks (GANs) has made the application of changing seasons or changing any other attributes about an image possible. A GAN is a type of machine learning that was designed by Ian Goodfellow in 2014 where two neural networks compete against each other. [^5] GANs are used in cybersecurity for understanding deepfakes, in healthcare you can understand tumor detection and anomalies in a patient scan, or in editing photographs you can sharpen the image. We used a GAN to approach our problem of day to night, night to day, and changing attributes in an image. 

## RELATED WORKS
One of the most popular papers in this problem space is the seminal paper by Isola et al. as they are credited with the discovery of Conditional Adversarial Networks (conditional GANs) in solving many image-to-image translation problems; their famous pix2pix model can be applied to many general-purpose scenarios such as object reconstruction from only edge maps, transforming daytime images into nighttime images, transforming black and white photos to colored phones, turning architectural labels into realistic photos of building facades, etc. The original model uses a U-Net-based architecture for the generator and a convolutional PatchGAN classifier for the discriminator. See next section for details and definitions on GANs.[^1]

Besides the day2night image translation task, previous work has been done in creating and applying “transient scene attributes”, which involve applying high-level properties that affect the scene of the image, making an image “more autumn”, “more warm”, “more rain”, “more winter”, “more moist”, “more night”, etc. The most prominent attributes were acquired through PCA, regressors trained to recognize these transient attributes, and an appearance transfer method developed to apply these same transient attributes to new images.[^2]

As of 2021, there have been attempts to comprehensively assess the research landscape of Image-to-Image Translation (I2I) problems due to the sheer amount of new advancements and discoveries made every year. Pang et al. not only looked at Conditional and Unconditional GANs (Generative Adversarial Networks), but delineated the different methods to train GANs as well. They listed the most common evaluation metrics specifically for GANs, focusing on objective image quality assessments, which include common GAN metrics such as inception score (IS) and Frechet inception distance (FID). One of the biggest takeaways of the paper is the creation of an organizational framework for grouping all I2I translation models: I2I translation models can be first divided into two-domain or multi-domain models. Two-domain I2I models can be further categorized as supervised I2I, unsupervised I2I, semi-supervised I2I, and few-shot I2I. Multi-domain models allow for the ability to use a single unified model to produce multiple outputs to complete different image translation tasks; these can be categorized into unsupervised multi-domain I2I, semi-supervised multi-domain I2I, and few-shot multi-domain I2I. For example, the original pix2pix model falls under the two-domain, supervised I2I category.[^3]

## METHODS

### *Generative Adversarial Networks*

A generative adversarial network (GAN) creates a competition between two opponents, a generator and a discriminator. The generator tries to create fake images that are realistic enough to convince the discriminator they are real. The discriminator’s job is to distinguish between real and fake images. 

GANs can be conditional or unconditional. In an unconditional GAN the only input is a random noise vector, and therefore there is no way to influence what the GAN generates. A conditional GAN uses the random noise vector as well, but also supplies additional information in the form of data labels, text, or image attributes. This gives the GAN a little added direction. In either version of the GAN, the model is trained by updating the parameters of the generator and the discriminator using optimization methods such as stochastic gradient descent, Adam, or RMSProp. Once the discriminator can no longer tell the difference between the real and fake images, the model has completed training. 

Evaluating the performance of a GAN is an open and difficult problem. Many studies choose to use Amazon Mechanical Turk (AMT) perceptual studies in which participants are presented a series of two images, a real and a fake, and asked to pick which one is real. A variety of objective image quality assessment metrics exist but their correlation to the intended performance of the model is often indirect. To list a few: Peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), inception score (IS), perceptual distance (PD), Frechet inception distance (FID), FCN score, and classification accuracy. 

### *Pix2Pix Model[^1]*

The pix2pix framework of Isola et al. uses a conditional GAN to map input images to output images by training on a labeled dataset. It also learns the loss function to train the mapping, making the model highly generalizable to a variety of image-to-image translation problems. 
If we express the generator, G, as a mapping from the random noise vector z and the input image x to generated image y, and the discriminator, D, as a function of the input image x and the generated image y then we can describe the objective function of a conditional GAN as:



This model also considers the L1 distance (instead of L2, which causes more blurring), which can be described as:


Which gives us a final objective function:



The random noise vector z ensures that the results are not purely deterministic. 

Many image-to-image translation tasks employ a encoder-decoder network, which passes the input through a downsampler until a bottleneck is reached and the input is passed back up through the upsampler. Pix2Pix instead uses a U-Net generator, which allows some high-level image features to bypass the bottleneck by skipping from the downsampler layer to the mirroring upsampler layer. 

The optimization technique alternates gradient descent for the discriminator and the generator. The model uses a minibatch SGD with the Adam solver, a learning rate of 0.0002, and momentum parameters B1 = 0.5 and B2 = 0.999. 

### *cycleGAN Model[^4]*

The pix2pix model has a drawback in that it requires paired images, which can be hard to come by. Therefore, we want to compare it with an unsupervised model, cycleGAN, which can learn to translate images without paired input-output examples. The concept employed for this task is known as ‘cycle consistency’ which requires that if we translate an input image into an output image, then that output image should translate back to the original input image. The cycle consistency loss can be described as:



### *Novel Autoencoder Model*

In an attempt to try to create a novel model similar to pix2pix, there was an attempt to create an autoencoder that has a similar structure as the pix2pix generator and create a neural network mapping day image encodings to night image encodings and a separate neural network mapping night image encodings to day image encodings.

The dataset for this model was greatly reduced, where the image with the highest daylight value and the image with the highest night values in the original dataset were taken as the day and night inputs in the model.  This left about 100 images for day and 100 images for night, with 80% of the image pairs being put into the training set and the remaining 20% being held out for testing.

Preprocessing into the autoencoder involves compressing the image down to a 64 by 64 pixel image. The autoencoder cycles through convolutional, max pooling, batch normalization, and leaky rectified linear layers to downsample the image. After 3 cycles, it stores the output of the final cycle as the encodings. To decode the encodings, the autoencoder cycles through convolutional transpose, batch normalization, dropout, and rectified linear layers until the output is upscaled back to 64 by 64. The autoencoder is trained with L1 loss to try to avoid image blurring in the same vein as the pix2pix model. The encoding to encoding neural networks are simple feed forward networks with 1 hidden layer.

## RESULTS

### *Day2Night with pix2pix and cycleGAN*


In the day-time image to night-time image translation task, the pix2pix model was allowed to train for 40,000 steps (roughly 6 hours on a 8GB Radeon RX 580 GPU) with a lambda L1 loss coefficient of 100. The cycleGAN trained for 6 hours as well, which equated to 4,000 epochs with a lambda of 10. 

### *Night2Day with pix2pix and cycleGAN*



For the night-time image to day-time image translation task, the cycleGAN did not need to be retrained due to its twin-generator and twin-discriminator architecture and cycle-consistency constraint. The pix2pix model was retrained for a much more limited 4,000 steps due to lack of training time. This is why the pix2pix performed much more poorly in the night-to-day task.

### *Novel Autoencoder Day2Night*



The mean absolute training loss for the autoencoder ended at 0.1282, where the outputs range from 0 to 1.  On top of that the feed forward network for day to night encodings had a mean squared training loss of 0.3844. The ending mean absolute error for the day to night model on the test set was 0.3578.  

### *Novel Autoencoder Night2Day*



The feed forward network for night to day encodings had a mean squared training loss of 0.3398. The mean absolute test error for this model was 0.3660.

## DISCUSSION 

The supervised nature of the pix2pix model helps to produce images which look much closer in overall quality to the ground truth image, while suffering slightly on the crispness of the image. At a quick glance, the pix2pix outputs look almost indistinguishable from the ground truth images, with only minor blurriness and visual artifacts appearing upon closer inspection. 

The cycleGAN model performed acceptably for an unsupervised model. The images are much crisper than the pix2pix outputs, owing possibly to the cycle consistency constraint enforcing a sort of structural rigidity to the image, but the generated night-time and day-time effects were less convincing. In many cases it simply dimmed the image, leaving the sky a slightly darker blue. In some especially foggy or snowy images, it hardly darkened the image at all. But in many cases it did successfully create a believable image and a much crisper one than pix2pix. The cycleGAN model augmented with a few training example pairs could potentially become a very powerful semi-supervised image translation model. 

The novel autoencoder model heavily blurred the output from the ground truth, but still matched the color and the structure of the ground truth image decently. This result could be improved by increasing the amount of data used and improving the encoding feed forward network architecture. While the encoding model had a lower error, the outputs for the night2day model are a lot cloudier and don’t match the color as well as the day2night model does.

### *Future Work*
With more time given, a PyTorch implementation of the CycleGAN, pix2pix, and other GAN models could be done and results compared, since the original authors of the pix2pix even admitted the PyTorch implementation produces output results comparable or even better than the original Torch implementation. It would be interesting to compare the results of similar models created with Tensorflow versus Pytorch implementations as well and compare the performance of both.[^4]

Furthermore, it would be interesting to see how a significantly longer training period could affect the believability of results for all three models.

---

## REFERENCES 
[^1]: Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A, “Image-to-Image Translation with Conditional Adversarial Networks”, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1125-1134.
[^2]: Pierre-Yves Laffont and Zhile Ren and Xiaofeng Tao and Chao Qian and James Hays, “Transient Attributes for High-Level Understanding and Editing of Outdoor Scenes”, ACM Transactions on Graphics (proceedings of SIGGRAPH), 33, 4, 2014.
[^3]: Pang, Yingxue and Lin, Jianxin and Qin, Tao and Chen, Zhibo, “Image-to-Image Translation: Methods and Applications”, arXiv preprint  arXiv:2101.08629, 2021.
[^4]: J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223–2232. 
[^5]: “Generative Adversarial Network.” Wikipedia, Wikimedia Foundation, 24 Apr. 2022, https://en.wikipedia.org/wiki/Generative_adversarial_network. 


