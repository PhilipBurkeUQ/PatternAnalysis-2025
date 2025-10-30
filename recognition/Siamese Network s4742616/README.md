# Siamese Network Classifier
## s4742616 Philip Burke COMP3710 Report


###Dataset Information

The dataset consisted of 33,126 dermoscopic .jpg images of benign and malignant skin lesions. Images varied in size and shape considerably. The test dataset provided on the ISIC2020 website contained no ground truth about the correct status of the lesion and so the choice was made to split the training dataset in order for predictions to be made about the generalisation of the performance of the algorithm.

There was significant class imbalance in the training set as 98.2% of the images were benign. 

###Pre-Processing
The pre-processing procedure was relatively straightforward. Since the data was all in .jpg format, it was converted to a tensor with 3 channels for RGB. The choice was made to keep all three rather than convert to greyscale as colour was found to be a predictor of melanoma (Mayo Clinic, 2025).

Due to the size of the dataset, loading the entire image directory into memory proved to be impossible and caused issues. Because of this, the decision was made to pre-process in batches as the data was being loaded to be used by the network. The decision obviously caused a bottleneck in performance, but allowed the algorithm to run without significant load on memory. This was done by creating two custom datasets in order to take advantage of torch's DataLoader, and in particular, the parallel workers that it is able to utilize.

The nature of a Siamese network meant that the network needed to have pairs of data, with labels reflecting the similarity of the images also. The custom dataset structure allowed for this to be easily loaded into to training process.

The actual preprocessing steps were incredibly straightforward. The only step was to convert to a tensor and then reshape into a 256*256 image. On reflection, to improve results, data augmentation such as horizontal flips or rotations should have been used as these would have allowed for a more robust model with better generalisation performance.

###Algorithm
The classifier worked by learning how to extract features from the images using a CNN that was optimized to minimize the Euclidean distance between 'similar' (i.e same label) images and maximise the distance between dissimilar images. From these extracted features a simple linear classifier was trained that would predict the class of the image.

The CNN had only 2 convolutional layers with ReLU activation and max pooling, before a fully connected layer mapped to a series of 'features'. A diagram of the CNN is shown below:



In order to extract the features, a pair of images, along with a label reflecting their similarity were loaded into a CNN. The CNN was then applied to both images and the Euclidean distance between the two was used as part of the loss function. This way, the network would learn how to extract features from images such that similar results i.e both benign or both malignant would give similar features which would make classifying the images easier.

After the network was trained to extract features from pairs of similar images, the linear classifier was trained using Binary Cross Entropy loss. The classifier would map from the features to a single point which would reflect the probability of a malignant lesion. Outputting as a probability (rather than just a classification) allowed more flexibility from the loss function. A diagram of the entire classification process is shown below:



###Results
Due to the massive imbalance in the classes, getting an 'accuracy of 0.8' as per the task requirements would be very simple as a classifier that predicted benign each time would get an accuracy of 0.983. The results in some other metrics are shown below:



###Dependencies



###References

https://www.mayoclinic.org/diseases-conditions/melanoma/in-depth/melanoma/art-20546856