# ei-keyword-spotting-project
Keyword Spotting using Edge Impulse
This is my second project using Edge Impulse, inspired by [my first one](https://github.com/numinousmuses/ei-smartphone-motion-project). The project uses a machine learning mdoel trained and deployed using Edge Impulse to spot the keywords `goodnight` and `up`. Since the project doesn't make use of motion sensors like accelerometers, this project should be deployable to any smartphone or microcontroller with a mic that is at least 16 bits (size of raw input data for the model). Since higher quality mics can be downscaled, there is only a minimum bit requirement for the microphone used.

View the complete project on Edge Impulse.

As for this repository, it contains the data retrieval, neural networks, engineered feature data, and some visualizations.

# project description

## data

The data used was obtained from the Tensorflow Speech Commands Dataset as well as a [Keywords Dataset by Shawn Hymel](https://github.com/ShawnHymel/custom-speech-commands-dataset/). The Tensorflow Speech Commands Dataset contains thousands of utterances of each of the following:
`* Backward
* Bed
* Cat
* Dog
* Down
* Eight
* Five
* Follow
* Forward
* Four
* Go
* Happy
* House
* Learn
* Left
* Marvin
* Nine
* No
* Off
* On
* One
* Right
* Seven
* Sheila
* Six
* Stop
* Three
* Tree
* Two
* Up
* Visual
* Wow
* Yes
* Zero`

Shawn Hymel's dataset contains the words:
* archie
* dracarys
* fenrir
* goodnight
* hadouken
* hey_archie
* hey_fenrir
* how_are_you
* joke
* trick_or_treat

In addition to words, background noise samples were also imported to augment the dataset by generating new samples by combining background noises with the current samples as well as changing the pitch.

For the project, only two keywords were chosen in order to abide to the project goal of deploying on microcontrollers or smartphones. The chosen commands are `goodnight` and `up`. In order to create a model able to identify the two words chosen, there need to be more than these two classes. The four classes needed are:
* goodnight
* up
* unknown
* random background noise

The unknown class is composed of non trigger words that don't bear a high similarity to the two keywords. This class and the random background class are needed to best enable the model to differentiate what words are spoken (whether they are the keywords or not) in addition to *if* words were being spoken.

So first, these datasets were imported and the 4 desired classes were separated then individually randomized. This was done in [Google Colab](https://colab.research.google.com/drive/1ZD_ZkqMV6e0_e3x2BH8uptzWTh-zBYUm?usp=sharing), credits to Shawn Hymel for the notebook and python script which separated the files.

Afterwards, the data was exported to Edge Impulse. In total, there are 20 mins of data for each of the classes, with each sample of data being 1 second long. The data was train/val/split 60/20/20. 

![raw data visualization](https://user-images.githubusercontent.com/103385201/181098822-82bf37b2-ce8a-4870-9b74-e37f0062376e.png)

Raw data visualization of samples of the up, goodnight, noise, and unknown classes.

## data processing

The data was processed using Mel Frequency Ceptral Coefficients (MFCC), which is ideal for human voice because through processing it creates an image from the audio data, so a neural network for image classification can be employed. MFCCs also mimic how the human ear perceives sound, meaning that the model can be considered to hear sound like humans.

![image](https://user-images.githubusercontent.com/103385201/181100274-1cccfbc2-d799-4c95-bd13-306aaaa765d2.png)

Example cepstral coefficients of a sample of each class. They may look similar, but neural networks can establish patterns and differentiate between them. A fun fact I learned today was that human speech is between 300-3400 Hz. The more you know. Moreover, audio data must be sampled at at least 2x the highest expected frequency, because due to the math behind FFS and MFCC, aliasing (unwanted overlaps in audio data) in a sample can occur. The theorem behind this is the Nyquist-Shannon sampling theorem. Applying this to human speech, it means you would want a sampling rate of at least 6800 Hz. 

## neural network

The model is composed of an input layer of 650 features, a reshape layer of 13 columns, two convolutional blocks made of a 1D convolutional layer (8 neurons for the first block, 16 for the second) and dropout layer (rate 0.25), a flatten layer, then an output layer. For the convolutional layers, the kernel size chosen was 3. The neural network was trained over 100 epochs. The notebook for this model should be found under the neural network folder. In the folder are also the h5 and tflite of this model, sawed under the fules ending with `model-ver-1`. The results of the first training are found below.

![image](https://user-images.githubusercontent.com/103385201/181104137-26739567-07ef-4c1b-b46c-936a82320403.png)

A 92 percent accuracy isn't bad, and should be functional for the purpose of simple detection on a microcontroller or smartphone, but attempts were still made to improve the accuracy of the model. The following image is the result after increasing the epochs to 150 and adding more data augmentation:

![image](https://user-images.githubusercontent.com/103385201/181106766-8417aab3-17ba-4d24-a25e-eb50ac260b5e.png)

The accuracy is more or less the same. This model can be found in the neural network folder as `model-ver-2`. An important consideration to keep in mind when making the model is model size. Increasing the size of the neural network to make it a deeper neural network may increase accuracy, but it may also increase the memory consumption, which in turn affects how deployable the model is to microcontrollers. In one more attempt to increase the accuracy of the model, 2D convolutional blocks were employed.

![image](https://user-images.githubusercontent.com/103385201/181109757-ea9bda2a-267a-47cd-aeb5-283af65f7718.png)

Seems like the first model was the best one. This model can be found as `model-ver-3`.

## model testing

The results of the testing is shown below.

![image](https://user-images.githubusercontent.com/103385201/181146236-374eea98-a4fe-40c5-bbad-efdfeae06886.png)

An accuracy of 89 percent. Based on the confusion matrix, `noise` and `goodnight` are able to be classified correctly almost 100% of the time, but the model is not as competent in classifying `unknown` and `up`. This is most likely due to how similar up is to some samples in the unknown dataset since up is a common phonetic component. A method to improve this would be to curate a dataset of unknowns dissimilar to `up` or to use a different keyword such as `upwards` or `to-the-sky!`.

## model and implementation analysis

The model collects then classifies audio snippets to detect the keywords `goodnight` and `up`. Despite having an accuracy of around 90%, this should be accurate enough for a practice project, but for proper implementation, a higher accuracy would be needed.

Completing this project, I learned what considerations should be made when choosing a probability threshold for a class. 

![image](https://user-images.githubusercontent.com/103385201/181146977-77326b2b-7eed-4bfd-94ba-6490e4f0ec84.png)

For example, say the task was to determine a probabiity threshold for the positive class using the graph above (credits to Edge Impulse for the graph). If false negatives were permitted, such as in a voice activation device where a false activation is completely unwanted, then the threshold would be set to point D. However, if some false positives were permissible, but absolutely no false negatives, such as in a healthcare diagnosis model, then point A would be chosen. Choosing a threshold is a compromise between whether and how many false positives or negatives you wish to permit.

# credits

Credits to the Introduction to Edge Impulse course and Shawn Hymel for the project walkthrough.
