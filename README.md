# ei-keyword-spotting-project
Keyword Spotting using Edge Impulse
This is my second project using Edge Impulse, inspired by [my first one](https://github.com/numinousmuses/ei-smartphone-motion-project). The project uses a machine learning mdoel trained and deployed using Edge Impulse to spot the keywords `goodnight` and `up`. Since the project doesn't make use of motion sensors like accelerometers, this project should be deployable to any smartphone or microcontroller with a mic that is at least 16 bits (size of raw input data for the model). Since higher quality mics can be downscaled, there is only a minimum bit requirement for the microphone used.

View the complete project on Edge Impulse.

As for this repository, it contains the data retrieval, neural networks, engineered feature data, and some visualizations.

# project description

## data

The data used was obtained from the Tensorflow Speech Commands Dataset as well as a (Keywords Dataset by Shawn Hymel)[https://github.com/ShawnHymel/custom-speech-commands-dataset/archive/]. The Tensorflow Speech Commands Dataset contains thousands of utterances of each of the following:
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

The data was processed using Mel Frequency Ceptral Coefficients (MFCC), which is ideal for human voice because through processing it creates an image from the audio data, so a neural network for image classification can be employed.

![image](https://user-images.githubusercontent.com/103385201/181100274-1cccfbc2-d799-4c95-bd13-306aaaa765d2.png)

Example cepstral coefficients of a sample of each class. They may look similar, but neural networks can establish patterns and differentiate between them. Also, a fun fact I learned today was that human speech is between 300-3400 Hz. The more you know. Moreover, audio data must be sampled at at least 2x the highest expected frequency, because due to the math behind FFS and MFCC, aliasing (unwanted overlaps in audio data) in a sample can occur. The theorem behind this is the Nyquist-Shannon sampling theorem. Applying this to human speech, it means you would want a sampling rate of at least 6800 Hz.


