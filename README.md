# DCGAN
Deep convolutional generative adversarial network (DCGAN) for image generation. The DCGAN model in `src` can be used to generate random images. The model must be trained on existing images so it can be able to learn the underlying image distribution to generate samples from it. The model accepts only tf_records files. Make sure to transform the raw `.png`, `.jpeg` images into `.tfrecord` file format with `src/tf_record_writer.py`. The file writer accepts only 64x64 pixel images. To train the model `src/train.py` must be executed. Everywhere in the code the appropriate paths to raw images, .tfrecord files etc. must be complemented.
Below are some examples of original and generated images by the model.

## Celebrity Faces Generation

The model was trained on about 200.000 images of celebrities (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Here are some cropped image examples from the dataset:<br/><br/>

![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/0.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/1.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/2.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/6.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/7.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/10.jpg)

While the model is training on the original images, it generates samples at the same time. Below are some samples generated after 100, 500, 2500 and 5000 training iterations. The training time until the 5000th iteration took about 1 hour on a GTX 760. To generate photorealistic samples more powerfull graphic cards is required as well as much longer training time (>24 h). <br/><br/>

![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/generated%20samples/gen_sample.png)

## MNIST Generation

In this example the model is trained on the famous hand written digits dataset (MNIST). : <br/><br/>
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/MNIST/original%20samples/mnist_samples.png)

After each epoch one sample is generated. In the end the progress of MNIST generation is summurized in a .gif:  <br/><br/>
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/MNIST/generated%20samples/MNIST_DCGAN_generation_animation.gif)


|   | Kunde seit  |Wohnort   | Beruf  | Einkommen  | Vermögensverhältnisse  | Verzug von Kreditraten| Credit-Score|
|---|---|---|---|---|---|---|---|
|1| 01.02.2012  | Bremen  | IT-Manager  | 69 000  | 120 000 | 0 | 98| |
|2| 27.08.2005  | Hannover  | Lehrer  | 52 000  | 12 000  | | 2| 50|
|3|  12.04.2018 | Hamburg  |  Lastkraftwagenfahrer| 38 000  | 14 000  | 0| 85| |
|4|  01.09.2019| Hamburg  |  Admin| 42 000  | 32 000  | 1| 75| |
|.|  .| .  |  .| .  | .  | .| .| |
|.|  .| .  |  .| .  | .  | .| .| |
|.|  .| .  |  .| .  | .  | .| .| |
|100 000|  08.07.2007| Oldenburg  |  Bürokaufmann| 36 000  | 3000  | 3| 17| |
