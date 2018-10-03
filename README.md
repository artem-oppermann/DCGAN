# DCGAN
Deep convolutional generative adversarial network (DCGAN)for image generation. The DCGAN model in `src` can be used to generate random images. The model must be trained on existing images so it can be able to learn the underlying image distribution to generate samples from it. The model accepts only tf_records files. Make sure to transform the raw .png, .jpeg images into .tfrecord fiel format with `src\tf_record_writer.py`. The file writer accepts only 64x64 pixel images.

## Celebrity Faces Generation
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/0.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/1.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/2.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/6.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/7.jpg)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/original%20samples/10.jpg)

![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/Celeb_faces/generated%20samples/gen_sample.png)

## MNIST Generation

![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/MNIST/original%20samples/mnist_samples.png)
![alt text](https://github.com/artem-oppermann/DCGAN/blob/master/MNIST/generated%20samples/MNIST_DCGAN_generation_animation.gif)
