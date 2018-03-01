from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from util import write_piano_roll_to_midi

import numpy as np



class BEATGAN():
    def __init__(self):
        self.img_rows = 96 
        self.img_cols = 84
        self.channels = 1
        self.num_classes = 14

        #had to mess around with learning rate a bit to help training
        optimizer = Adam(0.0002/10, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'], 
            optimizer=optimizer)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(128,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])
        # img = self.generator([noise])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, 
            optimizer=optimizer)
    
    

    def build_generator(self):

        model_input = Input(shape=(128,))
        hidden = Reshape((1,1,128))(model_input)
        hidden = Conv2DTranspose(1024,(1,1), strides = (1,1), name = 'h0')(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation('relu')(hidden)
        hidden_1 = Reshape((2,1,512))(hidden)
        hidden_1 = Conv2DTranspose(512, (2,1), strides = (2,1), name = 'h1')(hidden_1)
        hidden_1 = BatchNormalization()(hidden_1)
        hidden_1 = Activation('relu')(hidden_1)
        hidden_2 = Conv2DTranspose(256, (2,1), strides =(2,1), name = 'h2')(hidden_1)
        hidden_2 = BatchNormalization()(hidden_2)
        hidden_2 = Activation('relu')(hidden_2)
        hidden_3 = Conv2DTranspose(256, (2,1), strides = (2,1), name = 'h3')(hidden_2)
        hidden_3 = BatchNormalization()(hidden_3)
        hidden_3 = Activation('relu')(hidden_3)
        hidden_4 = Conv2DTranspose(128, (2,1), strides = (2,1), name = 'h4')(hidden_3)
        hidden_4 = BatchNormalization()(hidden_4)
        hidden_4 = Activation('relu')(hidden_4)
        hidden_5 = Conv2DTranspose(128, (3,1), strides= (3,1), name='h5')(hidden_4)
        hidden_5 = BatchNormalization()(hidden_5)
        hidden_5 = Activation('relu')(hidden_5)
        hidden_6 = Conv2DTranspose(16, (1,7), strides=(1,1), name = 'h6')(hidden_5)
        
        hidden_6 = BatchNormalization()(hidden_6)
        hidden_6 = Activation('relu')(hidden_6)
        hidden_7 = Conv2DTranspose(1, (1,12), strides = (1,12), name='h7')(hidden_6)
        
        output = Activation('tanh')(hidden_7)
        model = Model(inputs=model_input, outputs=output)
        model.summary()

        noise = Input(shape=(128,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, 128)(label))

        input = multiply([noise, label_embedding])

        img = model(input)

        return Model([noise, label], img)
       

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        test = np.ones((self.img_rows,self.img_cols,1))
        test = np.reshape(test,(1,self.img_rows,self.img_cols,1))
        
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.summary()

        img = Input(shape=img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, save_interval=50):

        # # Load the dataset
        X_train = np.load("/Users/jacob/Desktop/audio/musegan/preprocessing/numpy_samples/x_beats.npy")
        X_train = X_train[:,:96,:84]
        X_train = np.reshape(X_train,(14,96,84,1))
        
        X_train = np.repeat(X_train, 100, axis=0)
        
        
        self.save_truth(X_train)
        
        y_train = np.arange(0,14)
        # y_train = np.ones((14,))
        y_train = np.reshape(y_train,(14,1))
        y_train = np.repeat(y_train,100, axis=0 )
        # y_train = np.ones((14,1))
        half_batch = int(batch_size / 2)
        
        # Class weights:
        # To balance the difference in occurences of digit class labels. 
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch
        class_weights = [cw1, cw2]

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (half_batch, 128))
            # The labels of the digits that the generator tries to create and
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, half_batch).reshape(-1, 1)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])
            gen_imgs[gen_imgs>.75] = 1
            gen_imgs[gen_imgs<.75] = 0
            # gen_imgs = self.generator.predict([noise])
            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))
            
            # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = self.num_classes * np.ones(half_batch).reshape(-1, 1)

            # Train the discriminator
            print(imgs.shape)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels], class_weight=class_weights)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=class_weights)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 128))

            valid = np.ones((batch_size, 1))
            # Generator wants discriminator to label the generated images as the intended
            # digits
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels], class_weight=class_weights)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_model()
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 128))
        sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)
        print(sampled_labels)
        print(sampled_labels.shape)
        
        gen_imgs = self.generator.predict([noise, sampled_labels])
        #convert gen_imgs to midis and save
        for i in range(gen_imgs.shape[0]):
            sample = np.reshape(gen_imgs[0],(1,96,84))
            sample[sample>.75] = 1
            sample[sample<.75] = 0
            write_piano_roll_to_midi(sample,"./midis/%d.midi" % i, is_drum=True)
            
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.suptitle("BEATGAN: Generated Midi", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                # axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/beats_%d.png" % epoch)
        plt.close()
    
    #function to save grount truth
    def save_truth(self, img):
        r, c = 2, 5
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        gen_imgs = 0.5 * img + 0.5
        fig, axs = plt.subplots(r, c)
        fig.suptitle("BEATGAN: real midi", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/truth.png")
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "./saved_model/%s.json" % model_name
            weights_path = "./saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path, 
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "beatgan_generator")
        save(self.discriminator, "beatgan_discriminator")
        save(self.combined, "beatgan_adversarial")


if __name__ == '__main__':
    beatgan = BEATGAN()
    beatgan.train(epochs=10000, batch_size=14, save_interval=50)






