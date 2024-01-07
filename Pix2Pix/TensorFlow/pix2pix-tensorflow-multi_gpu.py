
import os
import numpy as np
import tensorflow as tf
import time
import glob
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,5,6'


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    

mirrored_strategy = tf.distribute.MirroredStrategy()

BUFFER_SIZE = 400
BATCH_SIZE = 128
IMG_WIDTH = 256
IMG_HEIGHT = 256


# In[ ]:



n_gpu = 4


def imshow(image, figsize=(6,6)):
    image = np.uint8(image)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
    

def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    
    width = tf.shape(image)[1]
    width_half = width // 2
    
    input_image = image[:, :width_half, :]
    target_image = image[:, width_half:, :]
    
    input_image = tf.cast(input_image, dtype=tf.float32)
    target_image = tf.cast(target_image, dtype=tf.float32)
    return input_image, target_image

@tf.function
def random_jittering_mirroring(input_image, target_image, height=286, width=286):
    
    #resizing to 286x286
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    #cropping (random jittering) to 256x256
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    
    input_image, target_image = cropped_image[0], cropped_image[1]
    
    if tf.random.uniform(()) > 0.5:
    # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)
        
        
    return input_image, target_image

def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1
    return input_image, target_image



def preprocess_fn(image_path):
    input_image, target_image = read_image(image_path)
    input_image, target_image = random_jittering_mirroring(input_image, target_image)
    input_image, target_image = normalize(input_image, target_image)
    return input_image, target_image    


# In[10]:


train_path = glob.glob('edges2shoes/train/*')


# In[11]:


val_path = glob.glob('edges2shoes/val/*')


# In[12]:


# In[13]:
AUTOTUNE = tf.data.experimental.AUTOTUNE

batch_size = 64 * n_gpu

train_dataset = tf.data.Dataset.from_tensor_slices(train_path)
train_dataset = train_dataset.map(preprocess_fn,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(batch_size)

train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)



batch_size = 64 * n_gpu

val_files = tf.data.Dataset.from_tensor_slices(val_path)
val_files = val_files.shuffle(256)
val_dataset = val_files.map(preprocess_fn)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)


# In[15]:


#for (a,b) in train_dataset.take(1):
#    print(a.shape)
#    imshow(a[0])
#    imshow(b[0])


# In[16]:


OUTPUT_CHANNELS = 3



def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(layers.BatchNormalization())
    #result.add(tfa.layers.InstanceNormalization())


  result.add(tf.keras.layers.LeakyReLU())

  return result



def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
   
  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
      ]
        
    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
      ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



#generator.summary()


# generator.save('pix-gen.h5')


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer, activation='sigmoid')(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# In[29]:




with mirrored_strategy.scope():
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam((2e-4)*n_gpu, beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam((2e-4)*n_gpu, beta_1=0.5, beta_2=0.999)


# In[30]:


loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


# In[31]:


def generator_loss(disc_generated_output, gen_output, target, real_labels):
    Lambda =  100
    bce_loss = loss(real_labels, disc_generated_output)

    gan_loss = tf.reduce_mean(bce_loss)
    gan_loss = gan_loss/ n_gpu

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    l1_loss = l1_loss / n_gpu
    #print(l1_loss)

    total_gen_loss = gan_loss + (Lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# In[32]:


def discriminator_loss(disc_real_output, disc_generated_output, real_labels, fake_labels):
    bce_loss_real = loss(real_labels, disc_real_output)
    real_loss = tf.reduce_mean(bce_loss_real)
    real_loss = real_loss / n_gpu

    bce_loss_generated = loss(fake_labels, disc_generated_output)
    generated_loss = tf.reduce_mean(bce_loss_generated)
    generated_loss = generated_loss / n_gpu

    total_disc_loss = real_loss + generated_loss
    total_disc_loss = total_disc_loss / 2
    return total_disc_loss




def train_step(inputs):
    input_image, target = inputs
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        real_targets = tf.ones_like(real_output)
        fake_targets = tf.zeros_like(real_output)

        gen_total_loss, gen_gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target, real_targets)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, real_targets, fake_targets)
        

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
                                              
  
    return gen_gan_loss, l1_loss, disc_loss                                           



# In[35]:


EPOCHS = 300

# In[36]:



@tf.function
def distributed_train_step(dist_inputs):
    gan_l, l1_l, disc_l = mirrored_strategy.run(train_step, args=(dist_inputs,))
    gan_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, gan_l, axis=None)
    l1_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, l1_l, axis=None)
    disc_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_l, axis=None) 
    return gan_loss, l1_loss, disc_loss 

    
def fit():
    for epoch in range(EPOCHS):
        n = 0
        gan_loss, l1_loss, disc_loss = 0, 0, 0
        for dist_inputs in train_dataset:
            n += 1
            gan_l, l1_l, disc_l = distributed_train_step(dist_inputs)
            gan_loss += gan_l
            l1_loss += l1_l
            disc_loss += disc_l
            
        gan_loss = gan_loss / n
        l1_loss = l1_loss / n
        disc_loss = disc_loss / n

    print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f, L1_loss: %.3f'  % (
        (epoch), EPOCHS, disc_loss, gan_loss, l1_loss))

    generator.save_weights('model_objective/gen_'+ str(epoch) + '.h5')


fit()