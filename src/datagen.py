import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalize_diagnose(img, mask):
  img = img /255
  mask = mask / 255

  mask[mask >= 0.5] = 1
  mask[mask < 0.5] = 0

  return (img,mask)



def data_generator(data, batch_size, augmentation, target_size, image_color_mode = 'rgb', mask_color_mode='grayscale'):
  image_data_gen = ImageDataGenerator(**augmentation)
  mask_data_gen = ImageDataGenerator(**augmentation)

  image_gen = image_data_gen.flow_from_dataframe(
    data,
    x_col='img_path',
    color_mode= image_color_mode,
    class_mode = None,
    target_size=target_size,
    batch_size=batch_size,
    seed = 1
  )

  mask_gen = mask_data_gen.flow_from_dataframe(
    data, 
    x_col = 'mask_path',
    color_mode=mask_color_mode,
    class_mode=None, 
    target_size=target_size,
    batch_size = batch_size,
    seed = 1 
  )

  train_gen = zip(image_gen, mask_gen)
  for img, mask in train_gen:
    img, mask = normalize_diagnose(img,mask)
    yield (img, mask)

