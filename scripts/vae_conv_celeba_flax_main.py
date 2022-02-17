"""
Author: Ang Ming Liang

You need to download celeba dataset from kaggle.
First make sure you have kaggle.json,
as explained at https://github.com/Kaggle/kaggle-api#api-credentials.
then run the following commands in a cell or terminal
```
mkdir /root/.kaggle
cp kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
rm kaggle.json
python download_celeba.py
```

"""
from absl import app
from absl import flags
from vae_conv_celeba_flax_lib import VAE_celeba

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'figdir', default="results",
    help=('The dataset we are interested to train out vae on')
)

flags.DEFINE_float(
    'learning_rate', default=1e-3,
    help=('The learning rate for the Adam optimizer.')
)

flags.DEFINE_float(
    'kl_coeff', default=1,
    help=('The kl coefficient for loss.')
)

flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'num_epochs', default=5,
    help=('Number of training epochs.')
)

flags.DEFINE_integer(
    'latents', default=256,
    help=('Number of latent variables.')
)

flags.DEFINE_integer(
    'image_size', default=64,
    help=('Image size for training.')
)

flags.DEFINE_string(
    'data_dir', default="kaggle",
    help=('Data directory for training.')
)

def main(argv):

  del argv
  vae = VAE_celeba(
        FLAGS.figdir,
        FLAGS.data_dir,
        FLAGS.image_size,
        FLAGS.num_epochs,
        FLAGS.batch_size, 
        FLAGS.learning_rate, 
        FLAGS.kl_coeff, 
        FLAGS.latents)
  vae.main()
  
if __name__ == '__main__':
  app.run(main)
