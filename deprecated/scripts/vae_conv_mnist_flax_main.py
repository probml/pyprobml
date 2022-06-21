from absl import app
from absl import flags
from vae_conv_mnist_flax_lib import VAE_mnist

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'figdir', default="mnist_results",
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
    'latents', default=7,
    help=('Number of latent variables.')
)

flags.DEFINE_integer(
    'train_set_size', default=50000,
    help=('Number of latent variables.')
)

flags.DEFINE_integer(
    'test_set_size', default=10000,
    help=('Number of latent variables.')
)


def main(argv):

  del argv
  vae = VAE_mnist(
        FLAGS.figdir,
        FLAGS.train_set_size,
        FLAGS.test_set_size,
        FLAGS.num_epochs,
        FLAGS.batch_size, 
        FLAGS.learning_rate, 
        FLAGS.kl_coeff, 
        FLAGS.latents)
  vae.main()
  
if __name__ == '__main__':
  app.run(main)
