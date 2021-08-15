from absl import app
from absl import flags
import torchvision.transforms as transforms
from data import  CelebADataModule

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'crop_size', default=128,
    help=('The dataset we are interested to train out vae on')
)

flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.')
)

flags.DEFINE_integer(
    'image_size', default=64,
    help=('Image size for training.')
)

flags.DEFINE_string(
    'data_dir', default="kaggle",
    help=('Data directory for training.')
)

def celeba_dataloader(bs, IMAGE_SIZE, CROP, DATA_PATH):
    trans = []
    trans.append(transforms.RandomHorizontalFlip())
    if CROP > 0:
        trans.append(transforms.CenterCrop(CROP))
    trans.append(transforms.Resize(IMAGE_SIZE))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    dm = CelebADataModule(data_dir=DATA_PATH,
                                target_type='attr',
                                train_transform=transform,
                                val_transform=transform,
                                download=True,
                                batch_size=bs)
    return dm


def main(argv):
    del argv

    bs = FLAGS.batch_size
    IMAGE_SIZE = FLAGS.image_size
    CROP = FLAGS.crop_size
    DATA_PATH = FLAGS.data_dir

    dm = celeba_dataloader(bs, IMAGE_SIZE, CROP, DATA_PATH)

    dm.prepare_data() # force download now
    dm.setup() # force make data loaders 

if __name__ == '__main__':
  app.run(main)
