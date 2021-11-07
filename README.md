# VisionPonder
VIT version of [ponderNet](https://arxiv.org/abs/2107.05407). Improve adaptive computation performance. This is the repository to use ponderNet in [VIT](https://arxiv.org/abs/2010.11929) (code from the pytorch ver. [repo](https://github.com/lucidrains/vit-pytorch)).

## Installation

Clone the respository.

```bash
git clone https://github.com/fryegg/visionponder.git
```

## Usage

```command
python train_cifar10.py --net ponder --data cifar10 --bs 64 --max-steps 20
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
