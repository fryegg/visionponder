# VisionPonder
VIT version of [ponderNet](https://arxiv.org/abs/2107.05407). Improve adaptive computation performance. This is the repository to use ponderNet in [VIT](https://arxiv.org/abs/2010.11929) (code from the pytorch ver. [repo](https://github.com/lucidrains/vit-pytorch)).

## Short Summary
We propose the vision ponder to have effective computation. Vision Ponder used VIT (Vision Transformer) \cite{dosovitskiy2021image}. model as a baseline and use PonderNet. We will also show computational steps on well-known datasets such as CIFAR10, CIFAR100, ImageNet to know how complex the setting between training and test set is to get high accuracy.

## Installation

Clone the respository.

```bash
git clone https://github.com/fryegg/visionponder.git
```

## Usage

```command
python train_cifar10.py --net ponder --data cifar10 --bs 64 --max-steps 20
```
## Results
- Python 3.6+
- PyTorch 1.0+
- epoch 10
| Dataset   | # Max Step | # patch size| Test loss | Test acc| Avg Halting Step|
|-----------|---------:|--------:|:-----------------:|:---------------------:||:---------------------:|
| CIFAR10  |    20    | 2  | 543.94| **68.98%**| **13.06**|
| CIFAR10  |    20    | 4  | 87.90| **60.31%**| **15.852**|
| CIFAR100 |    20    | 4   | 220.24| **30.02%**| **18.268**|


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
