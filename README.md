# GeneticAlgorithm
Car goes **brrrr**. It's actually a square with no wheels but whatever.

## Requirements

First of all, clone this repository (you can also download it as a .zip file)
```bash
git clone https://github.com/tiagoespsanto/GeneticAlgorithm.git
```

Please run the following command, to make sure that the required Python libraries are installed:
```bash
pip install -r requirements.txt
```

## How to run it?

Run the following command:
```bash
python genetic.py
```
The script lets you choose several options to personalize how the algorithm works:
1. **Tracks**: there are 2 tracks you can choose, the first one is shaped like the letter "M", and the second one ressembles an oval. You can also create your own tracks, by adding it to the source code
2. **Neural Network**: The network has 5 inputs (the distance sensors of the vehicle) and 2 outputs (speed and steering angle). You can change the number of neurons of the two hidden layers (as long as that number is greater than zero)
3. **Epochs for training**: you should train it for at least 20 epochs.
4. **Size of the Population**: the recommended minimum is 20, but please make sure that it is a multiple of 5, mutation is easier that way.
5. **Video Output**: Do you want to crate a cool video like the one in [Videos](#videos)? Just say yes.

## Videos
Oval Track:

https://user-images.githubusercontent.com/60974869/173205674-ee6cdec5-25c3-4146-b060-e85eeeb5d2d6.mp4

Slalom Track:

https://user-images.githubusercontent.com/60974869/173205942-192be650-b169-4074-ac0c-bd321d59e2bc.mp4
