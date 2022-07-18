# Neuromorphic Backdoors

## Installation

```bash
git clone https://github.com/GorkaAbad/NeuromorphicBackdoors
cd NeuromorphicBackdoors
pip install -r requirements.txt
```

## Usage

```bash
python main.py

usage: main.py [-h] [-T T] [-model MODEL] [-trigger TRIGGER]
               [-polarity POLARITY] [-trigger_size TRIGGER_SIZE] [-b B]
               [-b_test B_TEST] [-epochs N] [-j N] [-channels CHANNELS]
               [-data_dir DATA_DIR] [-out_dir OUT_DIR]
               [-dataname {gesture,cifar10}] [-resume RESUME] [-amp] [-cupy]
               [-opt OPT] [-lr LR] [-loss LOSS] [-momentum MOMENTUM]
               [-step_size STEP_SIZE] [-T_max T_MAX] [-epsilon EPSILON]
               [-target TARGET] [-pos POS] [-type TYPE] [-seed SEED]
               [-load_model] [-trigger_label TRIGGER_LABEL]

Classify DVS128 Gesture

options:
  -h, --help            show this help message and exit
  -T T                  simulating time-steps
  -model MODEL          model name
  -trigger TRIGGER      The index of the trigger label
  -polarity POLARITY    The polarity of the trigger
  -trigger_size TRIGGER_SIZE
                        The size of the trigger as the percentage of the image
                        size
  -b B                  batch size
  -b_test B_TEST        batch size for test
  -epochs N             number of total epochs to run
  -j N                  number of data loading workers (default: 1)
  -channels CHANNELS    channels of Conv2d in SNN
  -data_dir DATA_DIR    root dir of DVS128 Gesture dataset
  -out_dir OUT_DIR      root dir for saving logs and checkpoint
  -dataname {gesture,cifar10}
                        dataset name
  -resume RESUME        resume from the checkpoint path
  -amp                  automatic mixed precision training
  -cupy                 use CUDA neuron and multi-step forward mode
  -opt OPT              use which optimizer. SDG or Adam
  -lr LR                learning rate
  -loss LOSS            loss function
  -momentum MOMENTUM    momentum for SGD
  -step_size STEP_SIZE  step_size for StepLR
  -T_max T_MAX          T_max for CosineAnnealingLR
  -epsilon EPSILON      Percentage of poisoned data
  -target TARGET        Index for the target label
  -pos POS              Position of the triggger
  -type TYPE            Type of the trigger, 0 static, 1 dynamic
  -seed SEED            The random seed value
  -load_model           load model from checkpoint
  -trigger_label TRIGGER_LABEL
                        The index of the trigger label
```
