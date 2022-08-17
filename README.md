# Backdoor Attacks on Spiking NNs and Neuromorphic Datasets

## Installation

```bash
git clone https://github.com/GorkaAbad/NeuromorphicBackdoors
cd NeuromorphicBackdoors
pip install -r requirements.txt
```

## Usage

Donwload the datasets [DVS Gesture](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794) and [N-MNIST](https://www.garrickorchard.com/datasets/n-mnist). For more information about the datasets, refer to the [SpikingJelly doc](https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/neuromorphic_datasets.html).

The directory should be like this:

```text
.
|-- DvsGesture.tar.gz
|-- LICENSE.txt
|-- README.txt
`-- gesture_mapping.csv
```

Then provide the path to the dataset in the -data_dir argument.

```bash
python main.py

usage: main.py [-h] [-T T] [-model {dummy}] [-trigger TRIGGER] [-polarity {0,1,2}] [-trigger_size TRIGGER_SIZE] [-b B] [-b_test B_TEST] [-epochs N] [-j N] [-channels CHANNELS]
               [-data_dir DATA_DIR] [-out_dir OUT_DIR] [-dataname {gesture,cifar10,mnist}] [-resume RESUME] [-amp] [-opt OPT] [-lr LR] [-loss LOSS] [-momentum MOMENTUM]
               [-step_size STEP_SIZE] [-T_max T_MAX] [-epsilon EPSILON] [-pos POS] [-type TYPE] [-seed SEED] [-load_model] [-trigger_label TRIGGER_LABEL] [-tau TAU] [-use_plif]
               [-use_max_pool] [-detach_reset] [-save_path SAVE_PATH]

Backdoor Attacks on Spiking NNs and Neuromorphic Datasets

options:
  -h, --help            show this help message and exit
  -T T                  simulating time-steps
  -model {dummy}        Model name
  -trigger TRIGGER      The index of the trigger label
  -polarity {0,1,2}     The polarity of the trigger
  -trigger_size TRIGGER_SIZE
                        The size of the trigger as the percentage of the image size
  -b B                  batch size
  -b_test B_TEST        batch size for test
  -epochs N             number of total epochs to run
  -j N                  number of data loading workers (default: 1)
  -channels CHANNELS    channels of Conv2d in SNN
  -data_dir DATA_DIR    root dir of DVS128 Gesture dataset
  -out_dir OUT_DIR      root dir for saving logs and checkpoint
  -dataname {gesture,cifar10,mnist}
                        dataset name
  -resume RESUME        resume from the checkpoint path
  -amp                  automatic mixed precision training
  -opt OPT              use which optimizer. sgd or adam
  -lr LR                learning rate
  -loss LOSS            loss function
  -momentum MOMENTUM    momentum for SGD
  -step_size STEP_SIZE  step_size for StepLR
  -T_max T_MAX          T_max for CosineAnnealingLR
  -epsilon EPSILON      Percentage of poisoned data
  -pos POS              Position of the triggger
  -type TYPE            Type of the trigger, 0 static, 1 dynamic
  -seed SEED            The random seed value
  -load_model           load model from checkpoint
  -trigger_label TRIGGER_LABEL
                        The index of the trigger label
  -tau TAU              Tau for the LIF node
  -use_plif             Use PLIF
  -use_max_pool         Use MaxPool
  -detach_reset         Detach reset
  -save_path SAVE_PATH  Path to save the model
```

## Reproducing Experiments

### DVS Gesture

#### Static Trigger

```bash
  python main.py -opt adam -loss cross -b 16 -T 16 -dataname gesture -epochs 65 -amp -epsilon 0.1 -polarity 1 -type 0 -pos top-left -trigger_size 0.1
```

#### Moving Trigger

```bash
    python main.py -opt adam -loss cross -b 16 -T 16 -dataname gesture -epochs 65 -amp -epsilon 0.1 -polarity 1 -type 1 -pos bottom-right -trigger_size 0.1
```

### N-MNIST

#### Static Trigger

```bash
  python main.py -opt adam -loss cross -b 16 -T 16 -dataname gesture -epochs 30 -amp -epsilon 0.001 -polarity 0 -type 0 -pos bottom-right -trigger_size 0.1
```

#### Moving Trigger

```bash
  python main.py -opt adam -loss cross -b 16 -T 16 -dataname gesture -epochs 30 -amp -epsilon 0.001 -polarity 0 -type 1 -pos middle -trigger_size 0.1
```
