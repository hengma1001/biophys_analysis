# Run CVAE 
```
python train_cvae.py -f ./cvae_input.h5 -d 3 -g 0 -b 100 -n all -e 10
```

## Installation on SUMMIT: 
1. TensorFlow gpu version from Anaconda
    ```
    conda install tensorflow-gpu 
    ```
2. Keras from pip
    ```
    pip install keras
    ```
python train_cvae.py -f ./cvae_input.h5 -g 2 -d 3 -b 64 -n all -e 100 
python train_cvae.py -f ../../biophys_analysis/traj_analysis/contact_maps_CoV.h5   -g 0 -d 3 -b 256 -n all -e 100
