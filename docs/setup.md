# Requirements and Setup

### Requirements
Currently only tested with the following configuration:
- Nvidia RTX 5070Ti (16 GiB vram)
- Intel Ultra 9 275HX (32 GiB ram)

### Troubleshooting
If you run out of ram - try changing processing parameters.
In particular - you could try changing the following config (not necessarily all at once):
```
config.preprocessing.window_sec: 3.0 -> 4.0
config.preprocessing.window_overlap: 0.9 -> 0.8
config.preprocessing.stft_nperseg: 64 -> 128
config.preprocessing.stft_overlap: 48 -> 32
config.preprocessing.use_cache: True -> False
```

### Setup
1. ``pip install -r requirements.txt``
2. Download our dataset and move it into `.our_data/our_5_class`
(example data path: `.our_data/our_5_class/subject9/subject_4.poly5`).
3. You might want to manually swap "Right Hand" and "Left Hand" in 
`.our_data/our_5_class/subject4/subject_10_markers.csv`, since the dataset in the repo has them swapped.
4. Run `simple_train.ipynb`
