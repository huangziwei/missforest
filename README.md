# README

Implementation of missForest in Python with a little help from ChatGPT (GPT-4).

## Installation

```bash
pip install git+https://github.com/huangziwei/missforest.git
```

## Usage
    
```python
from missforest import MissForest

mf = MissForest()
df_imputed = mf.fit_transform(df)
```