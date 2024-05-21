# ProteinCLIP

## Introduction and background

## Installation

To install proteinCLIP, start by cloning this repository. Then, create the requisite conda environment, activate it, and install ProteinCLIP in **editable mode** using pip. For example:

```bash
conda env create -f environment.yml
conda activate proteinclip
pip install -e ./
```

Note: we highly recommend the [mamba package manager](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) as an alternative to conda. 

In addition to installation, you will likely need to download data files if you intend to train ProteinCLIP yourself; all datasets we use can be found at [Zenodo](https://zenodo.org/records/11176863). 

## using ProteinCLIP

We provide pre-trained ProteinCLIP "adapter" models for the [ESM2 family of models](https://github.com/facebookresearch/esm/tree/main) as well as [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc). These models are available under the `pretrained` directory and can be loaded using provided functions; see below for an example.

```python
from proteinclip import model_utils

m = model_utils.load_proteinclip("esm", 33)  # For ESM2, 33-layer model

# Create a synthetic example
# Size corresponds to embedding dimension of "parent" protein language model
model_input = np.random.randn(1280)
# ProteinCLIP expects input to be unit-normalized
model_input /= np.linalg.norm(model_input)
x = m.predict(np.random.randn())
print(x.shape)  # (128,)
print(np.linalg.norm(x))  # 1.0; ProteinCLIP produces unit-norm vectors
```

Pre-trained models are available for the following models

* ESM2, 36-layer: `model_utils.load_proteinclip("esm", 36)`
* ESM2, 33-layer: `model_utils.load_proteinclip("esm", 33)`
* ESM2, 30-layer: `model_utils.load_proteinclip("esm", 30)`
* ESM2, 12-layer: `model_utils.load_proteinclip("esm", 12)`
* ESM2, 6-layer: `model_utils.load_proteinclip("esm", 6)`
* ProtT5: `model_utils.load_proteinclip("t5")`

These models are stored in the [ONNX format](https://github.com/onnx/onnx) so feel free to write your own loaders as well.

## Example training commands

### Training ProteinCLIP

To train ProteinCLIP yourself, you can use the pre-computed embeddings that we have provided above, or you can compute your own embeddings stored in a hdf5 format as (uniprot ID -> embedding array). After you have obtained a protein embedding file, pass it to training script as follows:

Example command:
```bash
python bin/train_protein_clip.py configs/clip_hparams.json /path/to/uniprot_sprot.dat.gz /path/to/protein_embedding.hdf5 --unitnorm -g text-embedding-3-large
```

### Training protein-protein interaction classifier

We provide a training command to automatically train a protein-protein classifier using the data splits provided by Bernett et al. The input to this training call is a directory to a training run of the above ProteinCLIP; the relevant hdf5 embeddings for proteins will be loaded, as well as the CLIP architecture itself (as specified by the `--clipnum` argument). 

Example command:
```bash
python bin/train_ppi.py configs/supervised_hparams.json -c ./protein_clip/version_0 --clipnum 1 -n ppi_classifier
```

## References
(1) Bernett, J., Blumenthal, D. B., & List, M. (2024). Cracking the black box of deep sequence-based protein–protein interaction prediction. Briefings in Bioinformatics, 25(2), bbae076.
