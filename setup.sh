# https://coremltools.readme.io/docs/installation
# https://formulae.brew.sh/cask/miniconda
brew install --cask miniconda
conda init "$(basename "${SHELL}")"
conda update -n base -c defaults conda
conda create --name coremltools-env
conda activate coremltools-env
conda install pip
conda install -c conda-forge coremltools

# https://jupyter.org/install
conda install -c conda-forge jupyterlab
jupyter-lab

conda install -c conda-forge opencv
conda install -c conda-forge matplotlib


# remove environment
conda deactivate
conda env remove -n coremltools-env