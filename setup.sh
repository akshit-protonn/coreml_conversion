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

# fetch test video samples
brew install youtube-dl
youtube-dl -f 134 -o "mkbhd_m1.%(ext)s" "https://www.youtube.com/watch?v=f4g2nPY-VZc"
brew install ffmpeg
ffmpeg -ss 00:00:14 -t 00:00:10 -i mkbhd_m1.mp4 -async 1 mkbhd_m1_clip.mp4