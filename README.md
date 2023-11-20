# deconvolution
Deconvolution pipeline with Pyxu

Requirements: 
$ my_env=deconvolution
$ my_branch=v2-dev
$ git clone https://github.com/matthieumeo/pycsou && cd pycsou/
$ git checkout "${my_branch}"
$ conda create --name "${my_env}"
$              --strict-channel-priority
$              --channel=conda-forge
$              --file=conda/requirements.txt
$ conda activate deconvolution
$ python -m pip install -e ".[complete_gpu]"
$ python -m pip install "napari[all]"
$ python -m pip install matplotlib
$ python -m pip install notebook
