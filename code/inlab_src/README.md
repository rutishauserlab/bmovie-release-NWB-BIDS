This folder contains the scripts utilized to create NWB files from lab-specific files used in the Rutishauser Lab at Cedars-Sinai Medical Center. Although these scripts are not directly relevant to the data release, they are included here as a reference for the generation of the NWB files.

To validate NWB files:
1- use nwbinspector https://github.com/NeurodataWithoutBorders/nwbinspector
supply a path to a directory containing NWB files
nwbinspector path/to/bmovie_NWBfiles/

2- use pynwb.validate https://pynwb.readthedocs.io/en/stable/validation.html
python -m pynwb.validate path/to/bmovie_NWBfiles/*.nwb



