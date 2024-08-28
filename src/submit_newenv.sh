#PBS -lselect=1:ncpus=1:mem=20gb
#PBS -lwalltime=01:00:00
#PBS -J 0-2

# subjob's index within the array

## All subjobs run independently of one another


module load anaconda3/personal
source activate sa

env_name="env_${PBS_ARRAY_INDEX}"
python -m venv $env_name

conda deactivate

source $env_name/bin/activate

cd /rds/general/user/bdobson/home/swmmanywhere_paper
pip install -e .

# Change to the submission directory
cd $PBS_O_WORKDIR

# Run program, passing the index of this subjob within the array
python src/experimenter.py --jobid=$PBS_ARRAY_INDEX --config_path=/rds/general/user/bdobson/ephemeral/swmmanywhere/cranbrook/cranbrook_hpc.yml

deactivate 
rm -rf $env_name
