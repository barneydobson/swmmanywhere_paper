#PBS -lselect=1:ncpus=1:mem=20gb
#PBS -lwalltime=01:00:00
#PBS -J 0-2

# subjob's index within the array

## All subjobs run independently of one another


module load anaconda3/personal
source sa_base/bin/activate

# Change to the submission directory
cd $PBS_O_WORKDIR

env_name="env_${PBS_ARRAY_INDEX}"
virtualenv-clone /rds/general/user/bdobson/home/swmmanywhere_paper/src/sa_base $env_name

deactivate

source $env_name/bin/activate

# Run program, passing the index of this subjob within the array
python experimenter.py --jobid=$PBS_ARRAY_INDEX --config_path=/rds/general/user/bdobson/ephemeral/swmmanywhere/cranbrook_node_1439.1/config.yml
deactivate 
rm -rf $env_name
