#PBS -lselect=1:ncpus=1:mem=20gb
#PBS -lwalltime=01:00:00
#PBS -J 0-2

# subjob's index within the array

## All subjobs run independently of one another
source /rds/general/user/bdobson/home/swmmanywhere_paper/src/sa_base/bin/activate

# Change to the submission directory
cd $PBS_O_WORKDIR

# Run program, passing the index of this subjob within the array
python experimenter.py --jobid=$PBS_ARRAY_INDEX --config_path=/rds/general/user/bdobson/ephemeral/swmmanywhere/cranbrook_node_1439.1/config.yml