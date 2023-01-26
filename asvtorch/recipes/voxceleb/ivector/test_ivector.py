# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

# Main script for voxceleb/ivector recipe.

import sys
import os
# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asvtorch', 1)[0])
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import numpy as np

from asvtorch.src.settings.abstract_settings import AbstractSettings
from asvtorch.src.settings.settings import Settings
import asvtorch.recipes.voxceleb.data_preparation as data_preparation  # same preparation as in deep embedding recipe
import asvtorch.src.misc.fileutils as fileutils
from asvtorch.src.frontend.feature_extractor import FeatureExtractor
from asvtorch.src.utterances.utterance_selector import UtteranceSelector
from asvtorch.src.utterances.utterance_list import UtteranceList
from asvtorch.src.ivector.kaldi_ubm_training import train_kaldi_ubm
from asvtorch.src.backend.vector_processing import VectorProcessor
from asvtorch.src.backend.plda import Plda
from asvtorch.src.evaluation.scoring import score_trials_plda, prepare_scoring
from asvtorch.src.evaluation.eval_metrics import compute_eer, compute_min_dcf
import asvtorch.src.backend.score_normalization as score_normalization
import asvtorch.src.misc.recipeutils as recipeutils
from asvtorch.src.ivector.posteriors import extract_posteriors
from asvtorch.src.ivector.ivector_extractor import IVectorExtractor
from asvtorch.src.ivector.gmm import Gmm

@dataclass
class RecipeSettings(AbstractSettings):
    start_stage: int = 0
    end_stage: int = 100
    preparation_datasets: Optional[List[str]] = None
    feature_extraction_datasets: List[str] = field(default_factory=lambda:[])
    augmentation_datasets: Dict[str, int] = field(default_factory=lambda:{})
    selected_iteration: int = 5

# Initializing settings:
Settings(os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'init_config.py')) 

# Set the configuration file for KALDI MFCCs:
Settings().paths.kaldi_mfcc_conf_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'mfcc.conf')

# Add recipe settings to Settings() (these settings may not be reusable enough to be included in settings.py)
Settings().recipe = RecipeSettings()  

# Get full path of run config file:
run_config_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'run_configs.py') 

# Get run configs from command line arguments
# sys.argv[1] = "ivec"
run_configs = sys.argv[1:]
run_configs = "prepare fe ubm post ivec ext_ivec score_ivec".split()
if not run_configs:
    sys.exit('Give one or more run configs as argument(s)!')

Settings().print()

# These small trial lists are used between epochs to compute EERs:
small_trial_list_list = [
    recipeutils.TrialList(trial_list_display_name='vox1_original', dataset_folder='voxceleb1', trial_file='veri_test.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_cleaned', dataset_folder='voxceleb1', trial_file='veri_test2.txt')
]

# These are the all VoxCeleb trial lists used for final testing:
full_trial_list_list = [
    *small_trial_list_list,
    recipeutils.TrialList(trial_list_display_name='vox1_extended_original', dataset_folder='voxceleb1', trial_file='list_test_all.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_extended_cleaned', dataset_folder='voxceleb1', trial_file='list_test_all2.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_hard_original', dataset_folder='voxceleb1', trial_file='list_test_hard.txt'),
    recipeutils.TrialList(trial_list_display_name='vox1_hard_cleaned', dataset_folder='voxceleb1', trial_file='list_test_hard2.txt')
]

# Run config loop:
for settings_string in Settings().load_settings(run_config_file, run_configs):
    
    # I-vector extraction, stage 7
    if Settings().recipe.start_stage <= 7 <= Settings().recipe.end_stage:

        ivector_extractor = IVectorExtractor.from_npz(Settings().recipe.selected_iteration, Settings().computing.device)

        print('Loading trial data...')
        trial_data = UtteranceList.load('trial_posteriors', folder=fileutils.get_posterior_folder())
        ivector_extractor.extract(trial_data)
        trial_data.save('trial_ivectors')

        print('Loading PLDA data...')
        training_data = UtteranceList.load('training_posteriors', folder=fileutils.get_posterior_folder())
        ivector_extractor.extract(training_data)
        training_data.save('plda_ivectors')
        
        ivector_extractor = None

print('All done!')
