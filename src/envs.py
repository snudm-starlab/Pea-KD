################################################################################
# Pea-KD: Parameter-efficient and accurate Knowledge Distillation
#
# Author: Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# Version : 1.0
# Date : Nov 19, 2020
# Main Contact: Ikhyun Cho
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
"""
Environment file. Mainly based on [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355).
"""

import os
import logging


logger = logging.getLogger(__name__)


PROJECT_FOLDER = os.path.dirname(__file__)
HOME_DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
HOME_OUTPUT_FOLDER = os.path.join(HOME_DATA_FOLDER, 'outputs/KD')
PREDICTION_FOLDER = os.path.join(HOME_DATA_FOLDER, 'outputs/predictions')
