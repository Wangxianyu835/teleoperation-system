"""Explicit target and input identity for unified hand workflows."""
from pathlib import Path

from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
from teleoperation.contracts.hand_models import HAND_MODELS



def add_target_arguments(parser):
    parser.add_argument('--hand-model',choices=HAND_MODELS,required=True)
    parser.add_argument('--backend',choices=('nn','dex'),required=True)
    parser.add_argument('--hand',choices=('left','right','both'),default='right')
    parser.add_argument('--input-profile',choices=(PALM_LENGTH_PROFILE,),default=PALM_LENGTH_PROFILE)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
