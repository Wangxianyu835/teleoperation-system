"""Compare unified NN/dex exports; this does not establish cross-person accuracy."""
import argparse
import json
from pathlib import Path

from teleoperation.contracts.constants import HAND_MODELS
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.learning.hand_evaluation import compare_hand_methods


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nn',type=Path,required=True)
    parser.add_argument('--dex',type=Path,required=True)
    parser.add_argument('--hand-model',choices=HAND_MODELS,required=True)
    parser.add_argument('--hand',choices=('left','right','both'),default='right')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sides=('left','right') if args.hand=='both' else (args.hand,)
    specs={(args.hand_model,s):load_hand_spec(args.hand_model,s) for s in sides}
    result=compare_hand_methods(args.nn,args.dex,specs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
