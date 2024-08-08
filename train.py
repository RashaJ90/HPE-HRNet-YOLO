import argparse
import json
import logging
import time







def train(args: argparse.Namespace, ):
    
    # train/val DataSet (assert)
    # defien dataloader args
    # train/val DataLoader  
    
    # Network steps
    pass


def main() -> None:
        parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train classification model",
        epilog=(
            "Usage examples:\n"
            "--save-frequency 1 --smoothing-alpha 0.1 --mixup-alpha 0.2 --aug-level 3 --resume-epoch 28\n"
        ),
        # formatter_class=cli.ArgumentHelpFormatter,
    )
        parser.add_argument()
        
        
        args = parser.parse_args()

        assert '' # assert args
        # check data/ model directory if exist , if not create new directory and print info
        # import labels config / data config/ features config ..
        # define paths : data/ aug, ... and check if paths exist
        # define train_sample, val_sample, print info of the path and size of sample
        # train 
        

if __name__ == "__main__":
    main()