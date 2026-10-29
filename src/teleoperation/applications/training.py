"""CLI composition for target-specific hand training on a shared human H5."""


def run(args):
    from teleoperation.learning.trainer import train_joint_models
    from teleoperation.retargeting.hand.device import resolve_device
    train_joint_models(input_path=args.input, root=args.checkpoint_root, hand_models=args.hand_model,
                       run_name=args.run_name, hand_side=args.hand_side, epochs=args.epochs,
                       batch_size=args.batch_size, learning_rate=args.learning_rate,
                       val_ratio=args.val_ratio, early_stopping_patience=args.early_stopping_patience,
                       seed=args.seed, device=resolve_device(args.device), resume=args.resume,
                       init_checkpoint=args.init_checkpoint,input_profile=args.input_profile)
    return 0
