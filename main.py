import argparse
import sys
sys.path.append("pytorch-SAC")

from sac import SAC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC algorithm arguments parser.")
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        default=False,
        dest="test",
        help="If you want to test the agent."
    )
    parser.add_argument(
        "-e", "--env_name",
        action="store",
        default="Ant-v3",
        type=str,
        dest="env_name",
        help="Env to train/test agent in."
    )
    parser.add_argument(
        "-l", "--log_dir",
        action="store",
        default="Ant-v3",
        type=str,
        dest="log_dir",
        help="Directory to store checkpoints and tensorboard summarys while training in."
    )
    parser.add_argument(
        "-c", "--continue_training",
        action="store_true",
        default=False,
        dest="continue_training",
        help="Continue training from a checkpoint"
    )
    parser.add_argument(
        "-r", "--render_testing",
        action="store_true",
        default=True,
        dest="render_testing",
        help="Render window when testing agent."
    )
    parser.add_argument(
        "-n", "--num_test_games",
        action="store",
        default=1,
        type=int,
        dest="num_test_games",
        help="How many games to play when testing."
    )

    parser.add_argument(
        "--version",
        action="version",
        version="PyTorch-SAC Version 0.1"
    )

    args = parser.parse_args()

    sac = SAC(env_name=args.env_name, data_save_dir=args.log_dir)
    if not args.test:
        sac.train(resume_training=args.continue_training)
    else:
        sac.test(render=args.render_testing, use_internal_policy=False, num_games=args.num_test_games)
