from spaghettini import load
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 Training")
    parser.add_argument("--cfg", type=str,
                        help="the path to the configuration file")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument("--test", "-t", action="store_true",
                        help="test the model only")
    args = parser.parse_args()

    assert os.path.basename(args.cfg) == "cfg.yaml"
    exp = load(args.cfg)
    if exp.exp_dir is None:
        exp_dir = os.path.dirname(args.cfg)
        print("generating default exp_dir: {}".format(exp_dir))
        exp.register_exp_dir(exp_dir)
    exp.launch(args.resume, cfg_path_source=args.cfg, test_only=args.test)
    print("finished")
    os._exit(0)

