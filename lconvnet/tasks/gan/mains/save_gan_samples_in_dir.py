import os
import argparse
from pathlib import Path
import yaml
import json
import tempfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--im_size", type=int)
    parser.add_argument("--num_channels", type=int)
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()

    for filename in Path(args.root_dir).glob('**/hparams.json'):
        # Get the GAN root. 
        gan_path = "/".join(str(filename).split("/")[:-2])
        
        # Form the dictionary that will later be saved as json. 
        config_dict = dict()
        config_dict["distrib1"] = dict(name= "GANSampler",
                filepath= "lconvnet/tasks/dualnets/distrib/gan_sampler.py",
                gan_config_json_path= os.path.join(gan_path, "hparams", 
                    "hparams.json"),
                sample_size=64,
                test_sample_size=64,
                generate_type="generated",
                dim=args.im_size * args.im_size * args.num_channels)
        config_dict["num_imgs"] = 50000
        config_dict["im_size"] = args.im_size 
        config_dict["num_channels"] = args.num_channels 
        config_dict["base_save_path"] = os.path.join(gan_path, "saved_samples")
        config_dict["dataset"] = args.dataset 
        config_dict["dataset_size"] = 50000

        # Save the dictionary as json. 
        temp = tempfile.NamedTemporaryFile(mode="wt", suffix=".json")
        try:
            temp_name = temp.name
            json.dump(config_dict, temp, sort_keys=True, indent=4)
            temp.seek(0)

            # Run the command to save images. 
            command = "python -um lconvnet.tasks.gan.mains.save_gan_samples {}".format(temp_name)
            print("Executing: \n \t{}".format(command))
            os.system(command)
        finally:
            temp.close()
