import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    from gi_tract_seg import utils
    from gi_tract_seg.training_pipeline import train

    utils.extras(config)
    return train(config)


if __name__ == "__main__":
    main()
