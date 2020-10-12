import torch
from pytorch_lightning import (
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import (
    resnet18,
    resnet50,
)

from model.model import (
    LitFood101,
    LitFood101KD,
)
from utils.args import get_program_level_args


def main():
    parser = get_program_level_args()
    parser = LitFood101.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint(monitor="avg_val_acc", mode="max")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        benchmark=False,
        checkpoint_callback=checkpoint_callback,
        precision=16 if args.amp_level != "O0" else 32,
    )

    # create model
    model = resnet18(pretrained=True)
    if args.use_knowledge_distillation:
        teacher_model = resnet50(pretrained=False)
        model = LitFood101KD(model, teacher_model, args)
    else:
        model = LitFood101(model, args)

    if args.evaluate:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        trainer.test(model, test_dataloaders=model.test_dataloader())
        return 0

    trainer.fit(model)

    trainer.test()


if __name__ == "__main__":
    main()
