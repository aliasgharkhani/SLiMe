import argparse


def add_base_args(parser):
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--attention_layers_to_use",
        nargs="+",
        type=str,
        default=[
            # 'down_blocks[0].attentions[0].transformer_blocks[0].attn1',
            # 'down_blocks[0].attentions[0].transformer_blocks[0].attn2',
            # 'down_blocks[0].attentions[1].transformer_blocks[0].attn1',
            # 'down_blocks[0].attentions[1].transformer_blocks[0].attn2',
            # 'down_blocks[1].attentions[0].transformer_blocks[0].attn1',
            # 'down_blocks[1].attentions[0].transformer_blocks[0].attn2',
            # 'down_blocks[1].attentions[1].transformer_blocks[0].attn1',
            # 'down_blocks[1].attentions[1].transformer_blocks[0].attn2',
            # 'down_blocks[2].attentions[0].transformer_blocks[0].attn1',
            # 'down_blocks[2].attentions[0].transformer_blocks[0].attn2',  ##########
            # 'down_blocks[2].attentions[1].transformer_blocks[0].attn1',
            # 'down_blocks[2].attentions[1].transformer_blocks[0].attn2',  ##########
            # 'up_blocks[1].attentions[0].transformer_blocks[0].attn1',
            "up_blocks[1].attentions[0].transformer_blocks[0].attn2",  ########## +
            # 'up_blocks[1].attentions[1].transformer_blocks[0].attn1',
            "up_blocks[1].attentions[1].transformer_blocks[0].attn2",  ########## +
            # 'up_blocks[1].attentions[2].transformer_blocks[0].attn1',
            "up_blocks[1].attentions[2].transformer_blocks[0].attn2",  ########## +
            # 'up_blocks[2].attentions[0].transformer_blocks[0].attn1',
            "up_blocks[2].attentions[0].transformer_blocks[0].attn2",  # +
            # 'up_blocks[2].attentions[1].transformer_blocks[0].attn1',
            "up_blocks[2].attentions[1].transformer_blocks[0].attn2",  # +
            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn1',
            # 'up_blocks[2].attentions[2].transformer_blocks[0].attn2',
            "up_blocks[3].attentions[0].transformer_blocks[0].attn1",
            # 'up_blocks[3].attentions[0].transformer_blocks[0].attn2',
            "up_blocks[3].attentions[1].transformer_blocks[0].attn1",  #############3
            # 'up_blocks[3].attentions[1].transformer_blocks[0].attn2',
            "up_blocks[3].attentions[2].transformer_blocks[0].attn1",
            # 'up_blocks[3].attentions[2].transformer_blocks[0].attn2',
            # 'mid_block.attentions[0].transformer_blocks[0].attn1',
            # 'mid_block.attentions[0].transformer_blocks[0].attn2'
        ],
    )

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--part_names", nargs="+", type=str)
    parser.add_argument("--checkpoint_dir", type=str, default="outputs")
    parser.add_argument("--text_prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser


def add_dataset_args(parser):
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pascal",
        choices=["sample", "pascal", "celeba"],
    )
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--test_data_dir", type=str)
    parser.add_argument("--min_crop_ratio", type=float, default=0.8)
    return parser


def add_train_args(parser):
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--train_mask_size", type=int, default=64)
    parser.add_argument("--train_t", nargs="+", type=int, default=[5, 100])
    parser.add_argument("--self_attention_loss_coef", type=float, default=1.0)
    parser.add_argument("--sd_loss_coef", type=float, default=0.005)
    return parser


def add_test_args(parser):
    parser.add_argument(
        "--masking",
        type=str,
        default="patched_masking",
        choices=["patched_masking", "simple"],
    )
    parser.add_argument("--num_patchs_per_side", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=400)
    parser.add_argument("--patch_threshold", type=float, default=0.2)
    parser.add_argument("--test_t", nargs="+", type=int, default=[100])
    parser.add_argument("--test_mask_size", type=int, default=512)
    parser.add_argument("--save_test_predictions", action="store_true", default=False)
    return parser


def init_args():
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    parser = add_dataset_args(parser)
    parser = add_train_args(parser)
    parser = add_test_args(parser)
    args = parser.parse_args()
    return args
