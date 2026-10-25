"""Optimizer / dataloader / LR-scheduler construction and the
``accelerator.prepare`` pass for the Anima trainer."""

from __future__ import annotations

import logging
import math
import os

import torch

from library.training import (
    AcceleratedBundle,
    OptimizerBundle,
    get_optimizer,
    get_optimizer_train_eval_fn,
    get_scheduler_fix,
)

logger = logging.getLogger(__name__)


def build_optimizer_and_dataloaders(
    args,
    accelerator,
    network,
    train_dataset_group,
    val_dataset_group,
    collator,
) -> OptimizerBundle:
    """Build optimizer, dataloaders, and LR scheduler; finalize max_train_steps."""
    accelerator.print("prepare optimizer, data loader etc.")

    # make backward compatibility for text_encoder_lr
    support_multiple_lrs = hasattr(
        network, "prepare_optimizer_params_with_multiple_te_lrs"
    )
    if support_multiple_lrs:
        text_encoder_lr = args.text_encoder_lr
    else:
        if (
            args.text_encoder_lr is None
            or isinstance(args.text_encoder_lr, float)
            or isinstance(args.text_encoder_lr, int)
        ):
            text_encoder_lr = args.text_encoder_lr
        else:
            text_encoder_lr = (
                None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
            )
    try:
        if support_multiple_lrs:
            results = network.prepare_optimizer_params_with_multiple_te_lrs(
                text_encoder_lr, args.unet_lr, args.learning_rate
            )
        else:
            results = network.prepare_optimizer_params(
                text_encoder_lr, args.unet_lr, args.learning_rate
            )
        if type(results) is tuple:
            trainable_params = results[0]
            lr_descriptions = results[1]
        else:
            trainable_params = results
            lr_descriptions = None
    except TypeError:
        trainable_params = network.prepare_optimizer_params(
            text_encoder_lr, args.unet_lr
        )
        lr_descriptions = None

    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)
    optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(optimizer, args)

    train_dataset_group.set_current_strategies()
    if val_dataset_group is not None:
        val_dataset_group.set_current_strategies()

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    persistent_workers = args.persistent_data_loader_workers and n_workers > 0

    dataloader_kwargs = {
        "batch_size": 1,
        "collate_fn": collator,
        "num_workers": n_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": args.dataloader_pin_memory,
    }
    if n_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_group if val_dataset_group is not None else [],
        shuffle=False,
        **dataloader_kwargs,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader)
            / accelerator.num_processes
            / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is"
        )

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    return OptimizerBundle(
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        optimizer_args=optimizer_args,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        text_encoder_lr=text_encoder_lr,
        lr_descriptions=lr_descriptions,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
    )


def _prepare_unet(args, accelerator, unet, *, swapping: bool):
    # Re-apply with unsloth_offload if needed (after base has already enabled it).
    if args.unsloth_offload_checkpointing and args.gradient_checkpointing:
        unet.enable_gradient_checkpointing(unsloth_offload=True)

    if not swapping:
        return accelerator.prepare(unet)

    model = accelerator.prepare(unet, device_placement=[False])
    accelerator.unwrap_model(model).move_to_device_except_swap_blocks(
        accelerator.device
    )
    accelerator.unwrap_model(model).prepare_block_swap_before_forward()
    return model


def prepare_with_accelerator(
    args,
    accelerator,
    network,
    opt: OptimizerBundle,
    unet,
    text_encoders,
    vae,
    vae_dtype,
    weight_dtype,
    cache_latents,
    *,
    swapping: bool,
) -> AcceleratedBundle:
    """Cast model dtypes, run accelerator.prepare, flip train/eval."""
    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)
    for t_enc in text_encoders:
        # None when the TE was never loaded (cache_text_encoder_outputs with
        # no sample prompts / live encoding -- qwen3_needed=False).
        if t_enc is None:
            continue
        t_enc.requires_grad_(False)
        # in case of cpu, dtype is already set to fp32 because cpu does not support fp16/bf16
        if t_enc.device.type != "cpu":
            t_enc.to(dtype=weight_dtype)

    unet = _prepare_unet(args, accelerator, unet, swapping=swapping)

    network, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            network,
            opt.optimizer,
            opt.train_dataloader,
            opt.val_dataloader,
            opt.lr_scheduler,
        )
    )

    if args.gradient_checkpointing:
        # according to TI example in Diffusers, train is required
        unet.train()
        for t_enc in text_encoders:
            if t_enc is not None:
                t_enc.train()
    else:
        unet.eval()
        for t_enc in text_encoders:
            if t_enc is not None:
                t_enc.eval()

    accelerator.unwrap_model(network).prepare_grad_etc(text_encoders, unet)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    return AcceleratedBundle(
        network=network,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
        training_model=network,
        unet=unet,
        text_encoders=text_encoders,
        unet_weight_dtype=weight_dtype,
    )
