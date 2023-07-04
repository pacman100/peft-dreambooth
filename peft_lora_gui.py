# v1: initial release
# v2: add open and save folder icons
# v3: Add new Utilities tab for Dreambooth folder preparation
# v3.1: Adding captionning of images to utilities

import gradio as gr
import logging
import time

# import easygui
import json
import math
import os
import subprocess
import pathlib
import argparse
from library.common_gui import (
    get_folder_path,
    remove_doublequote,
    get_file_path,
    get_any_file_path,
    get_saveasfile_path,
    color_aug_changed,
    save_inference_file,
    gradio_advanced_training,
    run_cmd_advanced_training,
    gradio_training,
    gradio_config,
    gradio_source_model,
    run_cmd_training,
    # set_legacy_8bitadam,
    update_my_data,
    check_if_model_exist,
    output_message,
    verify_image_folder_pattern,
)
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from library.tensorboard_gui import (
    gradio_tensorboard,
    start_tensorboard,
    stop_tensorboard,
)
from library.dataset_balancing_gui import gradio_dataset_balancing_tab
from library.utilities import utilities_tab
from library.merge_lora_gui import gradio_merge_lora_tab
from library.svd_merge_lora_gui import gradio_svd_merge_lora_tab
from library.verify_lora_gui import gradio_verify_lora_tab
from library.resize_lora_gui import gradio_resize_lora_tab
from library.sampler_gui import sample_gradio_config, run_cmd_sample

from library.custom_logging import setup_logging
from peft_utils import combine_unet_and_text_encoder

# Set up logging
log = setup_logging()

# from easygui import msgbox

folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
document_symbol = '\U0001F4C4'   # 📄
path_of_this_folder = os.getcwd()


def save_configuration(
    save_as,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    network_dim,
    network_alpha,
    enable_lora_for_conv_modules,
    train_text_encoder,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    original_file_path = file_path

    save_as_bool = True if save_as.get('label') == 'True' else False

    if save_as_bool:
        log.info('Save as...')
        file_path = get_saveasfile_path(file_path)
    else:
        log.info('Save...')
        if file_path == None or file_path == '':
            file_path = get_saveasfile_path(file_path)

    # log.info(file_path)

    if file_path == None or file_path == '':
        return original_file_path  # In case a file_path was provided and the user decide to cancel the open action

    # Return the values of the variables as a dictionary
    variables = {
        name: value
        for name, value in sorted(parameters, key=lambda x: x[0])
        if name not in ['file_path', 'save_as']
    }

    # Extract the destination directory from the file path
    destination_directory = os.path.dirname(file_path)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Save the data to the selected file
    with open(file_path, 'w') as file:
        json.dump(variables, file, indent=2)

    return file_path


def open_configuration(
    ask_for_file,
    file_path,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    network_dim,
    network_alpha,
    enable_lora_for_conv_modules,
    train_text_encoder,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    ask_for_file = True if ask_for_file.get('label') == 'True' else False

    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(file_path)

    if not file_path == '' and not file_path == None:
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data = json.load(f)
            log.info('Loading config...')

            # Update values to fix deprecated use_8bit_adam checkbox, set appropriate optimizer if it is set to True, etc.
            my_data = update_my_data(my_data)
    else:
        file_path = original_file_path  # In case a file_path was provided and the user decide to cancel the open action
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ['ask_for_file', 'file_path']:
            values.append(my_data.get(key, value))

    # This next section is about making the LoCon parameters visible if LoRA_type = 'Standard'
    if my_data.get('LoRA_type', 'Standard') == 'LoCon':
        values.append(gr.Row.update(visible=True))
    else:
        values.append(gr.Row.update(visible=False))

    return tuple(values)


def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    network_dim,
    network_alpha,
    enable_lora_for_conv_modules,
    train_text_encoder,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    weighted_captions,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
):
    print_only_bool = True if print_only.get('label') == 'True' else False
    log.info(f'Start training LoRA ...')
    headless_bool = True if headless.get('label') == 'True' else False

    if pretrained_model_name_or_path == '':
        output_message(
            msg='Source model information is missing', headless=headless_bool
        )
        return

    if train_data_dir == '':
        output_message(
            msg='Image folder path is missing', headless=headless_bool
        )
        return

    if not os.path.exists(train_data_dir):
        output_message(
            msg='Image folder does not exist', headless=headless_bool
        )
        return
    
    if not verify_image_folder_pattern(train_data_dir):
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            output_message(
                msg='Regularisation folder does not exist',
                headless=headless_bool,
            )
            return
        
        if not verify_image_folder_pattern(reg_data_dir):
            return

    if output_dir == '':
        output_message(
            msg='Output folder path is missing', headless=headless_bool
        )
        return

    if int(bucket_reso_steps) < 1:
        output_message(
            msg='Bucket resolution steps need to be greater than 0',
            headless=headless_bool,
        )
        return

    if noise_offset == '':
        noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg='Noise offset need to be a value between 0 and 1',
            headless=headless_bool,
        )
        return

    # if float(noise_offset) > 0 and (
    #     multires_noise_iterations > 0 or multires_noise_discount > 0
    # ):
    #     output_message(
    #         msg="noise offset and multires_noise can't be set at the same time. Only use one or the other.",
    #         title='Error',
    #         headless=headless_bool,
    #     )
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless_bool
    ):
        return

    if optimizer == 'Adafactor' and lr_warmup != '0':
        output_message(
            msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
            title='Warning',
            headless=headless_bool,
        )
        lr_warmup = '0'

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        try:
            # Extract the number of repeats from the folder name
            repeats = int(folder.split('_')[0])

            # Count the number of images in the folder
            num_images = len(
                [
                    f
                    for f, lower_f in (
                        (file, file.lower())
                        for file in os.listdir(
                            os.path.join(train_data_dir, folder)
                        )
                    )
                    if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
                ]
            )

            log.info(f'Folder {folder}: {num_images} images found')

            # Calculate the total number of steps for this folder
            steps = repeats * num_images

            # log.info the result
            log.info(f'Folder {folder}: {steps} steps')

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            log.info(
                f"Error: '{folder}' does not contain an underscore, skipping..."
            )

    if reg_data_dir == '':
        reg_factor = 1
    else:
        log.info(
            '\033[94mRegularisation images are used... Will double the number of steps required...\033[0m'
        )
        reg_factor = 2

    log.info(f'Total steps: {total_steps}')
    log.info(f'Train batch size: {train_batch_size}')
    log.info(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    log.info(f'Epoch: {epoch}')
    log.info(f'Regulatization factor: {reg_factor}')

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            / int(gradient_accumulation_steps)
            * int(epoch)
            * int(reg_factor)
        )
    )
    log.info(
        f'max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}'
    )

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_dreambooth_peft.py"'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution="{max_resolution}"'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'

    if float(learning_rate) == 0:
        output_message(
            msg='Please input learning rate values.',
            headless=headless_bool,
        )
        return

    run_cmd += f' --network_dim={network_dim}'
    if enable_lora_for_conv_modules:
        run_cmd += f' --enable_lora_for_conv_modules'
    if train_text_encoder:
        run_cmd += f' --train_text_encoder'

    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    if scale_weight_norms > 0.0:
        run_cmd += f' --scale_weight_norms="{scale_weight_norms}"'

    run_cmd += f' --network_dropout="{network_dropout}"'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        # use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset_type=noise_offset_type,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        multires_noise_iterations=multires_noise_iterations,
        multires_noise_discount=multires_noise_discount,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    if print_only_bool:
        log.warning(
            'Here is the trainer command as a reference. It will not be executed:\n'
        )
        log.info(run_cmd)
    else:
        log.info(run_cmd)
        # Run the command
        if os.name == 'posix':
            os.system(run_cmd)
        else:
            subprocess.run(run_cmd)

        # check if output_dir/last is a folder... therefore it is a diffuser model
        last_dir = pathlib.Path(f'{output_dir}/{output_name}')

        if not last_dir.is_dir():
            # Copy inference model for v2 if required
            save_inference_file(
                output_dir, v2, v_parameterization, output_name
            )


def peft_lora_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
    headless=False,
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)

    gr.Markdown(
        'Train a custom model using kohya train network LoRA python code...'
    )
    (
        button_open_config,
        button_save_config,
        button_save_as_config,
        config_file_name,
        button_load_config,
    ) = gradio_config(headless=headless)

    (
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
        save_model_as,
        model_list,
    ) = gradio_source_model(
        save_model_as_choices=[
            'ckpt',
            'safetensors',
        ],
        headless=headless,
    )

    with gr.Tab('Folders'):
        with gr.Row():
            train_data_dir = gr.Textbox(
                label='Image folder',
                placeholder='Folder where the training folders containing the images are located',
            )
            train_data_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            train_data_dir_folder.click(
                get_folder_path,
                outputs=train_data_dir,
                show_progress=False,
            )
            reg_data_dir = gr.Textbox(
                label='Regularisation folder',
                placeholder='(Optional) Folder where where the regularization folders containing the images are located',
            )
            reg_data_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            reg_data_dir_folder.click(
                get_folder_path,
                outputs=reg_data_dir,
                show_progress=False,
            )
        with gr.Row():
            output_dir = gr.Textbox(
                label='Output folder',
                placeholder='Folder to output trained model',
            )
            output_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            output_dir_folder.click(
                get_folder_path,
                outputs=output_dir,
                show_progress=False,
            )
            logging_dir = gr.Textbox(
                label='Logging folder',
                placeholder='Optional: enable logging and output TensorBoard log to this folder',
            )
            logging_dir_folder = gr.Button(
                '📂', elem_id='open_folder_small', visible=(not headless)
            )
            logging_dir_folder.click(
                get_folder_path,
                outputs=logging_dir,
                show_progress=False,
            )
        with gr.Row():
            output_name = gr.Textbox(
                label='Model output name',
                placeholder='(Name of the model to output)',
                value='last',
                interactive=True,
            )
            training_comment = gr.Textbox(
                label='Training comment',
                placeholder='(Optional) Add training comment to be included in metadata',
                interactive=True,
            )
        train_data_dir.change(
            remove_doublequote,
            inputs=[train_data_dir],
            outputs=[train_data_dir],
        )
        reg_data_dir.change(
            remove_doublequote,
            inputs=[reg_data_dir],
            outputs=[reg_data_dir],
        )
        output_dir.change(
            remove_doublequote,
            inputs=[output_dir],
            outputs=[output_dir],
        )
        logging_dir.change(
            remove_doublequote,
            inputs=[logging_dir],
            outputs=[logging_dir],
        )
    with gr.Tab('Training parameters'):
        (
            learning_rate,
            lr_scheduler,
            lr_warmup,
            train_batch_size,
            epoch,
            save_every_n_epochs,
            mixed_precision,
            save_precision,
            num_cpu_threads_per_process,
            seed,
            caption_extension,
            cache_latents,
            cache_latents_to_disk,
            optimizer,
            optimizer_args,
        ) = gradio_training(
            learning_rate_value='0.0001',
            lr_scheduler_value='cosine',
            lr_warmup_value='10',
        )

        with gr.Row():
            network_dim = gr.Slider(
                minimum=1,
                maximum=1024,
                label='Network/LoRA Rank (Dimension)',
                value=8,
                step=1,
                interactive=True,
            )
            network_alpha = gr.Slider(
                minimum=1,
                maximum=1024,
                label='Network/LoRA Alpha',
                value=8,
                step=1,
                interactive=True,
                info='alpha for LoRA weight scaling',
            )
            enable_lora_for_conv_modules = gr.Checkbox(
                label='Enable LoRA for Conv modules',
                value=False,
                info='Enable LoRA for Conv modules',
            )
            train_text_encoder = gr.Checkbox(
                label='Train Text Encoder in addition to UNET',
                value=True,
                info='Train Text Encoder in addition to UNET',
            )

        with gr.Row():
            scale_weight_norms = gr.Slider(
                label='Scale weight norms',
                value=0,
                minimum=0,
                maximum=1,
                step=0.01,
                info='Max Norm Regularization is a technique to stabilize network training by limiting the norm of network weights. It may be effective in suppressing overfitting of LoRA and improving stability when used with other LoRAs. See PR for details.',
                interactive=True,
            )
            network_dropout = gr.Slider(
                label='Network dropout',
                value=0,
                minimum=0,
                maximum=1,
                step=0.01,
                info='Is a normal probability dropout at the neuron level. In the case of LoRA, it is applied to the output of down. Recommended range 0.1 to 0.5',
            )

        with gr.Row():
            max_resolution = gr.Textbox(
                label='Max resolution',
                value='512,512',
                placeholder='512,512',
                info='The maximum resolution of dataset images. W,H',
            )
            stop_text_encoder_training = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                label='Stop text encoder training',
                info='After what % of steps should the text encoder stop being trained. 0 = train for all steps.',
            )
            enable_bucket = gr.Checkbox(
                label='Enable buckets',
                value=True,
                info='Allow non similar resolution dataset images to be trained on.',
            )

        with gr.Row():
            no_token_padding = gr.Checkbox(
                label='No token padding', value=False
            )
            gradient_accumulation_steps = gr.Slider(
                label='Gradient accumulate steps',
                value='1',
                minimum=1,
                maximum=128,
                step=1,
            )
            weighted_captions = gr.Checkbox(
                label='Weighted captions',
                value=False,
                info='Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder.',
            )

        with gr.Row():
            prior_loss_weight = gr.Number(
                label='Prior loss weight', value=1.0
            )
            lr_scheduler_num_cycles = gr.Textbox(
                label='LR number of cycles',
                placeholder='(Optional) For Cosine with restart and polynomial only',
            )

            lr_scheduler_power = gr.Textbox(
                label='LR power',
                placeholder='(Optional) For Cosine with restart and polynomial only',
            )

        (
            # use_8bit_adam,
            xformers,
            full_fp16,
            gradient_checkpointing,
            shuffle_caption,
            color_aug,
            flip_aug,
            clip_skip,
            mem_eff_attn,
            save_state,
            resume,
            max_token_length,
            max_train_epochs,
            max_data_loader_n_workers,
            keep_tokens,
            persistent_data_loader_workers,
            bucket_no_upscale,
            random_crop,
            bucket_reso_steps,
            caption_dropout_every_n_epochs,
            caption_dropout_rate,
            noise_offset_type,
            noise_offset,
            adaptive_noise_scale,
            multires_noise_iterations,
            multires_noise_discount,
            additional_parameters,
            vae_batch_size,
            min_snr_gamma,
            save_every_n_steps,
            save_last_n_steps,
            save_last_n_steps_state,
            use_wandb,
            wandb_api_key,
            scale_v_pred_loss_like_noise_pred,
        ) = gradio_advanced_training(headless=headless)
        color_aug.change(
            color_aug_changed,
            inputs=[color_aug],
            outputs=[cache_latents],
        )

        (
            sample_every_n_steps,
            sample_every_n_epochs,
            sample_sampler,
            sample_prompts,
        ) = sample_gradio_config()

    with gr.Tab('Tools'):
        gr.Markdown(
            'This section provide Dreambooth tools to help setup your dataset...'
        )
        gradio_dreambooth_folder_creation_tab(
            train_data_dir_input=train_data_dir,
            reg_data_dir_input=reg_data_dir,
            output_dir_input=output_dir,
            logging_dir_input=logging_dir,
            headless=headless,
        )
        gradio_dataset_balancing_tab(headless=headless)
        gradio_merge_lora_tab(headless=headless)
        gradio_svd_merge_lora_tab(headless=headless)
        gradio_resize_lora_tab(headless=headless)
        gradio_verify_lora_tab(headless=headless)

    button_run = gr.Button('Train model', variant='primary')

    button_print = gr.Button('Print training command')

    # Setup gradio tensorboard buttons
    button_start_tensorboard, button_stop_tensorboard = gradio_tensorboard()

    button_start_tensorboard.click(
        start_tensorboard,
        inputs=logging_dir,
        show_progress=False,
    )

    button_stop_tensorboard.click(
        stop_tensorboard,
        show_progress=False,
    )

    settings_list = [
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
        logging_dir,
        train_data_dir,
        reg_data_dir,
        output_dir,
        max_resolution,
        learning_rate,
        lr_scheduler,
        lr_warmup,
        train_batch_size,
        epoch,
        save_every_n_epochs,
        mixed_precision,
        save_precision,
        seed,
        num_cpu_threads_per_process,
        cache_latents,
        cache_latents_to_disk,
        caption_extension,
        enable_bucket,
        gradient_checkpointing,
        full_fp16,
        no_token_padding,
        stop_text_encoder_training,
        # use_8bit_adam,
        xformers,
        save_model_as,
        shuffle_caption,
        save_state,
        resume,
        prior_loss_weight,
        network_dim,
        network_alpha,
        enable_lora_for_conv_modules,
        train_text_encoder,
        color_aug,
        flip_aug,
        clip_skip,
        gradient_accumulation_steps,
        mem_eff_attn,
        output_name,
        model_list,
        max_token_length,
        max_train_epochs,
        max_data_loader_n_workers,
        training_comment,
        keep_tokens,
        lr_scheduler_num_cycles,
        lr_scheduler_power,
        persistent_data_loader_workers,
        bucket_no_upscale,
        random_crop,
        bucket_reso_steps,
        caption_dropout_every_n_epochs,
        caption_dropout_rate,
        optimizer,
        optimizer_args,
        noise_offset_type,
        noise_offset,
        adaptive_noise_scale,
        multires_noise_iterations,
        multires_noise_discount,
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        additional_parameters,
        vae_batch_size,
        min_snr_gamma,
        weighted_captions,
        save_every_n_steps,
        save_last_n_steps,
        save_last_n_steps_state,
        use_wandb,
        wandb_api_key,
        scale_v_pred_loss_like_noise_pred,
        scale_weight_norms,
        network_dropout,
    ]

    button_open_config.click(
        open_configuration,
        inputs=[dummy_db_true, config_file_name] + settings_list,
        outputs=[config_file_name] + settings_list,
        show_progress=False,
    )

    button_load_config.click(
        open_configuration,
        inputs=[dummy_db_false, config_file_name] + settings_list,
        outputs=[config_file_name] + settings_list,
        show_progress=False,
    )

    button_save_config.click(
        save_configuration,
        inputs=[dummy_db_false, config_file_name] + settings_list,
        outputs=[config_file_name],
        show_progress=False,
    )

    button_save_as_config.click(
        save_configuration,
        inputs=[dummy_db_true, config_file_name] + settings_list,
        outputs=[config_file_name],
        show_progress=False,
    )

    button_run.click(
        train_model,
        inputs=[dummy_headless] + [dummy_db_false] + settings_list,
        show_progress=False,
    )

    button_print.click(
        train_model,
        inputs=[dummy_headless] + [dummy_db_true] + settings_list,
        show_progress=False,
    )

    return (
        train_data_dir,
        reg_data_dir,
        output_dir,
        logging_dir,
    )


def convert_peft_model_checkpoint_tab():
    gr.Markdown(
            'This utility will combine the PEFT text encoder and unet into a single safetensors file. '
            'This makes it easy to use with `https://github.com/AUTOMATIC1111/stable-diffusion-webui`'
        )
    with gr.Row():
        pretrained_model_name_or_path = gr.Textbox(
                label='Pretrained model name or path',
                placeholder='enter the path to custom model or name of pretrained model',
                value='runwayml/stable-diffusion-v1-5',
            )
        revision_id = gr.Textbox(
                label='Revision of pretrained model identifier from huggingface.co/models',
                placeholder='',
                value='',
            )
        peft_model_name_or_path = gr.Textbox(
                label='PEFT SD model name or path',
                placeholder='enter the path to custom model or name of pretrained model',
                value='',
            )
        peft_adapter_name = gr.Textbox(
                label='PEFT model\'s adapter name with which it was saved',
                placeholder='adapter name',
                value='default',
            )
        dump_path = gr.Textbox(
                label='Path to the output safetensors file for use with webui.',
                placeholder='lora_trained.safetensors',
                value='',
            )
        
        
        convert_button = gr.Button('Convert the checkpoint')
        float16 = gr.Checkbox(label='save in flat16 dtype', value=True)

        sd_checkpoint_revision = revision_id if revision_id != "" else None
        inputs = [
            pretrained_model_name_or_path,
            peft_model_name_or_path,
            dump_path,
            peft_adapter_name,
            float16,
            sd_checkpoint_revision
        ]

        convert_button.click(
            combine_unet_and_text_encoder,
            inputs=inputs,
            show_progress=False,
        )


def UI(**kwargs):
    css = ''

    headless = kwargs.get('headless', False)
    log.info(f'headless: {headless}')

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            log.info('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(
        css=css, title='PEFT DreamBooth GUI', theme=gr.themes.Default()
    )

    with interface:
        with gr.Tab('LoRA'):
            (
                train_data_dir_input,
                reg_data_dir_input,
                output_dir_input,
                logging_dir_input,
            ) = peft_lora_tab(headless=headless)
        with gr.Tab('Utilities'):
            utilities_tab(
                train_data_dir_input=train_data_dir_input,
                reg_data_dir_input=reg_data_dir_input,
                output_dir_input=output_dir_input,
                logging_dir_input=logging_dir_input,
                enable_copy_info_button=True,
                headless=headless,
            )
            with gr.Tab('Convert PEFT Model Chekpoint'):
                convert_peft_model_checkpoint_tab(headless=headless)

    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    log.info(launch_kwargs)
    interface.launch(**launch_kwargs)


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )
    parser.add_argument(
        '--headless', action='store_true', help='Is the server headless'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
        headless=args.headless,
    )
