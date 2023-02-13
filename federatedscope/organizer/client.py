import gradio as gr

# from federatedscope.organizer.module.event_handler import EventHandler

# Rules: Naming Components `tab1_tab2_label_module`

# TODO: add manager

with gr.Blocks() as demo:
    gr.Markdown("Welcome to FederatedScope Cloud Demo!")

    with gr.Tab("Task Runner"):
        with gr.Box():  # "Basic settings"
            gr.Markdown("Basic settings")
            with gr.Row():
                with gr.Box():
                    data_text = gr.Markdown("Data")
                    with gr.Tab("Choose"):
                        data = gr.Dropdown(
                            ['adult', 'credit', 'abalone', 'blog'],
                            value=['adult'],
                            label='vFL data')
                    with gr.Tab("Upload"):
                        data_text = gr.Markdown("Data")
                        data_upload_button = gr.UploadButton(
                            "Click to Upload "
                            "data",
                            file_types=['file'],
                            file_count="single")
                opts = gr.Textbox(label='Opts')
        with gr.Box():  # "Tuning setting"
            gr.Markdown("Tuning setting")
            optimizer = gr.Dropdown(['rs', 'bo_rf', 'bo_gp'],
                                    value=['bo_rf'],
                                    label='Optimizer')
            with gr.Row():
                model = gr.Dropdown(['lr', 'xgb', 'gbdt'],
                                    value=['lr', 'xgb'],
                                    multiselect=True,
                                    label='Model Selection')
                feat = gr.Dropdown([
                    '', 'min_max_norm', 'instance_norm', 'standardization',
                    'log_transform', 'uniform_binning', 'variance_filter',
                    'iv_filter'
                ],
                                   value=[
                                       '', 'min_max_norm', 'instance_norm',
                                       'standardization', 'log_transform',
                                       'uniform_binning', 'variance_filter',
                                       'iv_filter'
                                   ],
                                   multiselect=True,
                                   label='Feature Engineer')
            with gr.Box():
                lr = gr.Markdown("Learning Rate")
                with gr.Row():
                    min_lr = gr.Slider(0, 1, value=0.1, label='Minimum')
                    max_lr = gr.Slider(0, 1, value=0.8, label='Maximum')
            with gr.Box():
                yaml = gr.Markdown("Yaml (optional)")
                upload_button = gr.UploadButton("Click to Upload a Yaml",
                                                file_types=['file'],
                                                file_count="single")
            upload_button.upload(None, upload_button, None)

        # with gr.Row():
        #     task_input = [gr.Textbox(label='Yaml'), gr.Textbox(label='Opts')]

        task_input = [data, opts, model, feat, min_lr, max_lr, optimizer]
        task_button = gr.Button("Launch")

    with gr.Tab("ECS Manager"):
        with gr.Row():
            ecs_input = [
                gr.Textbox(label='IP'),
                gr.Textbox(label='User'),
                gr.Textbox(label='Password', type='password')
            ]
        ecs_button = gr.Button("Add")

    output = gr.Textbox(label='Logs', lines=10, interactive=False)

    shutdown_button = gr.Button("Shutdown")

    # # Event
    # task_button.click(handler.handle_create_task,
    #                   inputs=task_input,
    #                   outputs=output)
    # ecs_button.click(handler.handle_add_ecs, inputs=ecs_input,
    # outputs=output)
    # shutdown_button.click(handler.handle_shut_down,
    #                       inputs=None,
    #                       outputs=output)

demo.launch(share=False, server_name="0.0.0.0", debug=True, server_port=7860)
