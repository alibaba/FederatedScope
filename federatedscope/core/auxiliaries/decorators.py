def check_data_split(func):
    def wrapper(self, *args, **kwargs):
        # Obtain the value of `target_data_split_name` before running the function
        # TODO: more elegant way?
        if len(args) >= 1:
            target_data_split_name = args[0]
        elif 'target_data_split_name' in kwargs:
            target_data_split_name = kwargs['target_data_split_name']
        elif func.__name__ in ['train', 'finetune']:
            target_data_split_name = 'train'
        else:
            target_data_split_name = 'test'
        # Check if the `target_data_split` exists
        self.ctx.check_data_split(target_data_split_name)
        return func(self, *args, **kwargs)

    return wrapper


def use_diff(func):
    def wrapper(self, *args, **kwargs):
        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            before_metric = self.evaluate(target_data_split_name='val')

        num_samples_train, model_para, result_metric = func(
            self, *args, **kwargs)

        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            after_metric = self.evaluate(target_data_split_name='val')
            result_metric['val_total'] = before_metric['val_total']
            result_metric['val_avg_loss_before'] = before_metric[
                'val_avg_loss']
            result_metric['val_avg_loss_after'] = after_metric['val_avg_loss']

        return num_samples_train, model_para, result_metric

    return wrapper
