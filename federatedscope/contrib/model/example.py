from federatedscope.register import register_model


# Build you torch or tf model class here
class MyNet(object):
    pass


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):

    model = MyNet()

    return model


def call_my_net(model_config, local_data):
    if model_config.type == "mynet":
        model = ModelBuilder(model_config, local_data)
        return model


register_model("mynet", call_my_net)
