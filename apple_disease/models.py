"""The three ImageNet-pretrained backbones used by the paper."""
import timm


MODEL_NAMES = {
    "densenet121": "densenet121",
    "efficientnet_b7": "tf_efficientnet_b7.ra_in1k",
    "efficientnet_b7_noisy_student": "tf_efficientnet_b7.ns_jft_in1k",
}


def build_model(model_key: str, pretrained: bool = True):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_key!r}. Choose from: {', '.join(MODEL_NAMES)}")
    model = timm.create_model(MODEL_NAMES[model_key], pretrained=pretrained, num_classes=4)
    config = timm.data.resolve_model_data_config(model)
    return model, config
