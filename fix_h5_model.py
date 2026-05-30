import os
import json
import shutil
import h5py


SOURCE_MODEL = os.path.join("model", "brain_tumor_vgg16_best.h5")
FIXED_MODEL = os.path.join("model", "brain_tumor_vgg16_best_fixed.h5")


def remove_quantization_config(obj):
    """
    Recursively remove 'quantization_config' from Keras model config.
    This fixes loading issue:
    Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
    """

    if isinstance(obj, dict):
        obj.pop("quantization_config", None)

        for key, value in obj.items():
            remove_quantization_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


def main():
    if not os.path.exists(SOURCE_MODEL):
        raise FileNotFoundError(f"Source model not found: {SOURCE_MODEL}")

    shutil.copy2(SOURCE_MODEL, FIXED_MODEL)

    with h5py.File(FIXED_MODEL, "r+") as h5file:
        model_config = h5file.attrs.get("model_config")

        if model_config is None:
            raise ValueError("No model_config found inside the .h5 file.")

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        model_config_json = json.loads(model_config)

        remove_quantization_config(model_config_json)

        cleaned_config = json.dumps(model_config_json).encode("utf-8")

        h5file.attrs.modify("model_config", cleaned_config)

    print("Fixed model saved successfully:")
    print(FIXED_MODEL)


if __name__ == "__main__":
    main()