
from typing import Any, Dict, Optional, Union
from fairseq.models import FairseqDecoder, FairseqEncoder
from fairseq.file_io import PathManager
from fairseq import utils, checkpoint_utils
from collections import OrderedDict

def load_pretrained_component_from_model_modified(
    component: Union[FairseqEncoder, FairseqDecoder],
    checkpoint: str,
    strict: bool = True,
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    keys_info = component.load_state_dict(component_state_dict, strict=strict)
    return component, keys_info