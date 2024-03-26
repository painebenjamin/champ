# Description: This script is used to test the CHAMP pipeline on a single file.
# Usage: python inference.py
#
# This section is only necessary when running the script from the source code.
# It is used to add the source code to the system path.
# If you are running the script from the installed package, you can remove this section.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
# End of section
from PIL import Image
from champ.utils.video_utils import (
    get_pil_from_video,
    save_videos_from_pil
)

# Configurables
num_inference_steps = 20
guidance_scale = 3.5

# Load the input data
depth = get_pil_from_video("./example/depth.mp4")
normal = get_pil_from_video("./example/normal.mp4")
pose = get_pil_from_video("./example/pose.mp4")
segmentation = get_pil_from_video("./example/seg.mp4")
reference = Image.open("./example/reference.jpg")
guidance = {
    "depth": depth,
    "normal": normal,
    "dwpose": pose,
    "semantic_map": segmentation,
}

# Initialize the pipeline
import torch
from champ import CHAMPPipeline

#pipeline = CHAMPPipeline.from_pretrained(
#    "benjamin-paine/champ",
#    torch_dtype=torch.float16,
#    variant="fp16",
#    device="cuda",
#).to("cuda", dtype=torch.float16)

## Alternatively, you can use the following code to load the pipeline from a single file
## Substitute the repository with the path to the file to load locally
pipeline = CHAMPPipeline.from_single_file(
    "benjamin-paine/champ",
    torch_dtype=torch.float16,
    variant="fp16",
    device="cuda",
).to("cuda", dtype=torch.float16)

# The example videos are already standardized to the same size and length
width, height = depth[0].size
num_frames = len(depth)

# Execute it
output = pipeline(
    reference,
    guidance,
    width,
    height,
    num_frames,
    num_inference_steps,
    guidance_scale
).videos

# Save the output
save_videos_from_pil(
    output,
    "./output.mp4",
    fps=8
)
