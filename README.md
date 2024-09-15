# Diffusion_XAI  
Diverse and Creative Image Generation by automatically identifying the representations in the Text-to-Image Diffusion Model.


### Description  
Despite recent advancements, generating or editing images with text-to-image generation models often struggles to produce desired outcomes due to solid correlations embedded in the training data. While AI models are expected to generate novel and creative objects, current models tend to generate attributes like color, shape, and form that closely resemble reality, leading to cliched results. This issue extends to editing, where strong object-attribute associations (e.g., editing a red apple into a blue one) hinder achieving the intended modifications. Here, we aim to break these limitations by selectively enhancing or removing certain attributes, generating creative results from a human perspective. Specifically, we automatically detect internal feature maps representing specific attributes for a given prompt and manipulate these features to break conventional generation rules.  


### Example
```python
import torch
import pickle
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler
from matplotlib import gridspec

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id,requires_safety_checker=False,safety_checker=None).to('cuda')

with open("/channels/red_objects_channels.pkl", "rb") as f:
    channels = pickle.load(f)

out = pipe("An apple.",ablate_channels=channels)[0]
image = out.images[0]
plt.imshow(image)
plt.show()
```

  
<p align="center">
    <img src="assets/apple_ablations.jpg" width="600"> 
</p>

### Acknowledgement
```
This work was supported by KAIST-NAVER Hypercreative AI Center.
```
