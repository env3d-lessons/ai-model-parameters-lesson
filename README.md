# ai-model-parameters-lesson

A series of guided questions to explore how model size, GPU, and quantization.

---

# Q1

Visit https://colab.research.google.com/ (create a google account if necessary).  Start a new notebook (It'll be saved on your drive).  

Click on the drop down button at the top right hand corner and choose "change runtime type".

<img src="./q1.png" alt="Q1: Colab runtime type dropdown" width="500">

What are the available "runtimes" you can choose?

  a. CPU

  b. T4 GPU

  c. A100

  d. L4

---

# Q2

Select the T4 GPU runtime type and connect to it.  Which of the following is true?  Select all that applies:

  a. It has 16G RAM

  b. It has 15G VRAM

  c. You get 2 hours of time to use it

  d. You get over 100G of disk space

---

# Q3

Put the following code in the first cell and run it:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model

device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained("qwen/qwen3-0.6b")
model = AutoModelForCausalLM.from_pretrained("qwen/qwen3-0.6b").to(device)  
```

Select all the TRUE statements:

  a. System RAM is used for the model

  b. GPU RAM is used for the model

  c. The model is downloaded from huggingface

  d. The model size is around 1.5 Gigabytes

---

# Q4

Now enter this code in the 2nd cell below the 1st cell:

```python
# Optional: compile model for PyTorch 2.x (further speedup)
# model = torch.compile(model)

# Input prompt
prompt = "Once upon a time, in a land far, far away,"

# Tokenize input and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,      # set to False for greedy decoding (faster)
        temperature=0.7
    )

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

Run the above cell 3 times.  What was the fastest speed you got out of the above code?

  a. under 10 seconds

  b. 20 to 40 seconds

  c. 40 seconds to 1 minute

  d. Over 1 minute

---

# Q5

Change the device variable in cell one as follows:

```python
device = 'cuda'
```

And rerun cell 1.  Then run cell 2 three times as above.  What is the fastest time to completion for cell 2?

  a. under 10 seconds

  b. 20 to 40 seconds

  c. 40 seconds to 1 minute

  d. Over 1 minute

---

# Q6

Reset your runtime as below and delete all cells (you could also just start a new notebook):

<img src="./q6.png" alt="Q6: Colab runtime reset" width="500">

Put the following code in cell 1:

```python
# Import libraries
import torch
from diffusers import StableDiffusionPipeline

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline (512x512 default)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # basic Stable Diffusion model
    torch_dtype=torch.float16           # use half precision for GPU
).to(device)
```

How much GPU RAM is being used for this model?

  a. Under 1G

  b. Between 1-3G

  c. Between 3-4G

  d. Bigger than 4G

---

# Q7

Enter the following code into cell 2

```python

# Example prompt
prompt = "A futuristic city skyline at sunset, digital art"

image = pipe(prompt, guidance_scale=7.5).images[0]

# Display the generated image in Colab
image
```

How much GPU RAM is being used for this model?

  a. Under 1G

  b. Between 1-3G

  c. Between 3-4G

  d. Bigger than 4G

---

For Q8 to Q10, replace your notebook with the following code.

```python
prompt = "A futuristic city skyline at sunset, digital art"

random_seed = random.randint(0, 2**32 - 1)
generator = torch.Generator(device=device).manual_seed(random_seed)
image = pipe(
    prompt,
    guidance_scale=5.0,
    num_inference_steps=50,
    height=512,
    width=512,
    generator=generator
).images[0]

image
```

---

# Q8

What is the purpose of the random_seed? (Choose all that applies)

  a. Make sure images produced follows the text prompt

  b. Improves generation speed

  c. Controls image size

  d. Make image output deterministic

---

# Q9

What is the purpose of the num_inference_step? (Choose all that applies)

  a. Make sure images produced follows the text prompt

  b. Improves generation speed

  c. Controls image quality

  d. Make image output deterministic

---

# Q10

What is the purpose of the guidance_scale? (Choose all that applies)

  a. Make sure images produced follows the text prompt

  b. Improves generation speed

  c. There is no maximum to the guidance_scale

  d. Make image output deterministic

---

# Q11

Try the following code in Colab:

```python

from ultralytics import YOLO
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw
import requests
import io
from IPython.display import display

# -------------------------------
# 2️⃣ Load models
# -------------------------------

# YOLO segmentation model
yolo_model = YOLO("yolov8n-seg.pt")  # small segmentation model

# Diffusion model for inpainting
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16").to("cuda")

# -------------------------------
# 3️⃣ Load a static image
# -------------------------------
image_path = "http://images.cocodataset.org/val2017/000000039769.jpg"  # replace with your image path
response = requests.get(image_path)
image_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
display(image_pil) # Changed from image_pil.show(title="Original Image")

# -------------------------------
# 4️⃣ Run YOLO detection / segmentation
# -------------------------------
results = yolo_model.predict(image_pil, imgsz=640)  # PIL image is accepted directly

# We'll take the first detected object for simplicity
# Get bounding box of first detected object
if len(results[0].boxes) > 0:
    box = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Create mask (white rectangle over black background)
    mask_pil = Image.new("L", image_pil.size, 0)
    draw = ImageDraw.Draw(mask_pil)
    draw.rectangle([x1, y1, x2, y2], fill=255)

    display(mask_pil)
else:
    print("No objects detected. Exiting.")
    exit()

# -------------------------------
# 5️⃣ Generate edited image via inpainting
# -------------------------------
prompt = "A cute, fluffy, playful cartoon cat, vibrant colors, Pixar style, highly detailed, professional digital art"
# SDXL specific parameters removed as they are not needed for stable-diffusion-2-inpainting
negative_prompt = "blurry, ugly, deformed, text, watermark, bad anatomy, poorly drawn, disfigured, poor quality"

edited_image = pipe(
    prompt=prompt,
    image=image_pil,
    mask_image=mask_pil,
    negative_prompt=negative_prompt
).images[0]
display(edited_image) # Changed from edited_image.show(title="Edited Image")

# -------------------------------
# 6️⃣ Optional: Save results
# # -------------------------------
# image_pil.save("original.png")
# mask_pil.save("mask.png")
# edited_image.save("edited.png")

print("Pipeline complete: original, mask, and edited images saved.")

```

How many models are being used in the above code?

  a. 1

  b. 2

  c. 3

  d. 4