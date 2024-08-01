from fastai.vision.all import *
def is_cat(x):
    return x[0].isupper()

# Load the learner
learn = load_learner('model.pkl')

categories = ('Dog','Cat')
def classify_image(img):
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

#gradio interface
import gradio as gr
from gradio import Interface, Image as grImage, Label
from PIL import Image

# Create input and output components
image = gr.Image()
label = gr.Label()
examples = ['siamese.jpg']
# Create Gradio interface
intf = Interface(fn=classify_image, inputs=image, outputs=label, title="CAT and DOG Classifier", examples = examples)

intf.launch(inline=False)