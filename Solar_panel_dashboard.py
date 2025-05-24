
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image


st.title('🌞 Smart Solar Panel Inspection')
st.image((r'C:\Users\lenovo\Downloads\solar-energy-drones.jpg'))
st.sidebar.subheader("Discover")
options=st.sidebar.selectbox('',['Home','Inspection'])

if options =='Home':
  st.subheader('Welcome to the Smart Solar Panel Inspection app! ')
  st.write('This tool uses Artificial Intelligence (AI) to automatically classify various obstructions or damages on solar panels, including dust, snow,bird droppings,electrical-damage,physical-damage.')
  st.write('Maintaining clean and damage-free solar panels is crucial for maximizing energy efficiency.')
  st.subheader('🔍Features:')
  st.write('- Upload images of solar panels')
  st.write('- Instantly detect and classify issues')
if options =='Inspection':
  st.subheader("Upload a Solar Panel Image")
  uploaded_file = st.file_uploader('', type=["jpg", "jpeg", "png"])
  from PIL import Image
  if uploaded_file :
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    predict = st.button('Classify')
    if predict:
      # Load model
      model = models.resnet18(pretrained=False)
      num_classes = 6
      model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
      model.load_state_dict(torch.load(r"C:\Users\lenovo\Downloads\resnet18_2.pth", map_location=torch.device('cpu')))
      model.eval()

      # Class names
      class_names = ["Bird-drop","Clean","Dusty","Electrical-damage","Physical-Damage","Snow-Covered"]

      # Load and preprocess image
      transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
      img = Image.open(uploaded_file)
      img = transform(img).unsqueeze(0)

      # Predict
      outputs = model(img)
      _, predicted = torch.max(outputs, 1)

      # Output
      st.header('Predicted Class :')
      outputs=class_names[predicted.item()]
      st.subheader(outputs)#'Predicted class:', 

      if outputs=='Physical-Damage':
        st.subheader('Solution')
        st.write('Small cracks: apply UV-stable clear epoxy to prevent moisture ingress (temporary).')
        st.write('Major damage: Replace the panel.')
        st.write('Install protective mesh or low-impact shields in areas prone to hail or tree branches.')

      elif outputs == 'Bird-drop':
        st.subheader('Solution')
        st.write('Use a soft sponge with warm water or panel-safe biodegradable cleaning solution.')
        st.write('Avoid abrasive materials (they scratch the panel).')
        st.write('Use a long-handle non-metallic squeegee if the panel is on a roof.')
        st.write('Consider installing bird deterrents like spikes or netting.')

      elif outputs == 'Dusty':
        st.subheader('Solution')
        st.write('Clean with deionized water and a soft cloth or sponge.')
        st.write('For large-scale farms: use robotic cleaners or semi-automated washing arms.')
        st.write('Clean every 3–6 months depending on environment (more often in deserts).')

      elif outputs == 'Electrical-damage':
        st.subheader('Solution')
        st.write('Turn off the system before inspection!')
        st.write('Replace damaged junction boxes.')
        st.write('Re-solder broken connections.')
        st.write('Replace bypass diodes or rewiring if necessary')

      elif outputs == 'Snow-Covered':
        st.subheader('Solution')
        st.write('Use a soft roof rake with foam head — never use metal tools!.')
        st.write('Let sunlight melt thin snow (panels are dark and get warm)')
        st.write('Install tilted panels or snow guards for better shedding')
        st.write('For persistent snow areas: consider panel heating elements or hydrophobic coating')

st.sidebar.subheader('Model')
model = st.sidebar.button('ResNet18')
if model:
  st.subheader('About the model and Evaluation metrics')
  st.write('ResNet-18 is a lightweight and powerful deep learning model ideal for image classification tasks.')
  st.write('In this project, ResNet-18 is used to detect various types of damages and obstructions on solar panels.')
  st.write('It provides high accuracy while keeping the model size and training time efficient.')
  st.subheader('Classified Image dataset')
  st.image(r"C:/Users/lenovo/Pictures/Screenshots/Screenshot (88).png")
  st.subheader('Sample Images')
  st.image(r"C:/Users/lenovo/Pictures/Screenshots/Screenshot (87).png")
  st.subheader('Model Evaluation Motrics')
  st.image(r"C:/Users/lenovo/Pictures/Screenshots/Screenshot (86).png")
  st.subheader('Accuracy and Loss')
  st.image(r"C:/Users/lenovo/Pictures/Screenshots/Screenshot (92).png")
  st.subheader('Confusion Matrix')
  st.image(r"C:/Users/lenovo/Downloads/confusion_matrix.jpg")