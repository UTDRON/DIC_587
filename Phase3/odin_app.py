import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Forms")

# Visualization pane at the top
st.header("Visualization Pane")

# Add a simple plot to the visualization pane
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

# List of buttons on the left
st.sidebar.header("Common Attributes")

# Define some buttons on the left sidebar
button_list = ["Attribute 1", "Attribute 2", "Attribute 3", "Attribute 4"]

# Add buttons to the sidebar
selected_button = st.sidebar.radio("", button_list)

# 3 by 3 matrix of buttons at the bottom
st.subheader("Select Attributes")

# Define a 3 by 3 grid of buttons
button_matrix = np.array([[f"(Attr{i*3+j+1})" for j in range(3)] for i in range(3)])

# Display the buttons in a 3 by 3 grid
cols = st.columns(3)

for i in range(3):
    for j in range(3):
        cols[j].button(button_matrix[i, j])
