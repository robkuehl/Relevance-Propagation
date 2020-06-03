import plotly.express as px

def plotly_mnist_image(image):
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.Cividis, zmin=-0.15, zmax=0.15)
    fig.show()