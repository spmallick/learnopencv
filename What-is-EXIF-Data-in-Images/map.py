import exif
import tkinter
from PIL import Image, ImageTk
from tkintermapview import TkinterMapView


def decimal(val):
	dec = val[0] + val[1]/60 + val[2]/3600
	return dec


def get_location(img):
	img = exif.Image(img)
	lat = img.gps_latitude
	lon = img.gps_longitude
	# Convert Degree, min, sec to decimal.
	lat = decimal(lat)
	lon = decimal(lon)
	return lat, lon


def mark_pothole(img):
	pothole = ImageTk.PhotoImage(Image.open(img).resize((320, 200)))
	lat, lon = get_location(img)
	marker = map_widget.set_marker(lat, lon, 
		text='POTHOLE', image=pothole)
	marker.image_zoom_visibility=(19, 22)
	marker.hide_image(False)



root_tk = tkinter.Tk()
root_tk.geometry(f"{1280}x{720}")
root_tk.title("Pothole Visualization MAP")

# create map widget
map_widget = TkinterMapView(root_tk, 
	width=600, 
	height=400, 
	corner_radius=0)

map_widget.pack(fill="both", expand=True)

# google normal tile server
map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga", 
	max_zoom=22)

map_widget.set_position(13.014493, 77.634619)
map_widget.set_zoom(15)
map_widget.set_address("Kamanahalli, Bangalore", marker=True)


mark_pothole('potholes/pothole3.jpg')
mark_pothole('potholes/pothole6.jpg')
mark_pothole('potholes/pothole9.jpg')
mark_pothole('potholes/pothole7.jpg')


root_tk.mainloop()
