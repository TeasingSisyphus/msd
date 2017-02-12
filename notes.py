import tkinter as tk


def gui():
	global top 
	top = tk.Tk()
	L1 = tk.Label(top, text="Minimum Particle Size:").grid(row=0)
	# L1.pack()
	L2 = tk.Label(top, text="Minimum Frames per Particle:").grid(row=1)
	# L2.pack()
	E1 = tk.Entry(top, bd=5)
	E1.grid(row=0, column=1)
	# E1.pack()
	E2 = tk.Entry(top, bd=5)
	E2.grid(row=1, column=1)
	# E2.pack()
	B1 = tk.Button(top, text = "Analyze", command=lambda: printer(E1))
	B1.grid(row=2)
	# B1.pack()
	# B2 = tk.Button(top, text = "e2", command=lambda: printer(E2))
	# B2.pack()
	B3 = tk.Button(top, text = "Cancel", command = ender)
	B3.grid(row=2, column=1)
	# B3.pack()
	top.mainloop()

def printer(entry):
	print("got to the printer " + entry.get())

def ender():
	print("destroy")
	top.destroy()

# gui()