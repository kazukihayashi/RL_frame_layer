import colorsys
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy as np

def Draw(node, connectivity, node_color, line_width, line_color, node_text=None, line_text=None, scale=0.01, name=0):
	"""
	node[nk][3]or[nk][2]:(float) nodal coordinates
	connectivity[nm][2]	:(int)   connectivity to define member
	node_color[nk]		:(str)   color of node
	line_width[nm]		:(float) size of member
	line_color[nm]		:(str)   color of member
	"""
	##if len(connectivity) != len(section) or len(connectivity) != len(stress):
	##	raise ValueError("The size of connectivity, section, and stress must be equal.")
	
	node = np.array(node, dtype = float)
	connectivity = np.array(connectivity, dtype = int)
	line_width = np.array(line_width, dtype = float)
	
	fig = pyplot.figure()

	if len(node[0]) == 2:
		
		ax = pyplot.subplot()
		for i in range(len(connectivity)):
			line = Line2D([node[connectivity[i,0],0],node[connectivity[i,1],0]],[node[connectivity[i,0],1],node[connectivity[i,1],1]],linewidth=line_width[i]*scale,color=line_color[i])
			ax.add_line(line)
			if line_text is not None:
				ax.text((node[connectivity[i,0],0]+node[connectivity[i,1],0])/2,(node[connectivity[i,0],1]+node[connectivity[i,1],1])/2,line_text[i],horizontalalignment='center', verticalalignment='center',fontsize=4,color='White',bbox=dict(boxstyle='square,pad=0',fc='Black'))
		for i in range(len(node)):
			ax.plot([node[i,0]],[node[i,1]], "o", color=node_color[i], ms=4)
			if node_text is not None:
				ax.text(node[i,0]+0.7,node[i,1]+0.7,node_text[i],fontsize=4, color='Green')
		pyplot.xlim([np.min(node[:,0]),np.max(node[:,0])])
		pyplot.ylim([np.min(node[:,1]),np.max(node[:,1])])
		pyplot.tick_params(labelbottom="off",bottom="off",labelleft="off",left="off")
		pyplot.axis('scaled')
		pyplot.axis('off')
		if type(name) is int:
			pyplot.savefig(r'result\{0:0=2}.png'.format(name),dpi=300)
		elif type(name) is str:
			pyplot.savefig(r'result\{0}.png'.format(name),dpi=300)
		pyplot.close(fig)

	else:
		raise TypeError("node must eliminate z coordinates.")

	return fig

def graph(y,name):
	x = np.linspace(0,len(y)-1,len(y)).astype(int)
	pyplot.figure(figsize=(10,4))
	pyplot.plot(x,y,linewidth=1)
	xt = np.linspace(0,len(y)-1,11).astype(int)
	pyplot.xticks(xt,xt)

	# save figure
	pyplot.savefig(r"result\graph("+str(name)+").png")
	##pyplot.show()
	pyplot.close()

