from xml.etree import ElementTree
import pprint, sys

with open(sys.argv[1], 'rt') as f:
	tree = ElementTree.parse(f)

for s in tree.findall('.//p/s'):
	for w in s.findall('.//'):
		if w.tag == 'w':
			print(w.text, end = '')
	
	print()
