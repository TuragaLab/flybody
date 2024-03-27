"""Optional script for adding id labels __flyid__ to fly model components."""

import os
from lxml import etree

from dm_control import mjcf
import flybody


# The output XML of this script will be saved in this file.
output_xml_file = 'fruitfly-with-ids.xml'

# Load the original fly model.
flybody_path = os.path.dirname(flybody.__file__)
input_xml_path = os.path.join(flybody_path, 'fruitfly/assets/fruitfly.xml')
mjcf_model = mjcf.from_path(input_xml_path)
mjcf_model.find('joint', 'free').remove()

# Substitute label __flyid__ for original model name.
mjcf_model.model = '__flyid__'
# Generate new model.
arena = mjcf.RootElement()
attachment_frame = arena.attach(mjcf_model)
attachment_frame.add('freejoint')

# XML cleanup, almost the same as in make_fruitfly.py.
print('XML cleanup.')
xml_string = arena.to_xml_string('float', precision=3, zero_threshold=1e-7)

root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))
default_elem = root.find('default')
root.insert(3, default_elem[1])  # This line is different from make_fruitfly.py.
root.remove(default_elem)

print('Remove hashes from filenames.')
meshes = [mesh for mesh in root.find('asset').iter() if mesh.tag == 'mesh']
for mesh in meshes:
    name, extension = mesh.get('file').split('.')
    mesh.set('file', '.'.join((name[:-41], extension)))

print('Get string from lxml.')
xml_string = etree.tostring(root, pretty_print=True)
xml_string = xml_string.replace(
    b' class="__flyid__/"', b'')  # This line is different from make_fruitfly.py.

print('Remove gravcomp="0".')
xml_string = xml_string.replace(b' gravcomp="0"', b'')

print('Insert spaces between top level elements.')
lines = xml_string.splitlines()
newlines = []
for line in lines:
    newlines.append(line)
    if line.startswith(b'  <'):
        if line.startswith(b'  </') or line.endswith(b'/>'):
            newlines.append(b'')
newlines.append(b'')
xml_string = b'\n'.join(newlines)

print(f'Save {output_xml_file} to file.')
with open(output_xml_file, 'wb') as f:
    f.write(xml_string)

print('Done.')
