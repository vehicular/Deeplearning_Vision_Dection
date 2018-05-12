#!/usr/bin/evn python 
#coding:utf-8 
import os
import glob
import pandas as pd
try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 

file_srx = glob.glob('train/*.xml')
xml_list = []
for line in file_srx:
    tree = ET.parse(line)
    root = tree.getroot()         
    
    for member in root.findall('object'):
        value = (root.find('filename').text,
             int(root.find('size')[0].text),
             int(root.find('size')[1].text),
             member[0].text,
             int(member[4][0].text),
             int(member[4][1].text),
             int(member[4][2].text),
             int(member[4][3].text)
             )
        xml_list.append(value)      

column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
print(xml_df)
xml_df.to_csv('train.csv', index=None)
    
  