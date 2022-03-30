from email.mime import image
from fileinput import filename
import streamlit as slt
import pandas as pd
import os
import pickle
from PIL import Image
from math import exp, tanh
from tokenize import Exponent


image = Image.open(r'C:\\Users\ALPHA-Z\Desktop\\Masters 2 Project\\Image1.jpg')
slt.image(image, caption = 'Cancercarer')
slt.title('Breast Cancer Detection App')
slt.subheader('This prediction was made using the UCI Wisconsin Machine Learning Repository dataset. The digitalized image of a fine needle aspiration of biopsied cell nuclei was used to set the value of the observed attribute to view the likely classification - given the 661 prior cases')

clump_thickness = int(slt.slider('clump thickness', 1, 10))
cell_size_uniformity = int(slt.slider('cell size uniformity', 1, 10))
cell_shape_uniformity = int(slt.slider('cell shape uniformity', 1, 10))
marginal_adhesion = int(slt.slider('marginal adhesion', 1, 10))
single_epithelial_cell_size = int(slt.slider('single epithelial cell size', 1, 10))
bare_nuclei = int(slt.slider('bare_nuclei', 1, 10))
bland_chromatin = int(slt.slider('bland chromatin', 1, 10))
normal_nucleoli = int(slt.slider('normal nucleoli', 1, 10))
mitoses = int(slt.slider('mitoses', 1, 10))
data = {'clump thickness': clump_thickness,
        'cell size uniformity': cell_size_uniformity,
        'cell shape uniformity': cell_shape_uniformity,
        'marginal adhesion': marginal_adhesion,
        'single epithelial cell size': single_epithelial_cell_size,
        'bare nuclei': bare_nuclei,
        'bland chromatin': bland_chromatin,
        'normal nucleoli': normal_nucleoli,
        'mitoses': mitoses}

scaled_clump_thickness = (int(clump_thickness)-4.442170143)/2.818700075
scaled_cell_size_uniformity = (int(cell_size_uniformity)-3.150810003)/3.062900066
scaled_cell_shape_uniformity = (int(cell_shape_uniformity)-3.215229988)/2.986390114
scaled_marginal_adhesion = (int(marginal_adhesion)-2.830159903)/2.862469912
scaled_single_epithelial_cell_size = (int(single_epithelial_cell_size)-3.234260082)/2.221460104
scaled_bare_nuclei = (int(bare_nuclei)-3.544660091)/3.641190052
scaled_bland_chromatin = (int(bland_chromatin)-3.445100069)/2.447900057
scaled_normal_nucleoli = (int(normal_nucleoli)-2.869689941)/3.050430059
scaled_mitoses = (int(mitoses)-1.603219986)/1.731410027

perceptron_layer_1_output_0 = tanh( 0.0344199 + (scaled_clump_thickness*0.181135) + (scaled_cell_size_uniformity*0.217282) + (scaled_cell_shape_uniformity*0.196496) + (scaled_marginal_adhesion*0.125525) + (scaled_single_epithelial_cell_size*-0.0287599) + (scaled_bare_nuclei*0.314836) + (scaled_bland_chromatin*0.0702443) + (scaled_normal_nucleoli*0.18966) + (scaled_mitoses*0.185792) );
perceptron_layer_1_output_1 = tanh( -0.0365313 + (scaled_clump_thickness*-0.185626) + (scaled_cell_size_uniformity*-0.224576) + (scaled_cell_shape_uniformity*-0.203225) + (scaled_marginal_adhesion*-0.127989) + (scaled_single_epithelial_cell_size*0.029879) + (scaled_bare_nuclei*-0.321838) + (scaled_bland_chromatin*-0.0715547) + (scaled_normal_nucleoli*-0.194488) + (scaled_mitoses*-0.191226) );
perceptron_layer_1_output_2 = tanh( -0.0362404 + (scaled_clump_thickness*-0.184876) + (scaled_cell_size_uniformity*-0.223219) + (scaled_cell_shape_uniformity*-0.202266) + (scaled_marginal_adhesion*-0.127906) + (scaled_single_epithelial_cell_size*0.0295682) + (scaled_bare_nuclei*-0.321034) + (scaled_bland_chromatin*-0.0711169) + (scaled_normal_nucleoli*-0.193701) + (scaled_mitoses*-0.190348) );
perceptron_layer_1_output_3 = tanh( -0.0361874 + (scaled_clump_thickness*-0.18509) + (scaled_cell_size_uniformity*-0.223665) + (scaled_cell_shape_uniformity*-0.202517) + (scaled_marginal_adhesion*-0.127643) + (scaled_single_epithelial_cell_size*0.0299014) + (scaled_bare_nuclei*-0.320942) + (scaled_bland_chromatin*-0.0715655) + (scaled_normal_nucleoli*-0.194093) + (scaled_mitoses*-0.190596) );
perceptron_layer_1_output_4 = tanh( 0.0360132 + (scaled_clump_thickness*0.184484) + (scaled_cell_size_uniformity*0.222643) + (scaled_cell_shape_uniformity*0.201691) + (scaled_marginal_adhesion*0.12739) + (scaled_single_epithelial_cell_size*-0.0299674) + (scaled_bare_nuclei*0.320025) + (scaled_bland_chromatin*0.0714402) + (scaled_normal_nucleoli*0.193439) + (scaled_mitoses*0.189759) );
perceptron_layer_1_output_5 = tanh( 0.0354705 + (scaled_clump_thickness*0.183133) + (scaled_cell_size_uniformity*0.2203) + (scaled_cell_shape_uniformity*0.199823) + (scaled_marginal_adhesion*0.126653) + (scaled_single_epithelial_cell_size*-0.0295373) + (scaled_bare_nuclei*0.317894) + (scaled_bland_chromatin*0.0708924) + (scaled_normal_nucleoli*0.191821) + (scaled_mitoses*0.188106) );
perceptron_layer_1_output_6 = tanh( 0.0348187 + (scaled_clump_thickness*0.181724) + (scaled_cell_size_uniformity*0.218308) + (scaled_cell_shape_uniformity*0.197597) + (scaled_marginal_adhesion*0.125867) + (scaled_single_epithelial_cell_size*-0.0290229) + (scaled_bare_nuclei*0.315884) + (scaled_bland_chromatin*0.0705261) + (scaled_normal_nucleoli*0.19041) + (scaled_mitoses*0.186599) );
perceptron_layer_1_output_7 = tanh( -0.0355217 + (scaled_clump_thickness*-0.183151) + (scaled_cell_size_uniformity*-0.220295) + (scaled_cell_shape_uniformity*-0.199629) + (scaled_marginal_adhesion*-0.126905) + (scaled_single_epithelial_cell_size*0.0293504) + (scaled_bare_nuclei*-0.318226) + (scaled_bland_chromatin*-0.070681) + (scaled_normal_nucleoli*-0.19183) + (scaled_mitoses*-0.188182) );
perceptron_layer_1_output_8 = tanh( 0.0349867 + (scaled_clump_thickness*0.18201) + (scaled_cell_size_uniformity*0.21809) + (scaled_cell_shape_uniformity*0.197321) + (scaled_marginal_adhesion*0.126417) + (scaled_single_epithelial_cell_size*-0.029256) + (scaled_bare_nuclei*0.316366) + (scaled_bland_chromatin*0.0703615) + (scaled_normal_nucleoli*0.190555) + (scaled_mitoses*0.186607) );
perceptron_layer_1_output_9 = tanh( 0.0344097 + (scaled_clump_thickness*0.181045) + (scaled_cell_size_uniformity*0.217696) + (scaled_cell_shape_uniformity*0.196321) + (scaled_marginal_adhesion*0.125367) + (scaled_single_epithelial_cell_size*-0.0283902) + (scaled_bare_nuclei*0.314796) + (scaled_bland_chromatin*0.0700932) + (scaled_normal_nucleoli*0.189455) + (scaled_mitoses*0.185805) );

probabilistic_layer_combinations_0 = -0.0355815 +(0.58858 * perceptron_layer_1_output_0) - (0.605611 * perceptron_layer_1_output_1) - (0.603077 * perceptron_layer_1_output_2) - (0.603653 * perceptron_layer_1_output_3) + (0.601335 * perceptron_layer_1_output_4) + (0.596165 * perceptron_layer_1_output_5) + (0.591153 * perceptron_layer_1_output_6) - (0.596392 * perceptron_layer_1_output_7) + (0.59155 * perceptron_layer_1_output_8) + (0.588449 * perceptron_layer_1_output_9); 

diagnose = 1.0/(1.0 + (-exp(probabilistic_layer_combinations_0)));

if diagnose > 0.5:
    slt.header('The  tumor is most likely non-cancerous')
else:
    slt.header('The tumor is most likely cancerous')