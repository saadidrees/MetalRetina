#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:16:48 2022

@author: saad
"""
import numpy as np


def extractCodeSnippet(fname_script,txt_start,txt_end):
    string1 = (txt_start,txt_end)
    
    f = open(fname_script, 'r')
      
    flag = np.zeros(len(string1))
    l_num = np.zeros(len(string1),'int32')
    ctr_line = -1
    
    for line in f:  
        ctr_line += 1 
          
        # checking string is present in line or not
        for s in range(len(string1)):
            if string1[s] in line:
              flag[s] = 1
              l_num[s] = ctr_line
              # break 
    f.close() 
    
    if np.any(l_num==0):
        code_snippet = 'Cannot find code snippet'
        
    else:
        f = open(fname_script, 'r')
        code_snippet = f.readlines()
        code_snippet = code_snippet[l_num[0]:l_num[1]+1]
        code_snippet = ' '.join(code_snippet)
      
        f.close() 

    return code_snippet


# %% Write params and hyper params to text file
def dictToTxt(params_txt,fname_paramsTxt,f_mode='a'):
    fo = open(fname_paramsTxt,f_mode)

    if params_txt.__class__.__name__ == 'Functional':
        params_txt.summary(print_fn=lambda x: fo.write(x+'\n'))
    else:
        for k, v in params_txt.items():
            fo.write(str(k) + ' = '+ str(v) + '\n')
    
    fo.close()
    
    
def read_config(file_path, keys):
    values = {}
    with open(file_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key in keys:
                    try:
                        values[key] = eval(value)  # Safely convert numbers/bools
                    except:
                        values[key] = value  # Keep as string if eval fails
    return values

