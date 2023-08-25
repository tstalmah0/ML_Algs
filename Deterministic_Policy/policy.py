import sys, os

print("out")

input_path = os.path.join(os.path.dirname(__file__), 'input')
print(input_path)
title_text = "test=32"
output = os.path.join(os.path.dirname(__file__), 'output', title_text) + '.png'
print(output)