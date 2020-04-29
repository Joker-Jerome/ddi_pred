import os 

tmp_line_list = []
with open("tuning_task.txt", "r") as fo:
	for line in fo:
		tmp_line = 'export PATH="/home/zy92/anaconda3/bin:$PATH"; ' + line
		tmp_line_list.append(tmp_line)

with open("tuning_task_grace.txt", "w") as fi:
	for line in tmp_line_list:
		fi.write(line)


