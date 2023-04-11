import xml.etree.ElementTree as ET

# 加载XML文件
tree = ET.parse('my_output_file.xml')

# 获取所有tripinfos元素
tripinfos = tree.getroot()

# 计算waitingTime属性的平均值
waiting_times = [float(tripinfo.get('waitingTime')) for tripinfo in tripinfos]
average_waiting_time = sum(waiting_times) / len(waiting_times)
print(len(waiting_times))
print(sum(waiting_times))
print("平均等待时间：", average_waiting_time)