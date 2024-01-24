input_data = [2752,2685,2694,2666,2689]

# 第一个数据作为除数
divisor = input_data[0]

# 对所有数据进行除法操作
output_data = [value / divisor for value in input_data]

print("Input Data:", input_data)
print("Output Data:", output_data)