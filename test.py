from datetime import datetime

time = datetime.now().strftime("%Y%m%d_%H%M%S")
time = f"{time}.{file.filename.split('.')[-1]}"

print(time)