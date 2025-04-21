import csv 

def read_csv(path):
   # Tu código aquí 👇
   with open(path, 'r') as csvfile:      
      total = sum(int(r[1]) for r in csv.reader(csvfile))  

   return total

response = read_csv('app/data.csv')
print(response)