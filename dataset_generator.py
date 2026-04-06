import csv
import random

def generate_sample_trace(filename="sample_trace.csv", num_requests=500):
    nodes = ["S1", "S2", "S3", "S4", "S5", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['time', 'src', 'dst', 'holding_time', 'bit_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        current_time = 0.0
        for i in range(num_requests):
            src, dst = random.sample(nodes, 2)
            holding_time = random.expovariate(1.0)
            bit_rate = random.choice([100.0, 200.0, 400.0])
            
            writer.writerow({
                'time': current_time,
                'src': src,
                'dst': dst,
                'holding_time': holding_time,
                'bit_rate': bit_rate
            })
            
            current_time += random.expovariate(2.0) # inter-arrival time

if __name__ == "__main__":
    generate_sample_trace()
    print("Generated sample_trace.csv")
