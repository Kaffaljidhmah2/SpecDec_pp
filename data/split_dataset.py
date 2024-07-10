import json
import os
import random

def process(filename, num_train, num_dev, num_test):
    data = json.load(open(filename, 'r'))
    if num_train + num_dev + num_test != len(data):
        print(f"warning: the number of total datapoints {len(data)} does not match  {num_train} + {num_dev} + {num_test}.")
    
    random.shuffle(data)
    
    train_data = data[:num_train]
    dev_data = data[num_train:num_train + num_dev]
    test_data = data[num_train + num_dev:num_train + num_dev + num_test]
    
    # Save the splits into separate files
    base_dir = base_dir = os.path.dirname(filename) or '.'
    
    with open(f'{base_dir}/train.json', 'w') as train_file:
        json.dump(train_data, train_file, indent=2)
    
    with open(f'{base_dir}/dev.json', 'w') as dev_file:
        json.dump(dev_data, dev_file, indent=2)
    
    with open(f'{base_dir}/test.json', 'w') as test_file:
        json.dump(test_data, test_file, indent=2)

if __name__ == "__main__":
    import sys

    all_file = sys.argv[1]
    num_train = int(sys.argv[2])
    num_dev = int(sys.argv[3])
    num_test = int(sys.argv[4])
    process(all_file, num_train, num_dev, num_test)

