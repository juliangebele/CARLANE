import os
import json


def remove_empty_lanes(data_root, input_file):
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()
        new_file = open(os.path.join(data_root, input_file.split('.')[0] + '_new.json'), 'w')
        print(f"[{input_file}]: {len(lines)} total lines")

        total_operations = 0
        for i, line in enumerate(lines):
            line = json.loads(line)
            lanes_to_remove = []
            for j, lane in enumerate(line['lanes']):
                if lane.count(-2) == len(lane):
                    lanes_to_remove.append(j)

                if j == len(line['lanes']) - 1:
                    for lane_i in lanes_to_remove[::-1]:
                        total_operations += 1
                        line['lanes'].pop(lane_i)

            line = json.dumps(line)
            new_file.write(line + '\n')

    new_file.close()
    print(f"[{input_file}]: {total_operations} total operations\n")


if __name__ == '__main__':
    root_path = './MoLane/splits'
    files = ['source_train.json', 'source_val.json', 'target_val.json', 'target_test.json']
    for file_name in files:
        remove_empty_lanes(data_root=root_path, input_file=file_name)
