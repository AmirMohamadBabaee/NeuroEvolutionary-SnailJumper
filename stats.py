from cProfile import label
import matplotlib.pyplot as plt

def draw_stats(file_path='generation_information.gi'):
    total_text = None
    with open(file_path, 'r') as f:
        total_text = f.readlines()
    generation_min = []
    generation_avg = []
    generation_max = []
    for line in total_text:
        new_line = line.replace('[', '').replace(']', '').replace(',', '')
        current_list = list(map(int, new_line.split()))
        min_fitness = min(current_list)
        max_fitness = max(current_list)
        avg_fitness = sum(current_list) / len(current_list)

        generation_min.append(min_fitness)
        generation_avg.append(avg_fitness)
        generation_max.append(max_fitness)
        
    plt.plot(generation_min, label='Min fitness')
    plt.plot(generation_avg, label='Average fitness')
    plt.plot(generation_max, label='Max fitness')
    plt.title('fitness/generation graph')
    plt.xlabel('Generation #')
    plt.ylabel('fitness')
    plt.xticks(range(1, len(generation_avg)+1))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    draw_stats()