import xml.etree.ElementTree as ET
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

# Define a function to parse the XML and extract data
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract points and their categories
    points = {}
    for point in root.findall('point'):
        name = point.get('name')
        x = float(point.get('xPosition')) / 1e3 # [m]
        y = float(point.get('yPosition')) / 1e3 # [m]
        type_ = point.get('type', 'UNKNOWN')
        points[name] = {'x': x, 'y': y, 'type': type_}
    
    # Extract paths
    paths = []
    for path in root.findall('path'):
        source = path.get('sourcePoint')
        destination = path.get('destinationPoint')
        paths.append((source, destination))

    # Extract vehicles
    vehicles = []
    for vehicle in root.findall('vehicles'):
        name = vehicle.get('name')
        length = vehicle.get('length')
        vehicles.append((name, length))
    
    return points, paths, vehicles

def parse_log(file_path):
    with open(file_path) as f:
        content = f.read()
    lines = content.split('\n')

    # filter the inital positions
    init_pos_lines = [line for line in lines if '[DEBUG] Vehicle-' in line and 'conn_ack' not in line]
    # [DEBUG] Vehicle-01001: Sending: position;Q0021'  # example
    init_pos = {}
    for i in init_pos_lines:
        name = i.split('] ')[1].split(':')[0]
        if name not in init_pos:
            init_pos[name] = i.split(';')[-1]
    
    print('Initial Positions',init_pos)
    # filter the history positions
    history_lines = [line for line in lines if ' [DEBUG] time=' in line]
    #  [DEBUG] time=24411;move;vehicle=Vehicle-01001;edge=N0017 --- S0020;distance_ahead=1.883000
    history = {}
    for i in history_lines:
        time = i.split('] ')[1].split(';')[0].split('=')[1]
        name = i.split(';')[2].split('=')[1]
        origin = i.split(';')[3].split(' --- ')[0].split('=')[1]
        target = i.split(';')[3].split(' --- ')[1]
        distance = i.split(';')[4].split('=')[1]
        history[time] = [name,origin,target,distance]

    print('History',history)

    return init_pos, history


# Function to compute intermediate positions
def interpolate_position(origin, target, distance, node_loc):
    x1, y1 = node_loc[origin]
    x2, y2 = node_loc[target]
    dx, dy = x2 - x1, y2 - y1
    distance_total = np.sqrt(dx**2 + dy**2)
    if distance_total == 0:
        return x1, y1  # Avoid division by zero for zero-length paths
    factor = 1 - (float(distance) / distance_total)
    x = x1 + factor * dx
    y = y1 + factor * dy
    return x, y

# Function to animate the history events
def animate_history(points, paths, init_pos, history):
    # Plot the structure
    category_colors = {
        'HALT_POSITION': 'lightskyblue',
        'PARK_POSITION': 'teal',
    }

    fleet_colors = {
        1: 'red',
        2: 'tab:orange',
        3: 'magenta',
        4: 'gold',
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    node_loc = {}

    for name, point in points.items():
        x, y = point['x'], point['y']
        type_ = point['type']
        color = category_colors.get(type_, 'black')  # Default to black if type is unknown
        ax.plot(x, y, 'o', color=color, markeredgecolor='black', zorder=5)
        ax.text(x-0.200, y+0.500, name, fontsize=8, ha='right', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=10)
        node_loc[name] = (x, y)

    for source, destination in paths:
        if source in points and destination in points:
            x1, y1 = points[source]['x'], points[source]['y']
            x2, y2 = points[destination]['x'], points[destination]['y']
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                        arrowprops=dict(arrowstyle="->", color="black", lw=1, alpha=1))

    # Create vehicle markers
    vehicle_obj = {}
    for vehicle, node in init_pos.items():
        x, y = node_loc[node]
        fleet_number = int(vehicle.split('Vehicle-')[1][0:2])
        color = fleet_colors[fleet_number]
        marker = ax.plot(x, y, 's', color=color, markeredgecolor='black', zorder=20)[0]
        label = ax.text(x-0.200, y+0.500, vehicle.split('Vehicle-')[1], fontsize=8, ha='right',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1), zorder=20)
        vehicle_obj[vehicle] = [marker, label]

    # Parse history by time
    sorted_times = sorted(history.keys(), key=lambda t: int(t))
    vehicle_positions = {vehicle: init_pos[vehicle] for vehicle in init_pos.keys()}  # Current positions

    def update(frame):
        time = sorted_times[frame]
        events = [h for t, h in history.items() if t == time]
        
        for event in events:
            name, origin, target, distance = event
            x, y = interpolate_position(origin, target, distance, node_loc)
            
            # Update vehicle marker and label
            if name in vehicle_obj:
                marker, label = vehicle_obj[name]
                marker.set_data([x], [y])
                label.set_position((x-0.200, y+0.500))
                vehicle_positions[name] = target  # Update current position

        ax.set_title(f'Time: {time}', fontsize=12)

    # plot settings
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=5)
    #ax.set_xlabel('X Position')
    #ax.set_ylabel('Y Position')
    # ax.grid(True, which='both')
    # ax.grid(True, which='major')

    fig.tight_layout()
    #plt.savefig('map.png', dpi=300, bbox_inches='tight', pad_inches=0)

    ani = animation.FuncAnimation(fig, update, frames=len(sorted_times), interval=500, repeat=True)

    # Save the animation or display
    plt.show()

# Update the main function to call the animation
def main():
    parser = argparse.ArgumentParser(description="Visualisation of logistic network.")
    parser.add_argument('map_path', type=str, help="Path to the XML file")
    parser.add_argument('log_path', type=str, help="Path to the Log file")
    args = parser.parse_args()
    
    points, paths, vehicles = parse_xml(args.map_path)
    init_pos, history = parse_log(args.log_path)
    
    animate_history(points, paths, init_pos, history)

if __name__ == '__main__':
    main()
