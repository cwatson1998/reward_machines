from envs.grids.game_objects import Actions
import random, math, os
import numpy as np
from collections import defaultdict


class OfficeWorld:

    def __init__(self, randomization=None, seed=0, observation_type="agent", render_mode="rgb_array"):

        
        #print(f"DEBUG OfficeWorld: randomization = {randomization}, seed = {seed}")
        #print(f"DEBUG OfficeWorld: received args = randomization={randomization}, seed={seed}")
        self.render_mode = render_mode
        self.randomization = randomization
        if randomization is not None:
            self.rng = np.random.RandomState(seed)
        self.observation_type = observation_type
        self._load_map()
        self.map_height, self.map_width = 12,9
        

    def reset(self):
        
        self.agent = (2,1)
        if self.randomization == "objects":
            self._load_map()
        

        #if self.randomization is None:
        #    self.agent = (2,1)
        #else:
        #    raise NotImplementedError("Unimplemented")

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        
        x,y = self.agent
        self.agent = self._get_new_position(x,y,a)
        # if self.coffee_interval is not None:
            # 1. Tick all existing coffees.
            #for k in self.coffee_clocks:
            #    self.coffee_clock[k] += 1
            # 2. Update coffee state for first visit.
            #if self.objects(self.agent) == 'f'
        #     raise NotImplementedError("unimplemented")

            


    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x,y,action) not in self.forbidden_transitions:
            if action == Actions.up   : y+=1
            if action == Actions.down : y-=1
            if action == Actions.left : x-=1
            if action == Actions.right: x+=1
        return x,y


    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        agent_x,agent_y = self.agent
        if self.observation_type == "agent":
            return np.array([agent_x,agent_y])
        elif self.observation_type == "agent_objects":
            features = [agent_x, agent_y]
            raise NotImplementedError("Unimplemented")
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

    def show(self):
        lines = []
        for y in range(8,-1,-1):
            if y % 3 == 2:
                line = ""
                for x in range(12):
                    if x % 3 == 0:
                        line += "_"
                        if 0 < x < 11:
                            line += "_"
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        line += "_"
                    else:
                        line += " "
                lines.append(line)
            
            line = ""
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    line += "|"
                elif x % 3 == 0:
                    line += " "
                if (x,y) == self.agent:
                    line += "A"
                elif (x,y) in self.objects:
                    line += self.objects[(x,y)]
                else:
                    line += " "
                if (x,y,Actions.right) in self.forbidden_transitions:
                    line += "|"
                elif x % 3 == 2:
                    line += " "
            lines.append(line)
            
            if y % 3 == 0:      
                line = ""
                for x in range(12):
                    if x % 3 == 0:
                        line += "_"
                        if 0 < x < 11:
                            line += "_"
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        line += "_"
                    else:
                        line += " "
                lines.append(line)
        
        # Print the concatenated result
        result = "\n".join(lines)
        print(result)
        return result

    def render(self, mode="rgb_array"):
        print(f"the render mode is {self.render_mode}")
        if mode is None:
            mode = self.render_mode
        if mode != "rgb_array":
            raise NotImplementedError
        # Create RGB rendering of the office world
        cell_size = 40  # pixels per grid cell
        wall_thickness = 3
        
        # Total image dimensions
        img_height = 9 * cell_size
        img_width = 12 * cell_size
        
        # Create RGB image (height, width, 3)
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # white background
        
        # Define colors
        colors = {
            'wall': np.array([64, 64, 64]),      # dark gray
            'agent': np.array([255, 0, 0]),      # red
            'a': np.array([0, 255, 0]),          # green
            'b': np.array([0, 0, 255]),          # blue  
            'c': np.array([255, 255, 0]),        # yellow
            'd': np.array([255, 0, 255]),        # magenta
            'e': np.array([0, 255, 255]),        # cyan (mail)
            'f': np.array([139, 69, 19]),        # brown (coffee)
            'g': np.array([128, 128, 128]),      # gray (office)
            'n': np.array([0, 128, 0]),          # dark green (plants)
        }
        
        # Draw grid cells and objects
        for y in range(9):
            for x in range(12):
                # Calculate pixel coordinates (flip y for proper display)
                pixel_y = (8 - y) * cell_size
                pixel_x = x * cell_size
                
                # Draw object if present
                if (x, y) in self.objects:
                    obj_color = colors.get(self.objects[(x, y)], np.array([128, 128, 128]))
                    img[pixel_y:pixel_y+cell_size, pixel_x:pixel_x+cell_size] = obj_color
                
                # Draw agent
                if (x, y) == self.agent:
                    # Draw agent as a circle in the center of the cell
                    center_x = pixel_x + cell_size // 2
                    center_y = pixel_y + cell_size // 2
                    radius = cell_size // 3
                    
                    # Simple circle drawing
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                py = center_y + dy
                                px = center_x + dx
                                if 0 <= py < img_height and 0 <= px < img_width:
                                    img[py, px] = colors['agent']
        
        # Draw walls
        for y in range(9):
            for x in range(12):
                pixel_y = (8 - y) * cell_size
                pixel_x = x * cell_size
                
                # Draw walls based on forbidden transitions
                # Top wall
                if (x, y, Actions.up) in self.forbidden_transitions:
                    img[pixel_y:pixel_y+wall_thickness, pixel_x:pixel_x+cell_size] = colors['wall']
                
                # Bottom wall  
                if (x, y, Actions.down) in self.forbidden_transitions:
                    img[pixel_y+cell_size-wall_thickness:pixel_y+cell_size, pixel_x:pixel_x+cell_size] = colors['wall']
                
                # Left wall
                if (x, y, Actions.left) in self.forbidden_transitions:
                    img[pixel_y:pixel_y+cell_size, pixel_x:pixel_x+wall_thickness] = colors['wall']
                
                # Right wall
                if (x, y, Actions.right) in self.forbidden_transitions:
                    img[pixel_y:pixel_y+cell_size, pixel_x+cell_size-wall_thickness:pixel_x+cell_size] = colors['wall']
        
        # Draw outer boundaries
        img[0:wall_thickness, :] = colors['wall']  # top border
        img[-wall_thickness:, :] = colors['wall']  # bottom border  
        img[:, 0:wall_thickness] = colors['wall']  # left border
        img[:, -wall_thickness:] = colors['wall']  # right border
        
        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            # For human rendering, we would display the image
            # For now, just return the array (you could add matplotlib display here)
            return img
        else:
            return img

    def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step of each task to 1.
        """
        # if self.coffee_interval is not None:
        #     raise NotImplementedError("Unimplemented, because coffee interval adds partial observability")
        S = [(x,y) for x in range(12) for y in range(9)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x,y,a)
        return S,A,L,T # SALT xD
    
    def _set_objects_locations_features(self):
        objects_character_x_y_triple_list = [(v, k[0], k[1]) for k, v in self.objects.items()]
        objects_character_x_y_triple_list.sort()
        raise NotImplementedError("Unimplemented. This would be used for making better observations.")

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(1,1)] = "a"
        self.objects[(1,7)] = "b"
        self.objects[(10,7)] = "c"
        self.objects[(10,1)] = "d"
        self.objects[(7,4)] = "e"  # MAIL
        self.objects[(8,2)] = "f"  # COFFEE
        self.objects[(3,6)] = "f"  # COFFEE
        # f is coffee machine.
        # Brewing cofee will be x.
        # Ready coffee will be y.

        self.objects[(4,4)] = "g"  # OFFICE
        self.objects[(4,1)] = "n"  # PLANT
        self.objects[(7,1)] = "n"  # PLANT
        self.objects[(4,7)] = "n"  # PLANT
        self.objects[(7,7)] = "n"  # PLANT
        self.objects[(1,4)] = "n"  # PLANT
        self.objects[(10,4)] = "n" # PLANT
        if self.randomization == "objects":
            #print("DEBUG OfficeWorld: randomizing objects")
            keys = list(self.objects.keys())
            values = list(self.objects.values())
            self.rng.shuffle(values)
            self.objects = dict(zip(keys, values))
            self._set_objects_locations_features()


        # Adding walls
        self.forbidden_transitions = set()
        # general grid
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down)) 
                self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
            for x in [0,3,6,9]:
                self.forbidden_transitions.add((x,y,Actions.left))
                self.forbidden_transitions.add((x+2,y,Actions.right))
        # adding 'doors'
        for y in [1,7]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        # Adding the agent
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]

