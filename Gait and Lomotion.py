import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import time

class InchwormRobot:
    def __init__(self, num_actuators, segment_length=1.0):
        self.num_actuators = num_actuators
        self.segment_length = segment_length
        self.lift_height = 0.3  # Height when actuator lifts end off ground
        self.contraction_ratio = 0.7  # Contraction factor for middle actuators
        
        # Initialize positions and states
        self.reset_position()
        
    def reset_position(self):
        """Reset the inchworm to initial position"""
        self.actuator_states = np.zeros(self.num_actuators)  # 0 = off, 1 = on
        self.left_friction = True   # True = left end has friction, False = right end
        self.right_friction = False
        self.body_position = 0.0  # Overall body position
        self.current_step = 0
        self.step_in_cycle = 0  # Current step in 4-step cycle
        
        # Calculate segment positions based on actuator states
        self.update_positions()
        
    def get_middle_actuators(self):
        """Get indices of middle actuators (for propulsion)"""
        if self.num_actuators == 3:
            return [1]  # Only middle actuator
        elif self.num_actuators == 5:
            return [1, 2, 3]  # Actuators #2, #3, #4 (0-indexed: 1, 2, 3)
        elif self.num_actuators == 7:
            return [1, 2, 3, 4, 5]  # Middle 5 actuators
        else:
            # General case: exclude first and last actuator
            return list(range(1, self.num_actuators - 1))
    
    def update_positions(self):
        """Update segment positions based on actuator states and friction"""
        # Calculate body length based on middle actuator states
        middle_actuators = self.get_middle_actuators()
        middle_active = sum(self.actuator_states[i] for i in middle_actuators)
        
        # Body contracts when middle actuators are active
        if middle_active > 0:
            body_length = self.segment_length * self.num_actuators * self.contraction_ratio
        else:
            body_length = self.segment_length * self.num_actuators
        
        # Calculate end positions
        self.left_end_pos = self.body_position
        self.right_end_pos = self.body_position + body_length
        
        # Calculate heights (lifted ends have reduced friction)
        self.left_height = self.lift_height if self.actuator_states[0] == 1 else 0
        self.right_height = self.lift_height if self.actuator_states[-1] == 1 else 0
        
        # Generate sinusoidal body curvature
        self.generate_body_curve(body_length, middle_active)
        
        # Update friction states
        self.left_friction = self.left_height == 0  # Friction when on ground
        self.right_friction = self.right_height == 0
        
    def generate_body_curve(self, body_length, middle_active):
        """Generate sinusoidal curvature for the inchworm body"""
        # Number of points along the body for smooth curve
        self.num_curve_points = 20
        
        # Create x coordinates along the body
        x_normalized = np.linspace(0, 1, self.num_curve_points)  # 0 to 1
        self.body_x = self.left_end_pos + x_normalized * body_length
        
        # Calculate curvature amplitude based on gait phase
        base_amplitude = 0.15
        contraction_amplitude = 0.4
        
        # Amplitude varies based on which actuators are active
        if middle_active > 0:
            # During contraction/extension - higher curvature
            amplitude = contraction_amplitude
            # Frequency depends on number of active middle actuators
            frequency = 1.5 + 0.5 * middle_active
        else:
            # During transition phases - lower curvature
            amplitude = base_amplitude
            frequency = 1.0
            
        # Add phase shift based on current step for wave propagation
        phase_shift = self.current_step * 0.3
        
        # Generate sinusoidal height profile
        sine_curve = amplitude * np.sin(frequency * np.pi * x_normalized + phase_shift)
        
        # Apply different curvature patterns based on actuator states
        if self.actuator_states[0] == 1 and middle_active > 0:
            # Left end lifted, middle contracting - arch from left
            weight = np.exp(-3 * x_normalized)  # Exponential decay from left
            self.body_y = sine_curve * weight + self.left_height * (1 - x_normalized)
            
        elif self.actuator_states[-1] == 1 and middle_active > 0:
            # Right end lifted, middle contracting - arch from right  
            weight = np.exp(-3 * (1 - x_normalized))  # Exponential decay from right
            self.body_y = sine_curve * weight + self.right_height * x_normalized
            
        elif self.actuator_states[0] == 1:
            # Only left end lifted
            linear_lift = self.left_height * (1 - x_normalized)
            self.body_y = sine_curve * 0.3 + linear_lift
            
        elif self.actuator_states[-1] == 1:
            # Only right end lifted
            linear_lift = self.right_height * x_normalized
            self.body_y = sine_curve * 0.3 + linear_lift
            
        else:
            # All on ground - gentle wave motion
            self.body_y = sine_curve * 0.2
            
        # Ensure minimum ground contact
        self.body_y = np.maximum(self.body_y, 0)
        
    def gait_step_5_actuator(self):
        """Execute 4-step gait cycle for 5-actuator robot"""
        # Reset all actuators
        self.actuator_states.fill(0)
        
        if self.step_in_cycle == 0:
            # Step 1: Turn on actuator #1 (left end) to reduce friction
            self.actuator_states[0] = 1
            
        elif self.step_in_cycle == 1:
            # Step 2: Turn on actuators #2, #3, #4 (middle actuators) + keep #1 on
            self.actuator_states[0] = 1  # Keep left end lifted
            self.actuator_states[1] = 1  # Actuator #2
            self.actuator_states[2] = 1  # Actuator #3  
            self.actuator_states[3] = 1  # Actuator #4
            
        elif self.step_in_cycle == 2:
            # Step 3: Turn off actuator #1, turn on actuator #5 (right end)
            # Middle actuators #2, #3, #4 stay on
            self.actuator_states[1] = 1  # Actuator #2
            self.actuator_states[2] = 1  # Actuator #3
            self.actuator_states[3] = 1  # Actuator #4
            self.actuator_states[4] = 1  # Actuator #5 (right end)
            
        elif self.step_in_cycle == 3:
            # Step 4: Turn off actuators #2, #3, #4. Keep #5 on
            self.actuator_states[4] = 1  # Keep right end lifted
            
        # Update positions and move body if middle actuators cause propulsion
        old_position = self.body_position
        self.update_positions()
        
        # Body movement logic based on friction and middle actuator states
        middle_actuators = self.get_middle_actuators()
        middle_active = sum(self.actuator_states[i] for i in middle_actuators)
        
        if self.step_in_cycle == 1:
            # Contracting with left friction - body moves right
            if self.left_friction and middle_active > 0:
                self.body_position += 0.3
        elif self.step_in_cycle == 2:
            # Transition step - change friction end
            pass
        elif self.step_in_cycle == 3:
            # Extending with right friction - body moves right  
            if self.right_friction:
                self.body_position += 0.3
                
        self.step_in_cycle = (self.step_in_cycle + 1) % 4
        self.current_step += 1
        
    def gait_step_3_actuator(self):
        """Execute simplified 4-step gait cycle for 3-actuator robot"""
        self.actuator_states.fill(0)
        
        if self.step_in_cycle == 0:
            # Step 1: Lift left end
            self.actuator_states[0] = 1
            
        elif self.step_in_cycle == 1:
            # Step 2: Contract middle + keep left lifted
            self.actuator_states[0] = 1
            self.actuator_states[1] = 1
            self.body_position += 0.25
            
        elif self.step_in_cycle == 2:
            # Step 3: Switch to right end lifted
            self.actuator_states[1] = 1  # Keep middle contracted
            self.actuator_states[2] = 1  # Lift right end
            
        elif self.step_in_cycle == 3:
            # Step 4: Extend middle with right lifted
            self.actuator_states[2] = 1
            self.body_position += 0.25
            
        self.step_in_cycle = (self.step_in_cycle + 1) % 4
        self.current_step += 1
        self.update_positions()
        
    def gait_step_7_actuator(self):
        """Execute 4-step gait cycle for 7-actuator robot"""
        self.actuator_states.fill(0)
        
        if self.step_in_cycle == 0:
            # Step 1: Lift left end
            self.actuator_states[0] = 1
            
        elif self.step_in_cycle == 1:
            # Step 2: Activate middle 5 actuators + keep left lifted
            self.actuator_states[0] = 1
            for i in range(1, 6):  # Actuators #2-#6
                self.actuator_states[i] = 1
            self.body_position += 0.4
            
        elif self.step_in_cycle == 2:
            # Step 3: Switch to right end lifted, keep middle active
            for i in range(1, 6):  # Keep middle actuators
                self.actuator_states[i] = 1
            self.actuator_states[6] = 1  # Lift right end
            
        elif self.step_in_cycle == 3:
            # Step 4: Only right end lifted
            self.actuator_states[6] = 1
            self.body_position += 0.4
            
        self.step_in_cycle = (self.step_in_cycle + 1) % 4
        self.current_step += 1
        self.update_positions()
    
    def gait_step(self):
        """Execute one step of the appropriate gait cycle"""
        if self.num_actuators == 3:
            self.gait_step_3_actuator()
        elif self.num_actuators == 5:
            self.gait_step_5_actuator()
        elif self.num_actuators == 7:
            self.gait_step_7_actuator()
        else:
            # Default to 5-actuator logic
            self.gait_step_5_actuator()

def simulate_friction_based_gait(num_actuators_list=[3, 5, 7], cycles=3):
    """Simulate friction-based inchworm locomotion with sinusoidal curvature"""
    
    fig, axes = plt.subplots(len(num_actuators_list), 1, figsize=(14, 12))
    if len(num_actuators_list) == 1:
        axes = [axes]
    
    robots = []
    body_lines = []
    actuator_patches = []
    friction_indicators = []
    
    # Create robots and visualization elements
    for idx, num_act in enumerate(num_actuators_list):
        robot = InchwormRobot(num_act)
        robots.append(robot)
        
        ax = axes[idx]
        ax.set_xlim(-1, cycles * 2 + 3)
        ax.set_ylim(-0.3, 1.2)
        ax.set_title(f'{num_act}-Actuator Inchworm: Sinusoidal Curvature Locomotion')
        ax.set_xlabel('Position')
        ax.set_ylabel('Height')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Create curved body line
        body_line, = ax.plot([], [], 'b-', linewidth=4, label='Body curve')
        body_lines.append(body_line)
        
        # Create actuator position markers
        patches = []
        for i in range(num_act):
            # Actuator circle markers
            circle = Circle((0, 0), 0.05, facecolor='gray', edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            patches.append(circle)
        actuator_patches.append(patches)
        
        # Create friction indicators (ground contact symbols)
        left_friction = Circle((0, -0.15), 0.06, facecolor='red', alpha=0.8, 
                              edgecolor='black', linewidth=2)
        right_friction = Circle((0, -0.15), 0.06, facecolor='red', alpha=0.8,
                               edgecolor='black', linewidth=2)
        ax.add_patch(left_friction)
        ax.add_patch(right_friction)
        friction_indicators.append((left_friction, right_friction))
        
        # Add ground line for reference
        ax.axhline(y=0, color='brown', linewidth=2, alpha=0.7, linestyle='-')
        ax.text(0.5, -0.25, 'Ground', ha='center', fontsize=10, color='brown')
        
        # Add legend
        ax.legend(loc='upper right')
        ax.text(0.02, 0.98, 'Red circles = Friction contact\nOrange dots = Active actuators', 
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(frame):
        step = frame
        
        for idx, (robot, body_line, patches, friction_ind) in enumerate(
            zip(robots, body_lines, actuator_patches, friction_indicators)):
            
            # Execute gait step
            robot.gait_step()
            
            # Update curved body line
            body_line.set_data(robot.body_x, robot.body_y)
            
            # Update actuator position markers along the curve
            for i, patch in enumerate(patches):
                # Calculate actuator position along the curved body
                t = i / (robot.num_actuators - 1) if robot.num_actuators > 1 else 0
                curve_idx = int(t * (robot.num_curve_points - 1))
                curve_idx = min(curve_idx, robot.num_curve_points - 1)
                
                actuator_x = robot.body_x[curve_idx]
                actuator_y = robot.body_y[curve_idx]
                patch.center = (actuator_x, actuator_y)
                
                # Color and size based on actuator state
                if robot.actuator_states[i] == 1:
                    patch.set_facecolor('orange')
                    patch.set_radius(0.08)  # Larger when active
                    patch.set_alpha(0.9)
                else:
                    patch.set_facecolor('lightblue')
                    patch.set_radius(0.05)  # Smaller when inactive
                    patch.set_alpha(0.7)
            
            # Update friction indicators
            left_friction, right_friction = friction_ind
            left_friction.center = (robot.left_end_pos, -0.15)
            right_friction.center = (robot.right_end_pos, -0.15)
            
            # Show/hide friction indicators
            left_friction.set_alpha(0.9 if robot.left_friction else 0.2)
            right_friction.set_alpha(0.9 if robot.right_friction else 0.2)
            
            # Make friction indicators pulse when active
            if robot.left_friction:
                left_friction.set_facecolor('red')
                left_friction.set_radius(0.06 + 0.02 * np.sin(step * 0.5))
            if robot.right_friction:
                right_friction.set_facecolor('red') 
                right_friction.set_radius(0.06 + 0.02 * np.sin(step * 0.5))
            
            # Display step information for first robot
            if idx == 0:
                step_names = ["Step 1: Lift Left End", "Step 2: Contract Middle", 
                            "Step 3: Switch Friction", "Step 4: Extend Middle"]
                current_step_name = step_names[robot.step_in_cycle]
                axes[0].set_title(f'{robot.num_actuators}-Actuator Inchworm - {current_step_name} (Sinusoidal Curvature)')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=cycles*4*4, 
                                 interval=1000, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    return anim

def analyze_friction_gait_cycle():
    """Analyze the 4-step friction-based gait cycle"""
    
    print("Friction-Based Inchworm Gait Analysis")
    print("=" * 50)
    
    robot = InchwormRobot(5)  # Use 5-actuator as example
    
    print("\n5-Actuator Gait Cycle (as described in research):")
    print("=" * 55)
    
    step_descriptions = [
        "Step 1: Actuator #1 ON → Lift left end, reduce left friction",
        "Step 2: Actuators #2,#3,#4 ON → Contract body with left end lifted", 
        "Step 3: Actuator #1 OFF, #5 ON → Switch friction to right end",
        "Step 4: Actuators #2,#3,#4 OFF → Extend body with right end lifted"
    ]
    
    for cycle in range(2):  # Show 2 complete cycles
        print(f"\n--- Cycle {cycle + 1} ---")
        for step in range(4):
            robot.gait_step()
            
            print(f"\n{step_descriptions[step]}")
            print(f"  Actuator states: {['ON' if s else 'OFF' for s in robot.actuator_states]}")
            print(f"  Left friction: {'YES' if robot.left_friction else 'NO'}")
            print(f"  Right friction: {'YES' if robot.right_friction else 'NO'}")
            print(f"  Body position: {robot.body_position:.2f}")
            print(f"  Body length: {robot.right_end_pos - robot.left_end_pos:.2f}")

def plot_curvature_analysis():
    """Analyze and plot the sinusoidal curvature patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    robot = InchwormRobot(5)
    
    # Plot 1: Body curvature for each gait step
    ax1 = axes[0, 0]
    colors = ['red', 'orange', 'green', 'blue']
    step_names = ['Step 1: Lift Left', 'Step 2: Contract', 'Step 3: Switch', 'Step 4: Extend']
    
    for step in range(4):
        robot.gait_step()
        ax1.plot(robot.body_x, robot.body_y, color=colors[step], linewidth=3, 
                label=step_names[step], alpha=0.8)
        ax1.fill_between(robot.body_x, 0, robot.body_y, color=colors[step], alpha=0.2)
        
    ax1.axhline(y=0, color='brown', linewidth=2, label='Ground')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Height')
    ax1.set_title('Body Curvature During Gait Cycle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Curvature amplitude over time
    ax2 = axes[0, 1]
    robot.reset_position()
    steps = []
    max_heights = []
    avg_curvatures = []
    
    for step in range(20):
        robot.gait_step()
        steps.append(step)
        max_heights.append(np.max(robot.body_y))
        # Calculate average curvature (second derivative approximation)
        if len(robot.body_y) > 2:
            curvature = np.mean(np.abs(np.diff(robot.body_y, 2)))
            avg_curvatures.append(curvature)
        else:
            avg_curvatures.append(0)
    
    ax2.plot(steps, max_heights, 'r-o', label='Max Height', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, avg_curvatures, 'b-s', label='Avg Curvature', linewidth=2)
    
    ax2.set_xlabel('Step Number')
    ax2.set_ylabel('Max Height', color='red')
    ax2_twin.set_ylabel('Average Curvature', color='blue')
    ax2.set_title('Height and Curvature vs Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison of different actuator numbers
    ax3 = axes[1, 0]
    actuator_counts = [3, 5, 7]
    colors = ['red', 'green', 'blue']
    
    for i, num_act in enumerate(actuator_counts):
        robot = InchwormRobot(num_act)
        robot.gait_step()  # Go to step 1
        robot.gait_step()  # Go to step 2 (peak curvature)
        
        ax3.plot(robot.body_x, robot.body_y, color=colors[i], linewidth=3,
                label=f'{num_act} actuators', alpha=0.8)
    
    ax3.axhline(y=0, color='brown', linewidth=2)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Height')
    ax3.set_title('Curvature Comparison: Different Actuator Counts')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 3D trajectory over time
    ax4 = axes[1, 1]
    robot = InchwormRobot(5)
    
    # Collect body shapes over multiple steps
    all_x = []
    all_y = []
    step_numbers = []
    
    for step in range(12):  # 3 complete cycles
        robot.gait_step()
        for i, (x, y) in enumerate(zip(robot.body_x, robot.body_y)):
            all_x.append(x)
            all_y.append(y)
            step_numbers.append(step)
    
    # Create scatter plot with color coding for time
    scatter = ax4.scatter(all_x, all_y, c=step_numbers, cmap='viridis', alpha=0.6)
    ax4.axhline(y=0, color='brown', linewidth=2)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Height')
    ax4.set_title('Body Point Trajectories Over Time')
    plt.colorbar(scatter, ax=ax4, label='Step Number')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Friction-Based Inchworm Robot Simulation")
    print("Based on research paper gait cycle logic")
    print("=" * 50)
    
    # Analyze the gait cycle
    analyze_friction_gait_cycle()
    
    # Show curvature analysis
    plot_curvature_analysis()
    
    # Run animation
    print("\nStarting friction-based gait animation...")
    print("Watch the red circles (friction contact) and orange actuators (active)")
    anim = simulate_friction_based_gait([3, 5, 7], cycles=4)
    
    plt.show()



    #############################################33
    import numpy as np
import matplotlib.pyplot as plt

class JiakangPZT780C7Calculator:
    """
    Calculator for JIAKANG PZT780C7 Piezoelectric Bimorph Actuator
    Specifications: 51.8mm x 7.1mm x 0.78mm
    """
    
    def __init__(self):
        # Actuator specifications from JIAKANG PZT780C7
        self.length = 51.8e-3  # m (51.8mm)
        self.width = 7.1e-3    # m (7.1mm)
        self.thickness = 0.78e-3  # m (0.78mm)
        
        # Typical PZT bimorph properties (estimated for this model)
        self.d31 = -190e-12    # m/V (piezoelectric constant for bimorph)
        self.d33 = 780e-12     # C/N (longitudinal piezoelectric constant)
        self.youngs_modulus = 63e9  # Pa (Young's modulus of PZT)
        self.density = 7800    # kg/m³
        self.relative_permittivity = 1800
        self.epsilon_0 = 8.854e-12  # F/m
        
        # Bimorph specific parameters
        self.layer_thickness = self.thickness / 2  # Each layer thickness
        self.max_voltage = 200  # V (typical max for this size)
        self.resonant_freq = None  # Will be calculated
        
        # Calculate cross-sectional area
        self.cross_area = self.width * self.thickness
        
    def calculate_tip_displacement(self, voltage, load_force=0):
        """
        Calculate tip displacement for bimorph actuator
        For bimorph bending mode
        """
        # Free displacement (no load) - bimorph formula
        # δ = (3 * d31 * V * L²) / (2 * t²)
        free_displacement = (3 * abs(self.d31) * voltage * self.length**2) / (2 * self.layer_thickness**2)
        
        if load_force > 0:
            # Calculate blocking force
            blocking_force = self.calculate_blocking_force(voltage)
            # Displacement under load
            displacement = free_displacement * (1 - load_force / blocking_force)
            return max(0, displacement)
        
        return free_displacement
    
    def calculate_blocking_force(self, voltage):
        """Calculate maximum blocking force"""
        # F_block = (3 * E * w * t * d31 * V) / (2 * L²)
        blocking_force = (3 * self.youngs_modulus * self.width * self.layer_thickness * 
                         abs(self.d31) * voltage) / (2 * self.length**2)
        return blocking_force
    
    def calculate_lift_height(self, voltage, load_mass=0):
        """
        Calculate lift height (same as tip displacement for bimorph)
        """
        load_force = load_mass * 9.81  # Convert mass to force
        return self.calculate_tip_displacement(voltage, load_force)
    
    def calculate_contraction_ratio(self, voltage):
        """Calculate strain/contraction ratio"""
        # For longitudinal mode (if used in stack configuration)
        strain = self.d33 * voltage / self.thickness
        contraction_ratio = strain * 100  # Convert to percentage
        return contraction_ratio
    
    def calculate_resonant_frequency(self):
        """Calculate first resonant frequency for cantilever bimorph"""
        # Simplified formula for cantilever beam
        # f = (λ²/2π) * sqrt(E*I/(ρ*A*L⁴))
        # where λ ≈ 1.875 for first mode
        
        lambda_1 = 1.875  # First mode eigenvalue
        
        # Moment of inertia for rectangular cross-section
        I = (self.width * self.thickness**3) / 12
        
        # Mass per unit length
        mass_per_length = self.density * self.cross_area
        
        # Resonant frequency
        freq = (lambda_1**2 / (2 * np.pi)) * np.sqrt(
            (self.youngs_modulus * I) / (mass_per_length * self.length**4)
        )
        
        self.resonant_freq = freq
        return freq
    
    def calculate_capacitance(self):
        """Calculate actuator capacitance"""
        # C = ε₀ * εᵣ * A / t
        # For bimorph, two capacitors in parallel
        single_cap = (self.epsilon_0 * self.relative_permittivity * 
                     self.length * self.width) / self.layer_thickness
        total_capacitance = 2 * single_cap  # Two layers in parallel
        return total_capacitance
    
    def calculate_power_consumption(self, voltage, frequency):
        """Calculate power consumption at given frequency"""
        capacitance = self.calculate_capacitance()
        # P = V² * f * C
        power = voltage**2 * frequency * capacitance
        return power
    
    def calculate_response_time(self, drive_resistance=1000):
        """Calculate electrical response time"""
        capacitance = self.calculate_capacitance()
        # τ = R * C
        response_time = drive_resistance * capacitance
        return response_time
    
    def calculate_energy_density(self, voltage):
        """Calculate energy density"""
        electric_field = voltage / self.layer_thickness
        permittivity = self.epsilon_0 * self.relative_permittivity
        # Energy density = ½ * ε * E²
        energy_density = 0.5 * permittivity * electric_field**2
        return energy_density
    
    def performance_analysis(self, voltage_range=None, load_range=None):
        """Comprehensive performance analysis"""
        if voltage_range is None:
            voltage_range = np.linspace(0, self.max_voltage, 50)
        if load_range is None:
            load_range = np.linspace(0, 0.1, 20)  # 0 to 100g
        
        results = {
            'voltage': voltage_range,
            'displacement': [],
            'blocking_force': [],
            'power_100hz': [],
            'load_mass': load_range,
            'displacement_vs_load': []
        }
        
        # Calculate for different voltages
        for v in voltage_range:
            disp = self.calculate_tip_displacement(v)
            force = self.calculate_blocking_force(v)
            power = self.calculate_power_consumption(v, 100)  # At 100Hz
            
            results['displacement'].append(disp * 1e6)  # Convert to μm
            results['blocking_force'].append(force * 1000)  # Convert to mN
            results['power_100hz'].append(power * 1000)  # Convert to mW
        
        # Calculate displacement vs load at max voltage
        for mass in load_range:
            disp = self.calculate_lift_height(self.max_voltage, mass)
            results['displacement_vs_load'].append(disp * 1e6)  # Convert to μm
        
        return results
    
    def print_specifications(self):
        """Print actuator specifications and calculated parameters"""
        print("=" * 60)
        print("JIAKANG PZT780C7 Piezoelectric Bimorph Actuator Analysis")
        print("=" * 60)
        print(f"Physical Dimensions:")
        print(f"  Length: {self.length*1000:.1f} mm")
        print(f"  Width: {self.width*1000:.1f} mm")
        print(f"  Thickness: {self.thickness*1000:.2f} mm")
        print(f"  Cross-sectional area: {self.cross_area*1e6:.2f} mm²")
        
        print(f"\nMaterial Properties (PZT):")
        print(f"  d31 coefficient: {self.d31*1e12:.0f} pm/V")
        print(f"  d33 coefficient: {self.d33*1e12:.0f} pm/V")
        print(f"  Young's modulus: {self.youngs_modulus/1e9:.0f} GPa")
        print(f"  Density: {self.density} kg/m³")
        
        # Calculate key performance parameters
        capacitance = self.calculate_capacitance()
        resonant_freq = self.calculate_resonant_frequency()
        max_displacement = self.calculate_tip_displacement(self.max_voltage)
        max_force = self.calculate_blocking_force(self.max_voltage)
        
        print(f"\nPerformance at {self.max_voltage}V:")
        print(f"  Maximum tip displacement: {max_displacement*1e6:.1f} μm")
        print(f"  Maximum blocking force: {max_force*1000:.2f} mN")
        print(f"  Capacitance: {capacitance*1e9:.1f} nF")
        print(f"  First resonant frequency: {resonant_freq:.1f} Hz")
        print(f"  Response time (1kΩ): {self.calculate_response_time()*1e6:.1f} μs")
        
        # Calculate some practical examples
        print(f"\nPractical Examples:")
        print(f"  Displacement at 100V: {self.calculate_tip_displacement(100)*1e6:.1f} μm")
        print(f"  Displacement at 50V: {self.calculate_tip_displacement(50)*1e6:.1f} μm")
        print(f"  Power at 100V, 100Hz: {self.calculate_power_consumption(100, 100)*1000:.1f} mW")
        print(f"  Lift with 10g load at 200V: {self.calculate_lift_height(200, 0.01)*1e6:.1f} μm")

def plot_performance_curves(calculator):
    """Plot performance characteristics"""
    results = calculator.performance_analysis()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Displacement vs Voltage
    ax1.plot(results['voltage'], results['displacement'], 'b-', linewidth=2)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Tip Displacement (μm)')
    ax1.set_title('Displacement vs Voltage')
    ax1.grid(True, alpha=0.3)
    
    # Blocking Force vs Voltage
    ax2.plot(results['voltage'], results['blocking_force'], 'r-', linewidth=2)
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Blocking Force (mN)')
    ax2.set_title('Blocking Force vs Voltage')
    ax2.grid(True, alpha=0.3)
    
    # Power Consumption vs Voltage (at 100Hz)
    ax3.plot(results['voltage'], results['power_100hz'], 'g-', linewidth=2)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Power Consumption (mW)')
    ax3.set_title('Power vs Voltage (100Hz)')
    ax3.grid(True, alpha=0.3)
    
    # Displacement vs Load Mass (at max voltage)
    ax4.plot(results['load_mass']*1000, results['displacement_vs_load'], 'm-', linewidth=2)
    ax4.set_xlabel('Load Mass (g)')
    ax4.set_ylabel('Displacement (μm)')
    ax4.set_title(f'Displacement vs Load ({calculator.max_voltage}V)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage and calculations
if __name__ == "__main__":
    # Create calculator instance
    calc = JiakangPZT780C7Calculator()
    
    # Print comprehensive specifications
    calc.print_specifications()
    
    # Calculate specific scenarios
    print("\n" + "="*60)
    print("SPECIFIC CALCULATION EXAMPLES")
    print("="*60)
    
    # Example 1: Displacement calculations
    voltages = [50, 100, 150, 200]
    print("\nDisplacement Calculations:")
    for v in voltages:
        disp = calc.calculate_tip_displacement(v)
        print(f"  At {v}V: {disp*1e6:.2f} μm")
    
    # Example 2: Load carrying capacity
    print(f"\nLoad Carrying Capacity at {calc.max_voltage}V:")
    loads = [0.005, 0.010, 0.020, 0.050]  # 5g, 10g, 20g, 50g
    for mass in loads:
        lift = calc.calculate_lift_height(calc.max_voltage, mass)
        print(f"  With {mass*1000:.0f}g load: {lift*1e6:.2f} μm")
    
    # Example 3: Frequency response
    print(f"\nFrequency Response:")
    freq = calc.calculate_resonant_frequency()
    print(f"  First resonant frequency: {freq:.1f} Hz")
    print(f"  Recommended operating range: 0.1 - {freq*0.3:.1f} Hz")
    
    # Example 4: Power analysis
    print(f"\nPower Analysis at 100V:")
    frequencies = [1, 10, 100, 1000]
    for f in frequencies:
        power = calc.calculate_power_consumption(100, f)
        print(f"  At {f}Hz: {power*1000:.2f} mW")
    
    # Plot performance curves
    print(f"\nGenerating performance plots...")
    plot_performance_curves(calc)



    #################################################3
    import numpy as np
import matplotlib.pyplot as plt

class JiakangPZT780C7Calculator:
    """
    Calculator for JIAKANG PZT780C7 Piezoelectric Bimorph Actuator
    Specifications: 51.8mm x 7.1mm x 0.78mm
    """
    
    def __init__(self):
        # Actuator specifications from JIAKANG PZT780C7
        self.length = 51.8e-3  # m (51.8mm)
        self.width = 7.1e-3    # m (7.1mm)
        self.thickness = 0.78e-3  # m (0.78mm)
        
        # Typical PZT bimorph properties (estimated for this model)
        self.d31 = -190e-12    # m/V (piezoelectric constant for bimorph)
        self.d33 = 400e-12     # m/V (longitudinal piezoelectric constant)
        self.youngs_modulus = 63e9  # Pa (Young's modulus of PZT)
        self.density = 7800    # kg/m³
        self.relative_permittivity = 1800
        self.epsilon_0 = 8.854e-12  # F/m
        
        # Bimorph specific parameters
        self.layer_thickness = self.thickness / 2  # Each layer thickness
        self.max_voltage = 200  # V (typical max for this size)
        self.resonant_freq = None  # Will be calculated
        
        # Calculate cross-sectional area
        self.cross_area = self.width * self.thickness
        
    def calculate_tip_displacement(self, voltage, load_force=0):
        """
        Calculate tip displacement for bimorph actuator
        For bimorph bending mode
        """
        # Free displacement (no load) - bimorph formula
        # δ = (3 * d31 * V * L²) / (2 * t²)
        free_displacement = (3 * abs(self.d31) * voltage * self.length**2) / (2 * self.layer_thickness**2)
        
        if load_force > 0:
            # Calculate blocking force
            blocking_force = self.calculate_blocking_force(voltage)
            # Displacement under load
            displacement = free_displacement * (1 - load_force / blocking_force)
            return max(0, displacement)
        
        return free_displacement
    
    def calculate_blocking_force(self, voltage):
        """Calculate maximum blocking force"""
        # F_block = (3 * E * w * t * d31 * V) / (2 * L²)
        blocking_force = (3 * self.youngs_modulus * self.width * self.layer_thickness * 
                         abs(self.d31) * voltage) / (2 * self.length**2)
        return blocking_force
    
    def calculate_lift_height(self, voltage, load_mass=0):
        """
        Calculate lift height (same as tip displacement for bimorph)
        """
        load_force = load_mass * 9.81  # Convert mass to force
        return self.calculate_tip_displacement(voltage, load_force)
    
    def calculate_contraction_ratio(self, voltage):
        """Calculate strain/contraction ratio"""
        # For longitudinal mode (if used in stack configuration)
        strain = self.d33 * voltage / self.thickness
        contraction_ratio = strain * 100  # Convert to percentage
        return contraction_ratio
    
    def calculate_resonant_frequency(self):
        """Calculate first resonant frequency for cantilever bimorph"""
        # Simplified formula for cantilever beam
        # f = (λ²/2π) * sqrt(E*I/(ρ*A*L⁴))
        # where λ ≈ 1.875 for first mode
        
        lambda_1 = 1.875  # First mode eigenvalue
        
        # Moment of inertia for rectangular cross-section
        I = (self.width * self.thickness**3) / 12
        
        # Mass per unit length
        mass_per_length = self.density * self.cross_area
        
        # Resonant frequency
        freq = (lambda_1**2 / (2 * np.pi)) * np.sqrt(
            (self.youngs_modulus * I) / (mass_per_length * self.length**4)
        )
        
        self.resonant_freq = freq
        return freq
    
    def calculate_capacitance(self):
        """Calculate actuator capacitance"""
        # C = ε₀ * εᵣ * A / t
        # For bimorph, two capacitors in parallel
        single_cap = (self.epsilon_0 * self.relative_permittivity * 
                     self.length * self.width) / self.layer_thickness
        total_capacitance = 2 * single_cap  # Two layers in parallel
        return total_capacitance
    
    def calculate_power_consumption(self, voltage, frequency):
        """Calculate power consumption at given frequency"""
        capacitance = self.calculate_capacitance()
        # P = V² * f * C
        power = voltage**2 * frequency * capacitance
        return power
    
    def calculate_response_time(self, drive_resistance=1000):
        """Calculate electrical response time"""
        capacitance = self.calculate_capacitance()
        # τ = R * C
        response_time = drive_resistance * capacitance
        return response_time
    
    def calculate_energy_density(self, voltage):
        """Calculate energy density"""
        electric_field = voltage / self.layer_thickness
        permittivity = self.epsilon_0 * self.relative_permittivity
        # Energy density = ½ * ε * E²
        energy_density = 0.5 * permittivity * electric_field**2
        return energy_density
    
    def performance_analysis(self, voltage_range=None, load_range=None):
        """Comprehensive performance analysis"""
        if voltage_range is None:
            voltage_range = np.linspace(0, self.max_voltage, 50)
        if load_range is None:
            load_range = np.linspace(0, 0.1, 20)  # 0 to 100g
        
        results = {
            'voltage': voltage_range,
            'displacement': [],
            'blocking_force': [],
            'power_100hz': [],
            'load_mass': load_range,
            'displacement_vs_load': []
        }
        
        # Calculate for different voltages
        for v in voltage_range:
            disp = self.calculate_tip_displacement(v)
            force = self.calculate_blocking_force(v)
            power = self.calculate_power_consumption(v, 100)  # At 100Hz
            
            results['displacement'].append(disp * 1e6)  # Convert to μm
            results['blocking_force'].append(force * 1000)  # Convert to mN
            results['power_100hz'].append(power * 1000)  # Convert to mW
        
        # Calculate displacement vs load at max voltage
        for mass in load_range:
            disp = self.calculate_lift_height(self.max_voltage, mass)
            results['displacement_vs_load'].append(disp * 1e6)  # Convert to μm
        
        return results
    
    def calculate_dragonskin_force(self, thickness_mm, displacement_mm, dragonskin_type="30"):
        """
        Calculate force required to deform Dragon Skin silicone
        
        Parameters:
        thickness_mm: thickness of Dragon Skin in mm
        displacement_mm: desired displacement in mm
        dragonskin_type: "10", "20", or "30" (Shore A hardness)
        """
        # Dragon Skin properties based on type
        properties = {
            "10": {"youngs_modulus": 0.09e6, "shore_a": 10},  # Pa
            "20": {"youngs_modulus": 0.5e6, "shore_a": 20},   # Pa (estimated)
            "30": {"youngs_modulus": 1.0e6, "shore_a": 30}    # Pa
        }
        
        if dragonskin_type not in properties:
            dragonskin_type = "30"  # Default to Dragon Skin 30
            
        E = properties[dragonskin_type]["youngs_modulus"]
        
        # Convert to meters
        thickness = thickness_mm / 1000
        displacement = displacement_mm / 1000
        
        # Calculate strain
        strain = displacement / thickness
        
        # Calculate stress (σ = E × ε)
        stress = E * strain
        
        # Calculate force (F = σ × A)
        # Assuming the Dragon Skin covers the full actuator contact area
        contact_area = self.length * self.width
        force = stress * contact_area
        
        return force, strain, stress
    
    def calculate_displacement_with_dragonskin(self, voltage, dragonskin_thickness_mm, 
                                             dragonskin_type="30", max_dragonskin_displacement=None):
        """
        Calculate actuator displacement when working against Dragon Skin load
        
        Parameters:
        voltage: Applied voltage (V)
        dragonskin_thickness_mm: Dragon Skin thickness (mm)
        dragonskin_type: Dragon Skin type ("10", "20", "30")
        max_dragonskin_displacement: Maximum displacement to calculate (mm), if None uses free displacement
        """
        # Calculate free displacement first
        free_displacement = self.calculate_tip_displacement(voltage)
        
        if max_dragonskin_displacement is None:
            max_dragonskin_displacement = free_displacement * 1000  # Convert to mm
        
        # Calculate blocking force
        max_blocking_force = self.calculate_blocking_force(voltage)
        
        # Iteratively find the equilibrium displacement
        # At equilibrium: actuator force = Dragon Skin resistance force
        
        displacement_range = np.linspace(0, min(max_dragonskin_displacement/1000, free_displacement), 100)
        
        for disp in displacement_range:
            # Calculate Dragon Skin resistance force at this displacement
            dragonskin_force, strain, stress = self.calculate_dragonskin_force(
                dragonskin_thickness_mm, disp * 1000, dragonskin_type)
            
            # Calculate actuator force at this displacement
            actuator_force = max_blocking_force * (1 - disp / free_displacement)
            
            # Check if forces are balanced (within 1% tolerance)
            if abs(actuator_force - dragonskin_force) / max_blocking_force < 0.01:
                return disp, dragonskin_force, strain, stress, actuator_force
        
        # If no equilibrium found, return the last calculated values
        return disp, dragonskin_force, strain, stress, actuator_force
    
    def analyze_dragonskin_performance(self, dragonskin_thickness_mm=1.0, dragonskin_type="20"):
        """Comprehensive analysis with Dragon Skin load"""
        print(f"\n" + "="*70)
        print(f"DRAGON SKIN {dragonskin_type} LOAD ANALYSIS ({dragonskin_thickness_mm}mm thick)")
        print("="*70)
        
        voltages = [50, 100, 150, 200]
        
        print(f"\nDragon Skin {dragonskin_type} Properties:")
        properties = {
            "10": {"youngs_modulus": 0.09e6, "shore_a": 10},
            "20": {"youngs_modulus": 0.5e6, "shore_a": 20},
            "30": {"youngs_modulus": 1.0e6, "shore_a": 30}
        }
        E = properties[dragonskin_type]["youngs_modulus"]
        print(f"  Shore A Hardness: {properties[dragonskin_type]['shore_a']}")
        print(f"  Young's Modulus: {E/1000:.1f} kPa")
        print(f"  Thickness: {dragonskin_thickness_mm} mm")
        
        print(f"\nActuator Performance with Dragon Skin Load:")
        print(f"{'Voltage (V)':<12} {'Free Disp (μm)':<15} {'With Load (μm)':<15} {'Reduction (%)':<15} {'Load Force (mN)':<15}")
        print("-" * 75)
        
        for voltage in voltages:
            free_disp = self.calculate_tip_displacement(voltage)
            
            try:
                loaded_disp, force, strain, stress, actuator_force = self.calculate_displacement_with_dragonskin(
                    voltage, dragonskin_thickness_mm, dragonskin_type)
                
                reduction = ((free_disp - loaded_disp) / free_disp) * 100
                
                print(f"{voltage:<12} {free_disp*1e6:<15.1f} {loaded_disp*1e6:<15.1f} {reduction:<15.1f} {force*1000:<15.2f}")
                
            except:
                print(f"{voltage:<12} {free_disp*1e6:<15.1f} {'Error':<15} {'N/A':<15} {'N/A':<15}")
        
        # Detailed analysis at maximum voltage
        print(f"\nDetailed Analysis at {self.max_voltage}V:")
        try:
            final_disp, dragon_force, strain, stress, act_force = self.calculate_displacement_with_dragonskin(
                self.max_voltage, dragonskin_thickness_mm, dragonskin_type)
            
            print(f"  Final displacement: {final_disp*1e6:.1f} μm")
            print(f"  Dragon Skin strain: {strain*100:.2f}%")
            print(f"  Dragon Skin stress: {stress/1000:.2f} kPa")
            print(f"  Resistance force: {dragon_force*1000:.2f} mN")
            print(f"  Actuator force: {act_force*1000:.2f} mN")
            
            # Calculate efficiency
            free_disp_max = self.calculate_tip_displacement(self.max_voltage)
            efficiency = (final_disp / free_disp_max) * 100
            print(f"  Load efficiency: {efficiency:.1f}%")
            
        except Exception as e:
            print(f"  Error in detailed analysis: {str(e)}")

    def print_specifications(self):
        """Print actuator specifications and calculated parameters"""
        print("=" * 60)
        print("JIAKANG PZT780C7 Piezoelectric Bimorph Actuator Analysis")
        print("=" * 60)
        print(f"Physical Dimensions:")
        print(f"  Length: {self.length*1000:.1f} mm")
        print(f"  Width: {self.width*1000:.1f} mm")
        print(f"  Thickness: {self.thickness*1000:.2f} mm")
        print(f"  Cross-sectional area: {self.cross_area*1e6:.2f} mm²")
        
        print(f"\nMaterial Properties (PZT):")
        print(f"  d31 coefficient: {self.d31*1e12:.0f} pm/V")
        print(f"  d33 coefficient: {self.d33*1e12:.0f} pm/V")
        print(f"  Young's modulus: {self.youngs_modulus/1e9:.0f} GPa")
        print(f"  Density: {self.density} kg/m³")
        
        # Calculate key performance parameters
        capacitance = self.calculate_capacitance()
        resonant_freq = self.calculate_resonant_frequency()
        max_displacement = self.calculate_tip_displacement(self.max_voltage)
        max_force = self.calculate_blocking_force(self.max_voltage)
        
        print(f"\nPerformance at {self.max_voltage}V:")
        print(f"  Maximum tip displacement: {max_displacement*1e6:.1f} μm")
        print(f"  Maximum blocking force: {max_force*1000:.2f} mN")
        print(f"  Capacitance: {capacitance*1e9:.1f} nF")
        print(f"  First resonant frequency: {resonant_freq:.1f} Hz")
        print(f"  Response time (1kΩ): {self.calculate_response_time()*1e6:.1f} μs")
        
        # Calculate some practical examples
        print(f"\nPractical Examples:")
        print(f"  Displacement at 100V: {self.calculate_tip_displacement(100)*1e6:.1f} μm")
        print(f"  Displacement at 50V: {self.calculate_tip_displacement(50)*1e6:.1f} μm")
        print(f"  Power at 100V, 100Hz: {self.calculate_power_consumption(100, 100)*1000:.1f} mW")
        print(f"  Lift with 10g load at 200V: {self.calculate_lift_height(200, 0.01)*1e6:.1f} μm")

def plot_performance_curves(calculator):
    """Plot performance characteristics"""
    results = calculator.performance_analysis()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Displacement vs Voltage
    ax1.plot(results['voltage'], results['displacement'], 'b-', linewidth=2)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Tip Displacement (μm)')
    ax1.set_title('Displacement vs Voltage')
    ax1.grid(True, alpha=0.3)
    
    # Blocking Force vs Voltage
    ax2.plot(results['voltage'], results['blocking_force'], 'r-', linewidth=2)
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Blocking Force (mN)')
    ax2.set_title('Blocking Force vs Voltage')
    ax2.grid(True, alpha=0.3)
    
    # Power Consumption vs Voltage (at 100Hz)
    ax3.plot(results['voltage'], results['power_100hz'], 'g-', linewidth=2)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Power Consumption (mW)')
    ax3.set_title('Power vs Voltage (100Hz)')
    ax3.grid(True, alpha=0.3)
    
    # Displacement vs Load Mass (at max voltage)
    ax4.plot(results['load_mass']*1000, results['displacement_vs_load'], 'm-', linewidth=2)
    ax4.set_xlabel('Load Mass (g)')
    ax4.set_ylabel('Displacement (μm)')
    ax4.set_title(f'Displacement vs Load ({calculator.max_voltage}V)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage and calculations
if __name__ == "__main__":
    # Create calculator instance
    calc = JiakangPZT780C7Calculator()
    
    # Print comprehensive specifications
    calc.print_specifications()
    
    # Calculate specific scenarios
    print("\n" + "="*60)
    print("SPECIFIC CALCULATION EXAMPLES")
    print("="*60)
    
    # Example 1: Displacement calculations
    voltages = [50, 100, 150, 200]
    print("\nDisplacement Calculations:")
    for v in voltages:
        disp = calc.calculate_tip_displacement(v)
        print(f"  At {v}V: {disp*1e6:.2f} μm")
    
    # Example 2: Load carrying capacity
    print(f"\nLoad Carrying Capacity at {calc.max_voltage}V:")
    loads = [0.005, 0.010, 0.020, 0.050]  # 5g, 10g, 20g, 50g
    for mass in loads:
        lift = calc.calculate_lift_height(calc.max_voltage, mass)
        print(f"  With {mass*1000:.0f}g load: {lift*1e6:.2f} μm")
    
    # Example 3: Frequency response
    print(f"\nFrequency Response:")
    freq = calc.calculate_resonant_frequency()
    print(f"  First resonant frequency: {freq:.1f} Hz")
    print(f"  Recommended operating range: 0.1 - {freq*0.3:.1f} Hz")
    
    # Example 4: Power analysis
    print(f"\nPower Analysis at 100V:")
    frequencies = [1, 10, 100, 1000]
    for f in frequencies:
        power = calc.calculate_power_consumption(100, f)
        print(f"  At {f}Hz: {power*1000:.2f} mW")
    
    # Calculate Dragon Skin 20 performance with 1mm thickness
    print(f"\n" + "="*60)
    print("DRAGON SKIN 20 LOAD CALCULATIONS (1mm thick)")
    print("="*60)
    
    # Analyze Dragon Skin 20 performance
    calc.analyze_dragonskin_performance(dragonskin_thickness_mm=1.0, dragonskin_type="20")
    
    # Specific calculation for your question
    voltage = 200  # Maximum voltage
    try:
        final_disp, dragon_force, strain, stress, act_force = calc.calculate_displacement_with_dragonskin(
            voltage, 1.0, "20")
        
        print(f"\n🎯 ANSWER TO YOUR QUESTION:")
        print(f"   With 1mm Dragon Skin 20 load at {voltage}V:")
        print(f"   Final displacement: {final_disp*1e6:.1f} μm")
        print(f"   Displacement in mm: {final_disp*1000:.3f} mm")
        
        # Compare with free displacement
        free_disp = calc.calculate_tip_displacement(voltage)
        reduction = ((free_disp - final_disp) / free_disp) * 100
        print(f"   Free displacement would be: {free_disp*1e6:.1f} μm")
        print(f"   Reduction due to Dragon Skin: {reduction:.1f}%")
        
    except Exception as e:
        print(f"   Error in calculation: {str(e)}")
        # Fallback calculation
        free_disp = calc.calculate_tip_displacement(200)
        print(f"   Free displacement at 200V: {free_disp*1e6:.1f} μm")
        print(f"   With Dragon Skin load, expect ~50-80% reduction")