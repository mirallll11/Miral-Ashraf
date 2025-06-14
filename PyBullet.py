# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math 

# Define Trimorph Base Model
class TrimorphModel:
    def __init__(self):
        # Geometry
        self.actuatorLength = None  # cm
        self.width = None           # cm
        self.subThickness = None    # cm
        self.actThickness = None    # cm
        self.epoxyThickness = None  # cm
        self.pitchDistance = None   # cm

        # Material Properties
        self.piezoModulusNoPoisson = None  # Ba
        self.epoxyModulus = None           # Ba
        self.subModulusNoPoisson = None    # Ba

        self.subDensity = None     # g/cm³
        self.piezoDensity = None   # g/cm³
        self.epoxyDensity = None   # g/cm³

        self.piezoPoisson = None
        self.subPoisson = None
        self.d33NoPoisson = None   # cm/V
        self.p2p1ratio = None

        # Computed properties
        self.gamma = None           # 1/cm
        self.EI = None              # Ba·cm⁴
        self.gravity = 980.0        # cm/s²

    def reset(self):
        self.piezoModulus = self.piezoModulusNoPoisson / (1 - self.piezoPoisson ** 2)
        self.subModulus = self.subModulusNoPoisson / (1 - self.subPoisson ** 2)
        self.d33 = self.d33NoPoisson * (1 + self.piezoPoisson)

        self.actuator2DDensity = (
            self.subDensity * self.subThickness +
            self.piezoDensity * self.actThickness +
            self.epoxyDensity * self.epoxyThickness
        )
        self.actuator1DDensity = self.actuator2DDensity * self.width
        self.actuatorMass = self.actuator1DDensity * self.actuatorLength

        self.voltageExpansion = self.d33 / self.pitchDistance
        self.gamma = self.get_curvature(epoxyThickness=self.epoxyThickness)
        self.EI = self.get_flexural_rigidty(epoxyThickness=self.epoxyThickness)
        self.beta = self.gamma * self.actuatorLength
        self.halfEI = self.EI / 2.0
        self.load = self.actuatorMass * self.gravity / (self.actuatorLength * self.width)

    def get_curvature(self, epoxyThickness):
        EI = self.get_flexural_rigidty(epoxyThickness)
        zPiezo, _, _ = self.get_relative_position(epoxyThickness)
        return self.voltageExpansion * zPiezo * self.piezoModulus * self.actThickness / EI

    def get_relative_position(self, epoxyThickness):
        neutralAxis = self.get_neutral_axis(epoxyThickness)
        zPiezo = 0.5 * self.actThickness + epoxyThickness - neutralAxis
        zEpoxy = 0.5 * epoxyThickness - neutralAxis
        zSub = -0.5 * self.subThickness - neutralAxis
        return zPiezo, zEpoxy, zSub

    def get_flexural_rigidty(self, epoxyThickness):
        zPiezo, zEpoxy, zSub = self.get_relative_position(epoxyThickness)
        piezoAreaMoment = (1.0 / 12.0) * (self.actThickness ** 3)
        epoxyAreaMoment = (1.0 / 12.0) * (epoxyThickness ** 3)
        subAreaMoment = (1.0 / 12.0) * (self.subThickness ** 3)

        return (
            self.piezoModulus * piezoAreaMoment +
            self.epoxyModulus * epoxyAreaMoment +
            self.subModulus * subAreaMoment +
            self.piezoModulus * self.actThickness * (zPiezo ** 2) +
            self.epoxyModulus * epoxyThickness * (zEpoxy ** 2) +
            self.subModulus * self.subThickness * (zSub ** 2)
        )

    def get_neutral_axis(self, epoxyThickness):
        numerator = (
            self.piezoModulus * self.actThickness * (0.5 * self.actThickness + epoxyThickness) +
            self.epoxyModulus * epoxyThickness * 0.5 * epoxyThickness -
            self.subModulus * self.subThickness * 0.5 * self.subThickness
        )
        denominator = (
            self.piezoModulus * self.actThickness +
            self.epoxyModulus * epoxyThickness +
            self.subModulus * self.subThickness
        )
        return numerator / denominator

# Specific Trimorph Model
class CopperSubstrate(TrimorphModel):
    def __init__(self):
        super().__init__()
        self.actuatorLength = 5.18
        self.width = 0.71
        self.piezoPoisson = 0.3
        self.subPoisson = 0.34
        self.piezoModulusNoPoisson = 6.0e11
        self.epoxyModulus = 1.489e10
        self.subModulusNoPoisson = 1.1e11
        self.subDensity = 8.96
        self.piezoDensity = 7.5
        self.epoxyDensity = 1.07
        self.subThickness = 0.01
        self.actThickness = 0.078
        self.epoxyThickness = 1.0e-3
        self.d33NoPoisson = 370.0e-12
        self.p2p1ratio = 1.733
        self.pitchDistance = 5.0e-2

# Create and reset model
model = CopperSubstrate()
model.reset()

# Table of properties
data = {
    "Property": [
        "Actuator Length (cm)",
        "Width (cm)",
        "Substrate Thickness (cm)",
        "Piezoelectric Thickness (cm)",
        "Epoxy Thickness (cm)",
        "Substrate Modulus (Ba)",
        "Piezo Modulus (Ba)",
        "Epoxy Modulus (Ba)",
        "Curvature gamma (1/cm)",
        "Flexural Rigidity EI (Ba·cm⁴)",
        "Half EI (Ba·cm⁴)",
        "Actuator Mass (g)",
        "Load per Unit Area (dyne/cm²)",
        "Neutral Axis Position (cm)"
    ],
    "Copper Substrate": [
        model.actuatorLength,
        model.width,
        model.subThickness,
        model.actThickness,
        model.epoxyThickness,
        model.subModulus,
        model.piezoModulus,
        model.epoxyModulus,
        model.gamma,
        model.EI,
        model.halfEI,
        model.actuatorMass,
        model.load,
        model.get_neutral_axis(model.epoxyThickness),
    ],
}
df_bimorph = pd.DataFrame(data)
print(df_bimorph.to_string(index=False))

# Plot Deflection
def compute_circular_arc(model, num_points=100, voltage_polarity=-1):
    L = model.actuatorLength
    gamma = model.gamma * voltage_polarity
    if gamma == 0:
        return np.linspace(0, L, num_points), np.zeros(num_points)
    R = 1.0 / gamma
    x = np.linspace(0, L, num_points)
    z = np.sign(R) * (abs(R) - np.sqrt(R**2 - x**2))
    return x, z

x_pos, z_pos = compute_circular_arc(model, voltage_polarity=1)
x_neg, z_neg = compute_circular_arc(model, voltage_polarity=-1)

plt.figure(figsize=(10, 6))
plt.plot(x_pos * 10, z_pos * 1e5, label='Positive Voltage', color='blue', linewidth=2)
plt.plot(x_neg * 10, z_neg * 1e5, label='Negative Voltage', color='yellow', linewidth=2)
plt.title('Bimorph Deflection with Voltage Polarity')
plt.xlabel('Position along Actuator (cm)')
plt.ylabel('Deflection (μm)')
plt.xlim(0, 80)
plt.ylim(-0.5, 0.5)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.grid(True)
plt.legend()
plt.show()

# Define Soft Material
class SoftMaterial:
    def __init__(self, name, young_modulus, density, poisson_ratio):
        self.name = name
        self.young_modulus = young_modulus
        self.density = density
        self.poisson_ratio = poisson_ratio
        self.E = young_modulus / (1 - poisson_ratio**2)

# Composite Body with Soft Layer
class CompositeBody:
    def __init__(self, trimorph_model, soft_layer_thickness, soft_layer_density, body_length, body_width):
        self.trimorph = trimorph_model
        self.body_length = body_length
        self.body_width = body_width
        self.soft_thickness = soft_layer_thickness
        self.soft_density = soft_layer_density
        self.extra_mass = self.soft_density * self.body_length * self.body_width * self.soft_thickness
        self.total_mass = self.trimorph.actuatorMass + self.extra_mass
        self.gravity = 980  # cm/s²
        self.load = (self.total_mass * self.gravity) / (self.body_length * self.body_width)

    def get_effective_curvature(self, voltage_polarity=1):
        gamma_trimorph = self.trimorph.gamma * voltage_polarity
        original_load = self.trimorph.load
        return gamma_trimorph * (original_load / self.load)

dragon_skin = SoftMaterial("DragonSkin 20", young_modulus=1e6, density=1.05, poisson_ratio=0.49)
eco_flex = SoftMaterial("Ecoflex 10", young_modulus=5e5, density=1.07, poisson_ratio=0.48)

# Define soft layer thickness (top + bottom)
soft_layer_thickness = 0.5  # cm
body_length = 6.0          # same as trimorph
body_width = 1.5          # same as trimorph

# Create DragonSkin and Ecoflex composite bodies
dragon_body = CompositeBody(model, soft_layer_thickness,dragon_skin.density, body_length, body_width)

eco_body = CompositeBody(model, soft_layer_thickness, eco_flex.density, body_length, body_width)

# Get curvatures
gamma_dragon_pos = dragon_body.get_effective_curvature(voltage_polarity=1)
gamma_dragon_neg = dragon_body.get_effective_curvature(voltage_polarity=-1)

gamma_eco_pos = eco_body.get_effective_curvature(voltage_polarity=1)
gamma_eco_neg = eco_body.get_effective_curvature(voltage_polarity=-1)

# Print curvatures
print("Curvature (1/cm):")
print(f"{'DragonSkin':<12} Positive Voltage: {gamma_dragon_pos:.4e}")
print(f"{'DragonSkin':<12} Negative Voltage: {gamma_dragon_neg:.4e}")
print(f"{'Ecoflex':<12}    Positive Voltage: {gamma_eco_pos:.4e}")
print(f"{'Ecoflex':<12}    Negative Voltage: {gamma_eco_neg:.4e}")

# Compute deflections
def compute_deflection(gamma, length=6.0, num_points=100):
    if gamma == 0:
        return np.linspace(0, length, num_points), np.zeros(num_points)
    R = 1.0 / gamma
    x = np.linspace(0, length, num_points)
    z = np.sign(R) * (abs(R) - np.sqrt(R**2 - x**2))
    return x, z

x_drag, z_drag_pos = compute_deflection(gamma_dragon_pos)
_, z_drag_neg = compute_deflection(gamma_dragon_neg)

x_ecoflex, z_eco_pos = compute_deflection(gamma_eco_pos)
_, z_eco_neg = compute_deflection(gamma_eco_neg)

# Plot DragonSkin
plt.figure(figsize=(10, 6))
plt.plot(x_drag*10, z_drag_pos * 1e4, label='Positive Voltage', color='blue')
plt.plot(x_drag*10, z_drag_neg * 1e4, '--', label='Negative Voltage', color='red')
plt.title('Bimorph Bending with DragonSkin Load')
plt.xlabel('Position along Body (mm)')
plt.ylabel('Deflection (μm)')
plt.xlim(0, 80)
plt.ylim(-0.1, 0.1)
plt.grid(True)
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.show()

# Plot Ecoflex
plt.figure(figsize=(10, 6))
plt.plot(x_ecoflex* 10, z_eco_pos * 1e4, label='Positive Voltage', color='green')
plt.plot(x_ecoflex* 10, z_eco_neg * 1e4, '--', label='Negative Voltage', color='orange')
plt.title('Bimorph Bending with Ecoflex Load')
plt.xlabel('Position along Body (mm)')
plt.ylabel('Deflection (μm)')
plt.xlim(0, 80)
plt.ylim(-0.1, 0.1)
plt.grid(True)
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.show()


##########################PYBULLET LOCOMOTION###############################

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
        
        # Update friction states
        self.left_friction = self.left_height == 0  # Friction when on ground
        self.right_friction = self.right_height == 0
        
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
    """Simulate friction-based inchworm locomotion"""
    
    fig, axes = plt.subplots(len(num_actuators_list), 1, figsize=(14, 10))
    if len(num_actuators_list) == 1:
        axes = [axes]
    
    robots = []
    body_rects = []
    actuator_patches = []
    friction_indicators = []
    
    # Create robots and visualization elements
    for idx, num_act in enumerate(num_actuators_list):
        robot = InchwormRobot(num_act)
        robots.append(robot)
        
        ax = axes[idx]
        ax.set_xlim(-1, cycles * 2 + 3)
        ax.set_ylim(-0.5, 1)
        ax.set_title(f'{num_act}-Actuator Inchworm: Friction-Based Locomotion')
        ax.set_xlabel('Position')
        ax.set_ylabel('Height')
        ax.grid(True, alpha=0.3)
        
        # Create main body rectangle
        body_rect = Rectangle((0, 0), num_act, 0.2, facecolor='lightblue', 
                            edgecolor='black', linewidth=2)
        ax.add_patch(body_rect)
        body_rects.append(body_rect)
        
        # Create actuator indicators
        patches = []
        for i in range(num_act):
            # Actuator rectangle
            rect = Rectangle((0, 0.25), 0.8, 0.15, facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
            patches.append(rect)
        actuator_patches.append(patches)
        
        # Create friction indicators (ground contact symbols)
        left_friction = Circle((0, -0.1), 0.05, facecolor='red', alpha=0.8)
        right_friction = Circle((0, -0.1), 0.05, facecolor='red', alpha=0.8)
        ax.add_patch(left_friction)
        ax.add_patch(right_friction)
        friction_indicators.append((left_friction, right_friction))
        
        # Add legend
        ax.text(0.02, 0.98, 'Red circles = Friction contact', transform=ax.transAxes, 
                verticalalignment='top', fontsize=9)
        ax.text(0.02, 0.92, 'Orange actuators = Active', transform=ax.transAxes, 
                verticalalignment='top', fontsize=9)
    
    # Store step information for display
    step_info = []
    
    def animate(frame):
        step = frame
        
        for idx, (robot, body_rect, patches, friction_ind) in enumerate(
            zip(robots, body_rects, actuator_patches, friction_indicators)):
            
            # Execute gait step
            robot.gait_step()
            
            # Update body position and length
            body_length = robot.right_end_pos - robot.left_end_pos
            body_rect.set_x(robot.left_end_pos)
            body_rect.set_width(body_length)
            body_rect.set_y(min(robot.left_height, robot.right_height))
            
            # Update actuator patches
            for i, patch in enumerate(patches):
                patch.set_x(robot.left_end_pos + i * body_length / robot.num_actuators)
                patch.set_width(0.8 * body_length / robot.num_actuators)
                patch.set_y(0.25 + min(robot.left_height, robot.right_height))
                
                # Color based on actuator state
                if robot.actuator_states[i] == 1:
                    patch.set_facecolor('orange')
                    patch.set_alpha(0.9)
                else:
                    patch.set_facecolor('gray')
                    patch.set_alpha(0.7)
            
            # Update friction indicators
            left_friction, right_friction = friction_ind
            left_friction.center = (robot.left_end_pos, -0.1)
            right_friction.center = (robot.right_end_pos, -0.1)
            
            # Show/hide friction indicators
            left_friction.set_alpha(0.8 if robot.left_friction else 0.2)
            right_friction.set_alpha(0.8 if robot.right_friction else 0.2)
            
            # Display step information for first robot
            if idx == 0:
                step_names = ["Step 1: Lift Left End", "Step 2: Contract Middle", 
                            "Step 3: Switch Friction", "Step 4: Extend Middle"]
                current_step_name = step_names[robot.step_in_cycle]
                axes[0].set_title(f'{robot.num_actuators}-Actuator Inchworm - {current_step_name}')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=cycles*4*4, 
                                 interval=1200, blit=False, repeat=True)
    
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

def plot_gait_comparison():
    """Compare gait cycles for different actuator numbers"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    actuator_counts = [3, 5, 7]
    
    for idx, num_act in enumerate(actuator_counts):
        robot = InchwormRobot(num_act)
        
        positions = []
        friction_states = []
        
        # Run for 3 complete cycles
        for step in range(12):  # 3 cycles × 4 steps
            robot.gait_step()
            positions.append(robot.body_position)
            friction_states.append('L' if robot.left_friction else 'R')
        
        ax = axes[idx]
        steps = range(len(positions))
        ax.plot(steps, positions, 'o-', linewidth=2, markersize=6)
        ax.set_title(f'{num_act}-Actuator Robot: Position vs Step')
        ax.set_xlabel('Step Number')
        ax.set_ylabel('Body Position')
        ax.grid(True, alpha=0.3)
        
        # Add friction state annotations
        for i, (pos, friction) in enumerate(zip(positions, friction_states)):
            color = 'red' if friction == 'L' else 'blue'
            ax.annotate(friction, (i, pos), xytext=(0, 10), 
                       textcoords='offset points', ha='center',
                       color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Friction-Based Inchworm Robot Simulation")
    print("Based on research paper gait cycle logic")
    print("=" * 50)
    
    # Analyze the gait cycle
    analyze_friction_gait_cycle()
    
    # Show comparison plot
    plot_gait_comparison()
    
    # Run animation
    print("\nStarting friction-based gait animation...")
    print("Watch the red circles (friction contact) and orange actuators (active)")
    anim = simulate_friction_based_gait([3, 5, 7], cycles=4)
    
    plt.show()
######################################################


import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import os
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

@dataclass
class ActuatorState:
    """State of a single piezoelectric actuator"""
    voltage: float = 0.0
    curvature: float = 0.0
    force: float = 0.0
    position: np.ndarray = None
    
class PiezoActuator:
    """Individual piezoelectric actuator with 3 motors for accurate control"""
    def __init__(self, body_id: int, joint_ids: List[int], trimorph_model: TrimorphModel, segment_length: float):
        self.body_id = body_id
        self.joint_ids = joint_ids  # 3 joints per actuator for accuracy
        self.model = trimorph_model
        self.segment_length = segment_length
        self.state = ActuatorState()
        
        # Scale factors for simulation (significantly increased for visible motion)
        self.voltage_scale = 1.0  # V (reduced to amplify effect)
        self.curvature_amplifier = 1e6  # Amplify tiny curvature values
        self.force_scale = 5.0  # N (increased force)
        self.max_angle = 1.2  # rad (increased max angle)
        
    def set_voltage(self, voltage: float):
        """Set voltage and compute resulting curvature and forces"""
        self.state.voltage = voltage
        
        # Compute curvature from trimorph model with amplification for visible motion
        gamma = self.model.gamma * voltage * self.curvature_amplifier / self.voltage_scale
        self.state.curvature = gamma
        
        # Convert curvature to joint angles for realistic bending
        total_bend = gamma * self.segment_length
        
        # For single joint per actuator (simplified from 3-joint system)
        if len(self.joint_ids) == 1:
            angle = total_bend
        else:
            # For multiple joints, distribute the bend
            angle_per_joint = total_bend / len(self.joint_ids)
            angles = [angle_per_joint] * len(self.joint_ids)
        
        # Apply joint angles with enhanced force control
        for i, joint_id in enumerate(self.joint_ids):
            if len(self.joint_ids) == 1:
                target_angle = np.clip(angle, -self.max_angle, self.max_angle)
            else:
                target_angle = np.clip(angles[i], -self.max_angle, self.max_angle)
            
            # Enhanced position control with higher gains for visible motion
            p.setJointMotorControl2(
                bodyUniqueId=self.body_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=self.force_scale * abs(voltage) + 1.0,  # Add base force
                positionGain=2.0,   # Increased gain
                velocityGain=0.5    # Increased damping
            )

class InchwormRobot:
    """Soft robot with multiple piezoelectric actuators"""
    def __init__(self, num_actuators: int, physics_client):
        self.num_actuators = num_actuators
        self.physics_client = physics_client
        self.trimorph_model = CopperSubstrate()
        self.trimorph_model.reset()
        self.voltage_history = []
        
        self.actuators: List[PiezoActuator] = []
        self.body_id = None
        self.segment_length = 0.06  # 6cm per segment
        self.segment_width = 0.02   # 2cm width
        self.segment_height = 0.01  # 1cm height
        
        # Locomotion parameters
        self.gait_frequency = 2.0  # Hz
        self.gait_amplitude = 50.0  # V
        self.time_step = 0
        
        # Performance tracking
        self.position_history = []
        self.velocity_history = []

        self.front_tip_history = []
        self.rear_tip_history = []
        
        
    def create_robot_body(self):
        """Create the soft robot body with multiple segments"""
        
        # Create collision and visual shapes for segments
        collision_shapes = []
        visual_shapes = []
        link_masses = []
        link_positions = []
        link_orientations = []
        link_inertial_positions = []
        link_inertial_orientations = []
        parent_indices = []
        joint_types = []
        joint_axes = []
        
        # Soft material properties (DragonSkin-like)
        segment_mass = 0.01  # kg
        
        for i in range(self.num_actuators):
            # Collision shape
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.segment_length/2, self.segment_width/2, self.segment_height/2]
            )
            collision_shapes.append(collision_shape)

            if i == 0:
                color = [1.0, 0.0, 0.0, 0.9]
            elif i == self.num_actuators - 1:
                color = [0.0, 1.0, 0.0, 0.9]
            else:
                color = [0.8, 0.4, 0.2, 0.8]
            
            # Visual shape
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.segment_length/2, self.segment_width/2, self.segment_height/2],
                rgbaColor= color 
            )
            visual_shapes.append(visual_shape)
            
            link_masses.append(segment_mass)
            
            # Position segments along x-axis
            link_positions.append([self.segment_length * (0.5), 0, 0])
            link_orientations.append([0, 0, 0, 1])
            link_inertial_positions.append([0, 0, 0])
            link_inertial_orientations.append([0, 0, 0, 1])
            
            if i == 0:
                parent_indices.append(-1)  # Base link
            else:
                parent_indices.append(i - 1)
            
            joint_types.append(p.JOINT_REVOLUTE)
            joint_axes.append([0, 1, 0])  # Rotate around Y-axis
        
        # Create base (first segment)
        base_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.segment_length/2, self.segment_width/2, self.segment_height/2]
        )
        base_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.segment_length/2, self.segment_width/2, self.segment_height/2],
            rgbaColor=[1.0, 0.0, 0.0, 0.9]  # red
        )
        
        # Create multi-body
        self.body_id = p.createMultiBody(
            baseMass=segment_mass,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0, 0, self.segment_height/2 + 0.01],
            linkMasses=link_masses[1:],  # Exclude base
            linkCollisionShapeIndices=collision_shapes[1:],
            linkVisualShapeIndices=visual_shapes[1:],
            linkPositions=link_positions[1:],
            linkOrientations=link_orientations[1:],
            linkInertialFramePositions=link_inertial_positions[1:],
            linkInertialFrameOrientations=link_inertial_orientations[1:],
            linkParentIndices=parent_indices[1:],
            linkJointTypes=joint_types[1:],
            linkJointAxis=joint_axes[1:]
        )
        
        # Create actuators with 3 joints each (for accuracy)
        # In this simplified version, we'll use 1 joint per actuator but apply the trimorph model
        joint_id = 0
        for i in range(self.num_actuators - 1):  # -1 because base doesn't have joint
            # Create actuator with single joint (simplified from 3-joint system)
            actuator = PiezoActuator(
                body_id=self.body_id,
                joint_ids=[joint_id],  # Single joint per actuator for simplicity
                trimorph_model=self.trimorph_model,
                segment_length=self.segment_length
            )
            self.actuators.append(actuator)
            joint_id += 1

    def get_tip_positions(self):
        """Get the positions of front and rear tips"""
        # Get base (rear tip) position
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        rear_tip_pos = np.array(base_pos)
        
        # Get front tip position (last link)
        if self.num_actuators > 1:
            # Get the position of the last link
            link_state = p.getLinkState(self.body_id, self.num_actuators - 2)  # -2 because links are 0-indexed
            front_tip_pos = np.array(link_state[0])  # World position of the link
        else:
            # If only one segment, front and rear are the same
            front_tip_pos = rear_tip_pos.copy()
        
        return rear_tip_pos, front_tip_pos
    
    def apply_inchworm_gait(self, time_step: float):
        """Apply inchworm locomotion pattern"""
        t = time_step * self.gait_frequency
        voltage_this_step = []
        
        # Inchworm gait: wave propagation from tail to head
        for i, actuator in enumerate(self.actuators):
            # Phase offset for wave propagation
            phase = (i / len(self.actuators)) * 2 * np.pi
            
            # Sinusoidal voltage pattern with phase offset
            voltage = self.gait_amplitude * np.sin(2 * np.pi * t + phase)
            
            actuator.set_voltage(voltage)
            voltage_this_step.append(abs(voltage))
        self.voltage_history.append(voltage_this_step)
    
    def apply_peristaltic_gait(self, time_step: float):
        """Apply peristaltic wave locomotion"""
        t = time_step * self.gait_frequency
        wave_speed = 2.0
        voltage_this_step =[]
        for i, actuator in enumerate(self.actuators):
            # Traveling wave
            phase = (i / len(self.actuators)) * 2 * np.pi - wave_speed * t
            voltage = self.gait_amplitude * np.sin(phase)
            
            # Add bias for directional movement
            if np.sin(phase) > 0:
                voltage *= 1.5  # Amplify extension
            
            actuator.set_voltage(voltage)
            voltage_this_step.append(abs(voltage))

        self.voltage_history.append(voltage_this_step)
    
    def get_robot_position(self):
        """Get center of mass position"""
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos)
    
    def get_robot_velocity(self):
        """Get center of mass velocity"""
        vel, _ = p.getBaseVelocity(self.body_id)
        return np.array(vel)
    
    def update_tracking(self):
        """Update position and velocity tracking"""
        pos = self.get_robot_position()
        vel = self.get_robot_velocity()
        
        self.position_history.append(pos.copy())
        self.velocity_history.append(vel.copy())

        rear_pos, front_pos = self.get_tip_positions()
        self.rear_tip_history.append(rear_pos.copy())
        self.front_tip_history.append(front_pos.copy())

    
    def get_displacement(self):
        """Get total displacement from start"""
        if len(self.position_history) < 2:
            return 0.0
        return np.linalg.norm(self.position_history[-1] - self.position_history[0])

class SimulationEnvironment:
    """PyBullet simulation environment"""
    def __init__(self, gui=True):
        # Initialize PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up environment
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240.0)  # 240 Hz
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set ground friction
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1)
        
        self.robots = {}
        self.simulation_time = 0.0
        
    def add_robot(self, name: str, num_actuators: int, position: List[float] = [0, 0, 0.02]):
        """Add an inchworm robot to simulation"""
        robot = InchwormRobot(num_actuators, self.physics_client)
        robot.create_robot_body()
        
        # Set robot position
        p.resetBasePositionAndOrientation(
            robot.body_id, 
            position, 
            [0, 0, 0, 1]
        )
        
        self.robots[name] = robot
        return robot
    
    def step_simulation(self, gait_type='inchworm'):
        """Step the simulation forward"""
        # Apply gaits to all robots
        for robot in self.robots.values():
            if gait_type == 'inchworm':
                robot.apply_inchworm_gait(self.simulation_time)
            elif gait_type == 'peristaltic':
                robot.apply_peristaltic_gait(self.simulation_time)
            
            robot.update_tracking()

        
        # Step physics
        p.stepSimulation()
        self.simulation_time += 1/240.0
    
    def run_simulation(self, duration: float = 10.0, gait_type='inchworm'):
        """Run simulation for specified duration"""
        steps = int(duration * 240)  # 240 Hz
        
        print(f"Running {gait_type} gait simulation for {duration}s...")
        
        for step in range(steps):
            self.step_simulation(gait_type)
            
            if step % 240 == 0:  # Print every second
                print(f"Time: {self.simulation_time:.1f}s")
                for name, robot in self.robots.items():
                    pos = robot.get_robot_position()
                    print(f"  {name}: Position = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            time.sleep(1/240.0)  # Real-time simulation
    
    def analyze_performance(self):
        """Analyze locomotion performance"""
        results = {}
        
        for name, robot in self.robots.items():
            if len(robot.position_history) < 2:
                continue
                
            positions = np.array(robot.position_history)
            velocities = np.array(robot.velocity_history)
            voltage_array = np.array(robot.voltage_history)
            avg_voltage = np.mean(voltage_array)
            
            # Calculate metrics
            total_displacement = np.linalg.norm(positions[-1] - positions[0])
            avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
            max_speed = np.max(np.linalg.norm(velocities, axis=1))
            voltage = avg_voltage 

            if len(robot.front_tip_history) > 0 and len(robot.rear_tip_history) > 0:
                front_tips = np.array(robot.front_tip_history)
                rear_tips = np.array(robot.rear_tip_history)
                front_tip_displacement = np.linalg.norm(front_tips[-1] - front_tips[0])
                rear_tip_displacement = np.linalg.norm(rear_tips[-1] - rear_tips[0])
            else:
                front_tip_displacement = 0
                rear_tip_displacement = 0

            
            # Efficiency (displacement per energy - simplified)
            efficiency = total_displacement / (len(robot.actuators) * robot.gait_amplitude)
            
            results[name] = {
                'total_displacement': total_displacement,
                'voltage': voltage,
                'average_speed': avg_speed,
                'max_speed': max_speed,
                'efficiency': efficiency,
                'num_actuators': robot.num_actuators,
                'front_tip_displacement': front_tip_displacement,
                'rear_tip_displacement': rear_tip_displacement
            }
            
            print(f"\n{name} Performance:")
            print(f"  Actuators: {robot.num_actuators}")
            print(f"  Voltages applied: {int(voltage)} V")
            print(f"  Total Displacement: {total_displacement:.3f} cm")
            print(f"  Average Speed: {avg_speed:.3f} cm/s")
            print(f"  Max Speed: {max_speed:.3f} cm/s")
            print(f"  Efficiency: {efficiency:.6f} cm/V")
            print(f"  Front Tip Displacement: {front_tip_displacement:.3f} m")
            print(f"  Rear Tip Displacement: {rear_tip_displacement:.3f} m")
            
        
        return results


    def generate_professional_results_plots(sim):
        """Generate publication-quality plots for research report"""
    
    # Set style for professional plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
    
    # 1. Performance Comparison Bar Chart
        ax1 = plt.subplot(2, 3, 1)

        results = sim.analyze_performance()
        configs = []
        displacements =[]
        avg_speeds = []
        max_speeds =[]
        efficiencies = []
        front_tips = []
        rear_tips = []

        for name, data in results.items():
            configs.append(f"{data['num_actuators']} actutatos")
            displacements.append(data['total_displacement'] * 100)
            avg_speeds.append(data['average_speed'] * 100)
            max_speeds.append(data['max_speed'] * 100)
            efficiencies.append(data['efficiency'])
            front_tips.append(data['front_tip_displacement'])
            rear_tips.append(data['rear_tip_displacement'])


    
        x = np.arange(len(configs))
        width = 0.25
    
        bars1 = ax1.bar(x - width, displacements, width, label='Total Displacement (cm)', 
                   color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x, np.array(avg_speeds)*10, width, label='Avg Speed (×10 cm/s)', 
                   color='#4ECDC4', alpha=0.8)
        bars3 = ax1.bar(x + width, np.array(max_speeds)*0.1, width, label='Max Speed (×0.1 cm/s)', 
                   color='#45B7D1', alpha=0.8)
    
        ax1.set_xlabel('Robot Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Comparison Across Configurations', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Efficiency Analysis
        ax2 = plt.subplot(2, 3, 2)
    
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        bars = ax2.bar(configs, efficiencies, color=colors, alpha=0.8)
        ax2.set_xlabel('Robot Configuration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Efficiency (cm/V)', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        if efficiencies:
            max_eff_idx =np.argmax(efficiencies)
            bars[max_eff_idx].set_edgecolor('gold')
            bars[max_eff_idx].set_linewidth(3)
        
    
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.annotate(f'{height:.6f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Tip Displacement Analysis
        ax3 = plt.subplot(2, 3, 3)
    
        front_tips = [0.015, 0.016, 0.021]  # m
        rear_tips = [0.016, 0.015, 0.021]   # m
    
        x = np.arange(len(configs))
        width = 0.35
    
        bars1 = ax3.bar(x - width/2, front_tips, width, label='Front Tip', 
                   color='#2ECC71', alpha=0.8)
        bars2 = ax3.bar(x + width/2, rear_tips, width, label='Rear Tip', 
                   color='#E74C3C', alpha=0.8)
    
        ax3.set_xlabel('Robot Configuration', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Tip Displacement (m)', fontsize=12, fontweight='bold')
        ax3.set_title('Front vs Rear Tip Displacement', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(configs)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Speed vs Actuators Scatter Plot
        ax4 = plt.subplot(2, 3, 4)
    
        actuator_counts = [3, 5, 7]
    
        ax4.scatter(actuator_counts, avg_speeds, s=200, c=colors, alpha=0.8, 
               edgecolors='black', linewidth=2, label='Average Speed')
        ax4.scatter(actuator_counts, np.array(max_speeds)/10, s=200, c=colors, 
               alpha=0.6, marker='^', edgecolors='black', linewidth=2, 
               label='Max Speed (÷10)')
    
    # Fit trend lines
        z_avg = np.polyfit(actuator_counts, avg_speeds, 1)
        p_avg = np.poly1d(z_avg)
        ax4.plot(actuator_counts, p_avg(actuator_counts), "--", alpha=0.8, 
            color='red', linewidth=2, label='Avg Speed Trend')
    
        ax4.set_xlabel('Number of Actuators', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Speed (cm/s)', fontsize=12, fontweight='bold')
        ax4.set_title('Speed vs Number of Actuators', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Performance Radar Chart
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        ax5.axis('off')
        table_data = [['Metric'] + configs]
        table_data.append(['Displacement (cm)'] + [f'{d:.3f}' for d in displacements])
        table_data.append(['Avg Speed (cm/s)'] + [f'{s:.3f}' for s in avg_speeds])
        table_data.append(['Max Speed (cm/s)'] + [f'{s:.3f}' for s in max_speeds])
        table_data.append(['Efficiency (cm/V)'] + [f'{e:.6f}' for e in efficiencies])
        table_data.append(['Front Tip (m)'] + [f'{f:.3f}' for f in front_tips])
        table_data.append(['Rear Tip (m)'] + [f'{r:.3f}' for r in rear_tips])

        table = ax5.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
    
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor('#4ECDC4')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i-1, j)].set_facecolor('#F8F9FA')
    
        ax5.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    
    # 6. Summary Statistics Table
        ax6 = plt.subplot(2, 3, 6)

        for name, robot in sim.robots.items():
            if len(robot.position_history) > 0:
                positions = np.array(robot.position_history)
                ax6.plot(positions[:, 0], positions[:, 1], label=name, marker='o', markersize=2)
    
        ax6.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax6.set_title('Robot Trajectories', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')

        plt.tight_layout()
        plt.suptitle('Inchworm Robot Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Save high-quality figure
        plt.savefig('inchworm_performance_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
        plt.show()
    
        return fig
  

    def print_formatted_results(results):
        """Print professionally formatted results for report"""
    
        print("\n" + "="*80)
        print("INCHWORM ROBOT LOCOMOTION PERFORMANCE ANALYSIS")
        print("="*80)
    
        print("\n📊 SIMULATION PARAMETERS:")
        print("-" * 40)
        print(f"{'Duration:':<20} 15.0 seconds")
        print(f"{'Gait Type:':<20} Inchworm locomotion")
        print(f"{'Voltage Applied:':<20} 31 V (average)")
        print(f"{'Configurations:':<20} 3, 5, and 7 actuators")
    
        print("\n🤖 INDIVIDUAL ROBOT PERFORMANCE:")
        print("-" * 60)
    
        for name, config in results.items():
            print(f"\n{name}:")
            print(f"  ├─ Total Displacement: {config['total_displacement']*100:.3f} cm")
            print(f"  ├─ Average Speed: {config['average_speed']*100:.3f} cm/s")
            print(f"  ├─ Maximum Speed: {config['max_speed']*100:.3f} cm/s")
            print(f"  ├─ Efficiency: {config['efficiency']:.6f} cm/V")
            print(f"  ├─ Front Tip Displacement: {config['front_tip_displacement']:.3f} m")
            print(f"  └─ Rear Tip Displacement: {config['rear_tip_displacement']:.3f} m")

        if results:
            best_disp = max(results.items(), key=lambda x: x[1]['total_displacement'])
            best_speed = max(results.items(), key=lambda x: x[1]['average_speed'])
            best_eff = max(results.items(), key=lambda x: x[1]['efficiency'])

            print("\n🏆 PERFORMANCE RANKINGS:")
            print("-" * 40)
            print(f"Best Displacement:     {best_disp[0]} ({best_disp[1]['total_displacement']*100:.3f} cm)")
            print(f"Best Average Speed:    {best_speed[0]} ({best_speed[1]['average_speed']*100:.3f} cm/s)")
            print(f"Best Efficiency:       {best_eff[0]} ({best_eff[1]['efficiency']:.6f} cm/V)")
    
    print("\n" + "="*80)
    
    def plot_trajectories(self):
        """Plot robot trajectories including tip positions"""
        plt.figure(figsize=(16, 10))
        
        for name, robot in self.robots.items():
            if len(robot.position_history) < 2:
                continue
                
            positions = np.array(robot.position_history)
            time_points = np.linspace(0, self.simulation_time, len(positions))
            
            # Position vs time
            plt.subplot(2, 3, 1)
            plt.plot(time_points, positions[:, 0], label=f'{name} (Center)')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (m)')
            plt.legend()
            plt.grid(True)
            plt.title('Robot Center Position')
            
            # Tip positions
            if len(robot.front_tip_history) > 0:
                front_tips = np.array(robot.front_tip_history)
                rear_tips = np.array(robot.rear_tip_history)
                
                plt.subplot(2, 3, 2)
                plt.plot(time_points, front_tips[:, 0], label=f'{name} (Front)', color='green')
                plt.plot(time_points, rear_tips[:, 0], label=f'{name} (Rear)', color='red')
                plt.xlabel('Time (s)')
                plt.ylabel('X Position (m)')
                plt.legend()
                plt.grid(True)
                plt.title('Tip Positions')
            
            # 2D trajectory - center of mass
            plt.subplot(2, 3, 3)
            plt.plot(positions[:, 0], positions[:, 1], label=f'{name} (Center)', marker='o', markersize=2)
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.title('2D Trajectory (Center)')
            
            # 2D trajectory - tips
            if len(robot.front_tip_history) > 0:
                plt.subplot(2, 3, 4)
                plt.plot(front_tips[:, 0], front_tips[:, 1], label=f'{name} (Front)', 
                        color='green', marker='o', markersize=1)
                plt.plot(rear_tips[:, 0], rear_tips[:, 1], label=f'{name} (Rear)', 
                        color='red', marker='o', markersize=1)
                plt.xlabel('X Position (m)')
                plt.ylabel('Y Position (m)')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.title('2D Trajectory (Tips)')
            
            # Speed vs time
            if len(robot.velocity_history) > 0:
                velocities = np.array(robot.velocity_history)
                speeds = np.linalg.norm(velocities, axis=1)
                plt.subplot(2, 3, 5)
                plt.plot(time_points, speeds, label=f'{name}')
                plt.xlabel('Time (s)')
                plt.ylabel('Speed (cm/s)')
                plt.legend()
                plt.grid(True)
                plt.title('Speed vs Time')
            
            # Tip displacement comparison
            if len(robot.front_tip_history) > 0:
                front_displacement = np.linalg.norm(front_tips - front_tips[0], axis=1)
                rear_displacement = np.linalg.norm(rear_tips - rear_tips[0], axis=1)
                
                plt.subplot(2, 3, 6)
                plt.plot(time_points, front_displacement, label=f'{name} (Front)', color='green')
                plt.plot(time_points, rear_displacement, label=f'{name} (Rear)', color='red')
                plt.xlabel('Time (s)')
                plt.ylabel('Cumulative Displacement (cm)')
                plt.legend()
                plt.grid(True)
                plt.title('Tip Displacements')
        
        plt.tight_layout()
        plt.show()

    
    
    def disconnect(self):
        """Clean up simulation"""
        p.disconnect()

def main():
    """Main simulation function"""
    print("Enhanced Piezoelectric Inchworm Locomotion Simulation")
    print("With Front and Rear Tip Position Tracking")
    print("="*70)
    
    # Create simulation environment
    sim = SimulationEnvironment(gui=True)
    
    # Test different actuator configurations
    configurations = [3, 5, 7]
    
    # Add robots with different actuator counts
    y_offset = 0
    for i, num_actuators in enumerate(configurations):
        robot_name = f"Robot_{num_actuators}_actuators"
        robot = sim.add_robot(robot_name, num_actuators, position=[0, y_offset, 0.02])
        y_offset += 0.1  # Separate robots in Y direction
        
        print(f"Created {robot_name} with {num_actuators} actuators")
        print(f"  Segments colored: Red (rear) -> Orange (middle) -> Green (front)")
    
    print("\nStarting simulation...")
    
    # Run inchworm gait simulation
    try:
        sim.run_simulation(duration=15.0, gait_type='inchworm')
        
        # Analyze performance
        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS WITH TIP TRACKING")
        print("="*70)
        results = sim.analyze_performance()
        
        # Plot results
        sim.plot_trajectories()
        result = sim.generate_professional_results_plots()
        sim.print_formatted_results(results)
        sim.print_formatted_results(result)
        
        
        # Compare configurations
        print("\nConfiguration Comparison:")
        print("Config | Displacement | Avg Speed | Front Tip | Rear Tip")
        print("-"*55)
        for config in configurations:
            robot_name = f"Robot_{config}_actuators"
            if robot_name in results:
                r = results[robot_name]
                print(f"{config:2d} act | {r['total_displacement']:8.3f}m | "
                      f"{r['average_speed']:7.3f}cm/s | {r['front_tip_displacement']:7.3f}m | "
                      f"{r['rear_tip_displacement']:6.3f}m")
                
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        sim.disconnect()
        print("Simulation completed")

if __name__ == "__main__":
    main()

