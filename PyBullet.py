import pybullet as p
import numpy as np
import math

class TrimorphModel(object):
    def __init__(self):
        self.gravity = 9.81
        self.actuatorLength = None
        self.width = None
        self.piezoPoisson = None
        self.subPoisson = None
        self.piezoModulusNoPoisson = None  # in CGS, 1 Ba = 0.1 Pa
        self.epoxyModulus = None  # in CGS, 1 Ba = 0.1 Pa
        self.subModulusNoPoisson = None
        self.subDensity = None
        self.piezoDensity = None
        self.epoxyDensity = None
        self.subThickness = None
        self.actThickness = None
        self.epoxyThickness = None
        self.d33NoPoisson = None
        self.p2p1ratio = None
        self.pitchDistance = None

    def reset(self):
        self.piezoModulus = self.piezoModulusNoPoisson / (1 - (self.piezoPoisson ** 2))  # in CGS, 1 Ba = 0.1 Pa
        self.subModulus = self.subModulusNoPoisson / (1 - (self.subPoisson ** 2))
        self.d33 = self.d33NoPoisson * (1 + self.piezoPoisson)
        self.actuator2DDensity = self.subDensity * self.subThickness + self.piezoDensity * self.actThickness + self.epoxyDensity * self.epoxyThickness
        self.actuator1DDensity = self.actuator2DDensity * self.width
        self.actuatorMass = self.actuator1DDensity * self.actuatorLength
        self.voltageExpansion = self.d33 / self.pitchDistance
        self.gamma = self.get_curvature(epoxyThickness=self.epoxyThickness)
        self.beta = self.gamma * self.actuatorLength
        self.EI = self.get_flexural_rigidty(epoxyThickness=self.epoxyThickness)
        self.halfEI = self.EI / 2.0
        self.load = self.actuatorMass * self.gravity / (self.actuatorLength * self.width)

    def get_curvature(self, epoxyThickness):
        EI = self.get_flexural_rigidty(epoxyThickness=epoxyThickness)
        zPiezo, _, _ = self.get_relative_position(epoxyThickness=epoxyThickness)
        gamma = self.voltageExpansion * zPiezo * self.piezoModulus * self.actThickness / EI
        return gamma

    def get_relative_position(self, epoxyThickness):
        neutralAxis = self.get_neutral_axis(epoxyThickness=epoxyThickness)
        zPiezo = 0.5 * self.actThickness + epoxyThickness - neutralAxis
        zEpoxy = 0.5 * epoxyThickness - neutralAxis
        zSub = -0.5 * self.subThickness - neutralAxis
        return zPiezo, zEpoxy, zSub

    def get_flexural_rigidty(self, epoxyThickness):
        zPiezo, zEpoxy, zSub = self.get_relative_position(epoxyThickness)
        piezoAreaMoment = 1.0 / 12.0 * (self.actThickness ** 3)
        epoxyAreaMoment = 1.0 / 12.0 * (epoxyThickness ** 3)
        subAreaMoment = 1.0 / 12.0 * (self.subThickness ** 3)
        EI = self.piezoModulus * piezoAreaMoment + self.epoxyModulus * epoxyAreaMoment + self.subModulus * subAreaMoment \
             + self.piezoModulus * self.actThickness * (zPiezo ** 2) + self.epoxyModulus * epoxyThickness * (
                         zEpoxy ** 2) \
             + self.subModulus * self.subThickness * (zSub ** 2)
        return EI

    def get_neutral_axis(self, epoxyThickness):
        neutralAxis = (self.piezoModulus * self.actThickness * (0.5 * self.actThickness + epoxyThickness)
                       + self.epoxyModulus * epoxyThickness * 0.5 * epoxyThickness
                       - self.subModulus * self.subThickness * 0.5 * self.subThickness) \
                      / (
                              self.piezoModulus * self.actThickness + self.epoxyModulus * epoxyThickness + self.subModulus * self.subThickness)
        return neutralAxis


class TrimorphModelParameter2(TrimorphModel):

    def __init__(self):
        TrimorphModel.__init__(self)
        self.actuatorLength = 2.5
        self.width = 1.3
        self.piezoPoisson = 0.29
        self.subPoisson = 0.34
        self.piezoModulusNoPoisson = 2.0e9  # in CGS, 1 Ba = 0.1 Pa
        self.epoxyModulus = 2.4e9  # in CGS, 1 Ba = 0.1 Pa
        self.subModulusNoPoisson = 2.5e9
        self.subDensity = 1.4
        self.piezoDensity = 1.78
        self.epoxyDensity = 12
        self.subThickness = 25.4e-4
        self.actThickness = 28.0e-4
        self.epoxyThickness = 10.0e-4
        self.d33NoPoisson = 20e-12
        self.p2p1ratio = 1.0
        self.pitchDistance = 0.5


class TrimorphModelParameter4(TrimorphModelParameter2):
    def __init__(self):
        TrimorphModelParameter2.__init__(self)
        self.piezoModulusNoPoisson = 2.0e9  # in CGS, 1 Ba = 0.1 Pa
        self.subModulusNoPoisson = 2.5e9
        self.subDensity = 1.4
        self.piezoDensity = 1.78
        self.actThickness = 25.0e-4
        self.d33NoPoisson = 20e-12


class modelParameter(TrimorphModelParameter4):
    def __init__(self, parameter='parameter 2'):
        if parameter == 'trimorph parameter 2':
            superclass = TrimorphModelParameter2
        elif parameter == 'trimorph parameter 4':
            superclass = TrimorphModelParameter4
        else:
            raise NotImplementedError('Model parameter not implemented')
        self.parameter = parameter
        self.superclass = superclass
        superclass.__init__(self)

    def reset(self):
        superclass = self.superclass
        superclass.reset(self)

    def get_curvature(self, *args, **kwargs):
        superclass = self.superclass
        return superclass.get_curvature(self, *args, **kwargs)

    def get_relative_position(self, *args, **kwargs):
        superclass = self.superclass
        return superclass.get_relative_position(self, *args, **kwargs)

    def get_flexural_rigidty(self, *args, **kwargs):
        superclass = self.superclass
        return superclass.get_flexural_rigidty(self, *args, **kwargs)

    def get_neutral_axis(self, *args, **kwargs):
        superclass = self.superclass
        return superclass.get_neutral_axis(self, *args, **kwargs)

# Robot class that builds the piezo 1D structure
class RobotBase(modelParameter):
    def __init__(self, parameter='trimorph parameter 2'):
        modelParameter.__init__(self, parameter=parameter)
        self._p = p

    def create_1d_multi_actuators(self,
                                  actuatorNumber,
                                  unitMotorNumber,
                                  actuatorMass,
                                  actuatorLength,
                                  actuatorWidth,
                                  actuatorThickness,
                                  basePosition=(0, 0, 0),
                                  baseOrientation=(0, 0, 0, 1)):
        N = actuatorNumber
        m = unitMotorNumber
        thickness = actuatorThickness
        width = actuatorWidth
        linkLength = actuatorLength / m
        jointLength = 0.5 * linkLength

        startBoxId = self._p.createCollisionShape(self._p.GEOM_BOX,
                                                  halfExtents=[0.5 * jointLength, 0.5 * width, 0.5 * thickness])
        linkBoxId = self._p.createCollisionShape(self._p.GEOM_BOX,
                                                 halfExtents=[0.5 * linkLength, 0.5 * width, 0.5 * thickness],
                                                 collisionFramePosition=[0.5 * linkLength, 0, 0])
        endBoxId = self._p.createCollisionShape(self._p.GEOM_BOX,
                                                halfExtents=[0.5 * jointLength, 0.5 * width, 0.5 * thickness],
                                                collisionFramePosition=[0.5 * jointLength, 0, 0])

        mass = actuatorMass / (2 * m)
        visualShapeId = -1
        basePosition = basePosition
        baseOrientation = baseOrientation

        link_Masses = [actuatorMass / m for i in range(N * m - 1)]
        link_Masses.append(actuatorMass / (2 * m))

        linkCollisionShapeIndices = [linkBoxId for i in range(N * m - 1)]
        linkCollisionShapeIndices.append(endBoxId)

        linkVisualShapeIndices = [-1 for i in range(N * m)]

        linkPositions = [[0.5 * jointLength, 0, 0]]
        for i in range(N * m - 1):
            linkPositions.append([linkLength, 0, 0])

        linkOrientations = [[0, 0, 0, 1] for i in range(N * m)]

        linkInertialFramePositions = [[0.5 * linkLength, 0, 0] for i in range(N * m - 1)]
        linkInertialFramePositions.append([0.5 * jointLength, 0, 0])

        linkInertialFrameOrientations = [[0, 0, 0, 1] for i in range(N * m)]
        indices = [i for i in range(N * m)]
        jointTypes = [self._p.JOINT_REVOLUTE for i in range(N * m)]

        axis = [[0, 1, 0] for i in range(N * m)]

        boxId = self._p.createMultiBody(mass,
                                        startBoxId,
                                        visualShapeId,
                                        basePosition,
                                        baseOrientation,
                                        linkMasses=link_Masses,
                                        linkCollisionShapeIndices=linkCollisionShapeIndices,
                                        linkVisualShapeIndices=linkVisualShapeIndices,
                                        linkPositions=linkPositions,
                                        linkOrientations=linkOrientations,
                                        linkInertialFramePositions=linkInertialFramePositions,
                                        linkInertialFrameOrientations=linkInertialFrameOrientations,
                                        linkParentIndices=indices,
                                        linkJointTypes=jointTypes,
                                        linkJointAxis=axis)

        jointNumber = self._p.getNumJoints(boxId)
        # Disable the default motors
        for joint in range(jointNumber):
            self._p.setJointMotorControl2(boxId,
                                          joint,
                                          self._p.VELOCITY_CONTROL,
                                          force=0)
        return [boxId, jointNumber]

    @staticmethod
    def generate_1d_motor_voltages(actuatorVoltages, actuatorNumber, unitMotorNumber):
        N = actuatorNumber
        m = unitMotorNumber
        motorVoltages = [actuatorVoltages[i] / m for i in range(N) for j in range(m)]
        return motorVoltages

    def voltage_torque_control_step(self, boxId, actuatorVoltages, TorVolThe, N, m, jointNumber, jointIndex, linkIndex,
                                    jointLength):
        motorVoltages = self.generate_1d_motor_voltages(actuatorVoltages, N, m)
        theta = []
        angularVelocities = []
        positions = []
        positionVelocities = []
        jointStates = self._p.getJointStates(self.boxId, jointIndex)
        linkStates = self._p.getLinkStates(boxId, linkIndex, computeLinkVelocity=1)
        for joint in range(jointNumber):
            theta.append(jointStates[joint][0])
            angularVelocities.append(jointStates[joint][1])
            positions.append(linkStates[joint][4])
            positionVelocities.append(linkStates[joint][6])
        positions, positionVelocities = self.get_positions_and_velocities(positions, positionVelocities, boxId,
                                                                          jointNumber,
                                                                          jointLength)
        Tor = [TorVolThe(theta[joint], angularVelocities[joint], motorVoltages[joint]) for joint in range(jointNumber)]
        self._p.setJointMotorControlArray(boxId,
                                          jointIndex,
                                          self._p.TORQUE_CONTROL,
                                          forces=Tor)
        
        if self._p is None:
            print("PyBullet client is not initialized.")
        return [theta, angularVelocities, positions, motorVoltages, Tor, positionVelocities]

    def start_point(self, boxId, jointLength):
        base_state = self._p.getBasePositionAndOrientation(boxId)
        base_position = base_state[0]
        base_orientation = base_state[1]
        base_velocity_state = self._p.getBaseVelocity(boxId)
        base_position_velocity = base_velocity_state[0]
        base_orientation_velocity = base_velocity_state[1]
        _, pitch, _ = self._p.getEulerFromQuaternion(base_orientation)
        _, pitch_velocity, _ = base_orientation_velocity
        length = jointLength / 2.0
        angle = math.pi - pitch
        angle_velocity = -pitch_velocity
        start_position = self.position_transform(base_position, length, angle)
        start_position_velocity = self.velocity_transform(base_position_velocity, length, angle, angle_velocity)
        return start_position, start_position_velocity

    def end_point(self, boxId, jointNumber, jointLength):
        state = self._p.getLinkState(boxId, jointNumber - 1, computeLinkVelocity=1)
        position = state[4]
        orientation = state[5]
        position_velocity = state[6]
        orientation_velocity = state[7]
        _, pitch, _ = self._p.getEulerFromQuaternion(orientation)
        _, pitch_velocity, _ = orientation_velocity
        length = jointLength
        angle = -pitch
        angle_velocity = -pitch_velocity
        end_position = self.position_transform(position, length, angle)
        end_position_velocity = self.velocity_transform(position_velocity, length, angle, angle_velocity)
        return end_position, end_position_velocity

    @staticmethod
    def position_transform(position, length, angle):
        return (position[0] + length * math.cos(angle),
                0.0,
                position[2] + length * math.sin(angle))

    @staticmethod
    def velocity_transform(position_velocity, length, angle, angle_velocity):
        return (position_velocity[0] - length * math.sin(angle) * angle_velocity,
                0.0,
                position_velocity[2] + length * math.cos(angle) * angle_velocity)

    def get_positions_and_velocities(self, positions, positionVelocities, boxId, jointNumber, jointLength):
        start_position, start_position_velocity = self.start_point(boxId, jointLength)
        end_position, end_position_velocity = self.end_point(boxId, jointNumber, jointLength)
        positions.insert(0, start_position)
        positions.append(end_position)
        positionVelocities.insert(0, start_position_velocity)
        positionVelocities.append(end_position_velocity)
        return positions, positionVelocities
    

from copy import deepcopy
from multiprocessing import Pool
import tempfile
from pybullet_utils import bullet_client
import pybullet
import math
import numpy as np
import datetime

class RobotBase(modelParameter):
    def __init__(self, parameter='trimorph parameter 2'):
        modelParameter.__init__(self, parameter=parameter)
        self._p = p
        self.connection_id = self._p.connect(self._p.DIRECT)  # or self._p.DIRECT for non-graphical mode
        if self.connection_id < 0:
            raise RuntimeError("Failed to connect to PyBullet!")

    def create_1d_multi_actuators(self,
                                  actuatorNumber,
                                  unitMotorNumber,
                                  actuatorMass,
                                  actuatorLength,
                                  actuatorWidth,
                                  actuatorThickness,
                                  basePosition=(0, 0, 0),
                                  baseOrientation=(0, 0, 0, 1)):
        N = actuatorNumber
        m = unitMotorNumber
        thickness = actuatorThickness
        width = actuatorWidth
        linkLength = actuatorLength / m
        jointLength = 0.5 * linkLength

        startBoxId = self._p.createCollisionShape(self._p.GEOM_BOX,
                                                  halfExtents=[0.5 * jointLength, 0.5 * width, 0.5 * thickness])
        linkBoxId = self._p.createCollisionShape(self._p.GEOM_BOX,
                                                 halfExtents=[0.5 * linkLength, 0.5 * width, 0.5 * thickness],
                                                 collisionFramePosition=[0.5 * linkLength, 0, 0])
        endBoxId = self._p.createCollisionShape(self._p.GEOM_BOX,
                                                halfExtents=[0.5 * jointLength, 0.5 * width, 0.5 * thickness],
                                                collisionFramePosition=[0.5 * jointLength, 0, 0])

        mass = actuatorMass / (2 * m)
        visualShapeId = -1
        basePosition = basePosition
        baseOrientation = baseOrientation

        link_Masses = [actuatorMass / m for i in range(N * m - 1)]
        link_Masses.append(actuatorMass / (2 * m))

        linkCollisionShapeIndices = [linkBoxId for i in range(N * m - 1)]
        linkCollisionShapeIndices.append(endBoxId)

        linkVisualShapeIndices = [-1 for i in range(N * m)]

        linkPositions = [[0.5 * jointLength, 0, 0]]
        for i in range(N * m - 1):
            linkPositions.append([linkLength, 0, 0])

        linkOrientations = [[0, 0, 0, 1] for i in range(N * m)]

        linkInertialFramePositions = [[0.5 * linkLength, 0, 0] for i in range(N * m - 1)]
        linkInertialFramePositions.append([0.5 * jointLength, 0, 0])

        linkInertialFrameOrientations = [[0, 0, 0, 1] for i in range(N * m)]
        indices = [i for i in range(N * m)]
        jointTypes = [self._p.JOINT_REVOLUTE for i in range(N * m)]

        axis = [[0, 1, 0] for i in range(N * m)]

        boxId = self._p.createMultiBody(mass,
                                        startBoxId,
                                        visualShapeId,
                                        basePosition,
                                        baseOrientation,
                                        linkMasses=link_Masses,
                                        linkCollisionShapeIndices=linkCollisionShapeIndices,
                                        linkVisualShapeIndices=linkVisualShapeIndices,
                                        linkPositions=linkPositions,
                                        linkOrientations=linkOrientations,
                                        linkInertialFramePositions=linkInertialFramePositions,
                                        linkInertialFrameOrientations=linkInertialFrameOrientations,
                                        linkParentIndices=indices,
                                        linkJointTypes=jointTypes,
                                        linkJointAxis=axis)

        jointNumber = self._p.getNumJoints(boxId)
        # Disable the default motors
        for joint in range(jointNumber):
            self._p.setJointMotorControl2(boxId,
                                          joint,
                                          self._p.VELOCITY_CONTROL,
                                          force=0)
        return [boxId, jointNumber]

    @staticmethod
    def generate_1d_motor_voltages(actuatorVoltages, actuatorNumber, unitMotorNumber):
        N = actuatorNumber
        m = unitMotorNumber
        motorVoltages = [actuatorVoltages[i] / m for i in range(N) for j in range(m)]
        return motorVoltages

    def voltage_torque_control_step(self, boxId, actuatorVoltages, TorVolThe, N, m, jointNumber, jointIndex, linkIndex,
                                    jointLength):
        motorVoltages = self.generate_1d_motor_voltages(actuatorVoltages, N, m)
        theta = []
        angularVelocities = []
        positions = []
        positionVelocities = []
        jointStates = self._p.getJointStates(self.boxId, self.jointIndex)
        linkStates = self._p.getLinkStates(boxId, linkIndex, computeLinkVelocity=1)
        for joint in range(jointNumber):
            theta.append(jointStates[joint][0])
            angularVelocities.append(jointStates[joint][1])
            positions.append(linkStates[joint][4])
            positionVelocities.append(linkStates[joint][6])
        positions, positionVelocities = self.get_positions_and_velocities(positions, positionVelocities, boxId,
                                                                          jointNumber,
                                                                          jointLength)
        Tor = [TorVolThe(theta[joint], angularVelocities[joint], motorVoltages[joint]) for joint in range(jointNumber)]
        self._p.setJointMotorControlArray(boxId,
                                          jointIndex,
                                          self._p.TORQUE_CONTROL,
                                          forces=Tor)
        return [theta, angularVelocities, positions, motorVoltages, Tor, positionVelocities]

    def start_point(self, boxId, jointLength):
        base_state = self._p.getBasePositionAndOrientation(boxId)
        base_position = base_state[0]
        base_orientation = base_state[1]
        base_velocity_state = self._p.getBaseVelocity(boxId)
        base_position_velocity = base_velocity_state[0]
        base_orientation_velocity = base_velocity_state[1]
        _, pitch, _ = self._p.getEulerFromQuaternion(base_orientation)
        _, pitch_velocity, _ = base_orientation_velocity
        length = jointLength / 2.0
        angle = math.pi - pitch
        angle_velocity = -pitch_velocity
        start_position = self.position_transform(base_position, length, angle)
        start_position_velocity = self.velocity_transform(base_position_velocity, length, angle, angle_velocity)
        return start_position, start_position_velocity

    def end_point(self, boxId, jointNumber, jointLength):
        state = self._p.getLinkState(boxId, jointNumber - 1, computeLinkVelocity=1)
        position = state[4]
        orientation = state[5]
        position_velocity = state[6]
        orientation_velocity = state[7]
        _, pitch, _ = self._p.getEulerFromQuaternion(orientation)
        _, pitch_velocity, _ = orientation_velocity
        length = jointLength
        angle = -pitch
        angle_velocity = -pitch_velocity
        end_position = self.position_transform(position, length, angle)
        end_position_velocity = self.velocity_transform(position_velocity, length, angle, angle_velocity)
        return end_position, end_position_velocity

    @staticmethod
    def position_transform(position, length, angle):
        return (position[0] + length * math.cos(angle),
                0.0,
                position[2] + length * math.sin(angle))

    @staticmethod
    def velocity_transform(position_velocity, length, angle, angle_velocity):
        return (position_velocity[0] - length * math.sin(angle) * angle_velocity,
                0.0,
                position_velocity[2] + length * math.cos(angle) * angle_velocity)

    def get_positions_and_velocities(self, positions, positionVelocities, boxId, jointNumber, jointLength):
        start_position, start_position_velocity = self.start_point(boxId, jointLength)
        end_position, end_position_velocity = self.end_point(boxId, jointNumber, jointLength)
        positions.insert(0, start_position)
        positions.append(end_position)
        positionVelocities.insert(0, start_position_velocity)
        positionVelocities.append(end_position_velocity)
        return positions, positionVelocities
    
class RobotSimulator(RobotBase):
    def __init__(self, parameter='trimorph parameter 2'):
        RobotBase.__init__(self, parameter=parameter)
        self.step = 0
        self.simTime = 0
        self.thickness = 0.1
        self.isGravity = True
        self.dateTime = datetime.datetime.today().strftime('%m_%d_%Y_%H_%M')
        self.N = None
        self.m = None
        self.drivenFrequency = None
        self.dampingEta = None
        self.recordStepInterval = None
        self.simCycles = None
        self.timeStep = None
        self.dataTime = None
        self.dataTheta = None
        self.dataAngularVelocities = None
        self.dataTor = None
        self.dataPositions = None
        self.dataMotorVoltages = None
        self.dataPositionVelocities = None
        self.linkLength = None
        self.jointLength = None
        self.jointIndex = None
        self.linkIndex = None
        self.boxId = None
        self.jointNumber = None
        self.theta = None
        self.angularVelocities = None
        self.positions = None
        self.motorVoltages = None
        self.Tor = None
        self.positionVelocities = None
        self.voltage = None
        self.filename = None
      

    def reset(self):
        RobotBase.reset(self)
        self.step = 0
        self.dataTime = []
        self.dataTheta = []
        self.dataAngularVelocities = []
        self.dataTor = []
        self.dataPositions = []
        self.dataMotorVoltages = []
        self.dataPositionVelocities = []
        self.linkLength = self.actuatorLength / self.m
        self.jointLength = 0.5 * self.linkLength
        self._p = None
        self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._p.resetSimulation()
        self._p.resetDebugVisualizerCamera(self.N * 15, -360, -16, [0, 0, 1])
        (self.boxId, self.jointNumber) = self.create_1d_multi_actuators(self.N, self.m, self.actuatorMass,
                                                                        self.actuatorLength,
                                                                        self.width, self.thickness)
        self.jointIndex = range(self.jointNumber)
        self.linkIndex = range(self.jointNumber + 1)

        self._p.setTimeStep(self.timeStep)
        if self.isGravity:
            self._p.setGravity(0, 0, -9.81)
        else:
            self._p.setGravity(0, 0, 0)

    def close(self):
        self._p.disconnect()

    def tor_vol_the(self, Theta, angularVelocity, Voltage):
        thetaTarget = -self.beta * Voltage
        K = self.halfEI / self.jointLength
        omega = 2 * math.pi * self.drivenFrequency
        Tor = -K * ((Theta - thetaTarget) + (self.dampingEta / omega) * angularVelocity)
        print(f"tor_vol_the: Theta={Theta}, angularVelocity={angularVelocity}, Voltage={Voltage}, Tor={Tor}")
        return Tor

    def print_progress(self):
        print('N=', self.N, ', m=', self.m, "Frequency=", round(self.drivenFrequency, 2), "Hz, Actuator length",
              self.actuatorLength,
              "cm, step=", self.step + 1)

    def sim_step(self, actuatorVoltages):
        if (self.step + 1) % 100000 == 0:
            self.print_progress()

        self.simTime = self.step * self.timeStep
        [self.theta, self.angularVelocities, self.positions, self.motorVoltages, self.Tor,
         self.positionVelocities] = self.voltage_torque_control_step(
            self.boxId,
            actuatorVoltages,
            self.tor_vol_the,
            self.N,
            self.m,
            self.jointNumber,
            self.jointIndex,
            self.linkIndex,
            self.jointLength)
        if (self.step + 1) % self.recordStepInterval == 0 or self.step == 0 or self.step == self.simCycles - 1:
            self.dataTime.append(self.simTime)
            self.dataTheta.append(self.theta)
            self.dataAngularVelocities.append(self.angularVelocities)
            self.dataPositions.append(self.positions)
            self.dataMotorVoltages.append(self.motorVoltages)
            self.dataTor.append(self.Tor)
            self.dataPositionVelocities.append(self.positionVelocities)
        self._p.stepSimulation()
        self.step = self.step + 1

    def get_positions(self, positions):
        jointLength = self.jointLength / 100.0
        offset = jointLength / 2.0
        scaled_positions = []
        for pos in positions:
            x = pos[0] / 100.0
            y = pos[1] / 100.0
            z = pos[2] / 100.0
            scaled_pos = (x + offset, y, z)
            scaled_positions.append(scaled_pos)
        return scaled_positions

    def get_shape(self, positions):
        positions = self.get_positions(positions)
        xAxis = np.array([positions[i][0] for i in range(len(positions))])
        zAxis = np.array([positions[i][2] for i in range(len(positions))])
        return xAxis, zAxis

    @staticmethod
    def get_shape_velocity(positionVelocities):
        xAxisVelocity = np.array([positionVelocities[i][0] for i in range(len(positionVelocities))])
        zAxisVelocity = np.array([positionVelocities[i][2] for i in range(len(positionVelocities))])
        return xAxisVelocity, zAxisVelocity

    def shape(self):
        positions = self.positions
        xAxis, zAxis = self.get_shape(positions)
        return xAxis, zAxis

    def shape_velocity(self):
        positionVelocities = self.positionVelocities
        xAxisVelocity, zAxisVelocity = self.get_shape_velocity(positionVelocities)
        return xAxisVelocity, zAxisVelocity

    def position_offset(self, positions):
        jointLength = self.jointLength
        offset = jointLength / 2.0
        for i in range(len(positions)):
            positions[i] = self.position_transform(positions[i], offset, angle=0.0)
        return positions

    def save_simulator_state(self):
        state_file = tempfile.NamedTemporaryFile(delete=False)
        self._p.saveBullet(bulletFileName=state_file.name)
        state_file.close()
        return state_file

    def load_simulator_state(self, state_file_name):
        self._p.restoreState(fileName=state_file_name)

    def copy(self, state_file_name):
        simulator = deepcopy(self)
        simulator.reset()
        simulator.load_simulator_state(state_file_name)
        return simulator

    def save(self):
        raise NotImplementedError

    def drive_cooldown(self, saving=True, closing=True):
        if closing:
            self.close()
        if not saving:
            return None
        self.save()
        return self.filename

    def get_phase_velocity(self, frequency):
        angularFrequency = 2 * np.pi * frequency
        massLoad = self.load / self.gravity
        return np.power(self.EI * (angularFrequency ** 2) / massLoad, 0.25)
    
    def generate_inchworm_voltages(self, t, frequency, amplitude=1.0):
        N = self.N  # number of actuators
        period = 1.0 / frequency
        step_time = period / N
        voltages = [0.0] * N
        phase_index = int((t % period) / step_time)

        voltages[phase_index % N] = amplitude
        voltages[(phase_index + 1) % N] = -amplitude
        return voltages
    
    def lock_joint_friction(self, jointIndex, enable = True):
        friction = 1000.0 if enable else 0.1
        self._p.changeDynamics(self.boxId, jointIndex, lateralFriction=friction)
        print(f"Joint {jointIndex} friction {'locked' if enable else 'unlocked'}.")

    
    def generate_double_stride_gait_voltage(self, t):
        N = self.N
        frequency = self.drivenFrequency
        period = 1.0 / frequency
        phase_duration = period / 8.0
        phase_index = int((t % period) / phase_duration)

        voltages = [0.0] * N

        if phase_index == 0:
            voltages[0] = 1.0
            voltages[1] = 1.0
            voltages[3] = -1.0
            voltages[4] = -1.0
            self.lock_joint_friction(0, True)
            self.lock_joint_friction(1, False)
            self.lock_joint_friction(2, False)
            self.lock_joint_friction(3, True)
            self.lock_joint_friction(4, True)
            self.lock_joint_friction(5, True)

        elif phase_index == 1:
            self.lock_joint_friction(1, False)
            self.lock_joint_friction(2, False)

        elif phase_index == 2:
            self.lock_joint_friction(0, False)
            self.lock_joint_friction(3, False)
            self.lock_joint_friction(2, True)
        
        elif phase_index == 3:
            voltages[0] = -1.0
            voltages[1] = -1.0
            voltages[2] = 1.0

        elif phase_index == 4:
            self.lock_joint_friction(0, True)
            self.lock_joint_friction(5, True)
            self.lock_joint_friction(2, False)
        
        elif phase_index == 5:
            voltages[0] = 1.0
            voltages[1] = 1.0
            voltages[2] = -1.0
            voltages[4]= -1.0

        elif phase_index == 6:
            self.lock_joint_friction(1, False)
            self.lock_joint_friction(3, False)
            self.lock_joint_friction(4, False)

        elif phase_index == 7:
            self.lock_joint_friction(2, True)

        return voltages
        

import matplotlib.pyplot as plt
import numpy as np
import time

# Initialize simulator
sim = RobotSimulator(parameter='trimorph parameter 2')

# Set simulation parameters
sim.N = 5              # Number of actuators
sim.m = 3              # Motors per actuator
sim.timeStep = 0.001    # Seconds
sim.simCycles = 50000    # Total steps
sim.drivenFrequency = 0.5  # Hz
sim.dampingEta = 1.0   # Damping factor
sim.recordStepInterval = 10  # Save every N steps


# Reset simulator
sim.reset()

# Prepare figure
plt.figure(figsize=(10, 6))
plt.title("Inchworm Crawling Motion (X-Z Plot)")
plt.xlabel("X Position")
plt.ylabel("Z Position")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label ='Ground')
plt.legend()

# Run simulation and update plot
for step in range(sim.simCycles):
    t = step * sim.timeStep
    voltages = sim.generate_double_stride_gait_voltage(t)
    sim.sim_step(voltages)

    if step % sim.recordStepInterval == 0 or step == sim.simCycles - 1:
        xAxis, zAxis = sim.shape()
        
        plt.cla()  # Clear previous frame
        plt.plot(xAxis, zAxis, 'o-', label=f"Time: {t:.2f}s")
        plt.xlim(np.min(xAxis) - 0.1, np.max(xAxis) + 0.1)
        plt.ylim(-300, 30)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label ='Ground')
        plt.legend()
        plt.pause(0.01)  # Update the plot
        time.sleep(0.01)

plt.show()